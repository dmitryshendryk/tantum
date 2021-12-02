import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class Attention(nn.Module):
    def __init__(self, mem_in=32, query_in=32, key_size=32, output_size=32):
        super(Attention, self).__init__()
        self.key = nn.Conv1d(mem_in, key_size, 1, padding=0)
        self.value = nn.Conv1d(mem_in, output_size, 1, padding=0)
        self.query = nn.Conv1d(query_in, key_size, 1, padding=0)
        self.key_size = key_size

    def forward(self, x1, x2):
        queries = self.query(x1)  # Batch x Values x Keys
        keys = self.key(x2)  # Batch x Keysize x Keys
        values = self.value(x2)  # Batch x Values x Keys
        u = torch.sum(queries.unsqueeze(2) * keys.unsqueeze(3), 1)/np.sqrt(self.key_size)
        w = F.softmax(u, dim=1)
        out = torch.sum(w.unsqueeze(1) * values.unsqueeze(3), 2)
        return out, w

class MultiHeadAttention(nn.Module):
    def __init__(self, mem_in=32, query_in=32, key_size=32, output_size=32, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.layers = nn.ModuleList([Attention(mem_in, query_in, key_size, output_size) for i in range(num_heads)])
        self.proj_down = nn.Conv1d(num_heads*output_size, query_in, 1, padding=0)
        self.mixing_layer1 = nn.Conv1d(query_in, query_in, 1, padding=0)
        self.mixing_layer2 = nn.Conv1d(query_in, query_in, 1, padding=0)
        self.norm1 = nn.LayerNorm(query_in)
        self.norm2 = nn.LayerNorm(query_in)

    def forward(self, query, context):
        x1 = query.reshape(query.size(0), query.size(1), -1)
        x2 = context.reshape(context.size(0), context.size(1), -1)

        # Apply attention for each head
        z1, ws = [], []
        for i in range(len(self.layers)):
            z, w = self.layers[i](x1, x2)
            z1.append(z)
            ws.append(w)
        z1 = torch.cat(z1, 1)

        # Project down. Layer norm is a bit fiddly here - it wants the dimensions to normalize over to be the last dimensions
        z2 = self.norm1((self.proj_down(z1) + x2).transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # Mixing layer
        z3 = self.norm2((self.mixing_layer2(F.relu(self.mixing_layer1(
            z2))) + z2).transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        if len(query.size()) == 4:
            z3 = z3.reshape(query.size(0), query.size(1), query.size(3), query.size(3))        

        return z3, z1
        

# query = torch.rand(128, 256, 24, 24)
# query_ = torch.reshape(query, (query.size(0), -1 , query.size(1)))
# print(query_.shape)


# multihead_attn = MultiHeadAttention(n_head=8, d_model=256, d_k=32, d_v=32)
# attn_output, attn_weights = multihead_attn(query_, query_, query_)
# attn_output = attn_output.reshape(*list(query.size()))
# print(f'attn_output: {attn_output.size()}, attn_weights: {attn_weights.size()}')