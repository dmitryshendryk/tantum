import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import IPython.display as display
from torchvision.datasets import STL10

import sys

PROJECT_ROOT = '/Users/dmitry/Documents/Personal/study/kaggle/tantum'
sys.path.append(PROJECT_ROOT)

from tantum.model.attention.multihead import MultiHeadAttention

device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
print(device)

BATCH_SIZE = 100
dataset = STL10("stl10", split='train', download=True)
def getBatch(BS=10, offset=0, display_labels=False):
    xs = []
    labels = []
    for i in range(BS):
        x, y = dataset[offset + i]
        x = (np.array(x)-128.0)/128.0
        x = x.transpose(2, 0, 1)
        
        np.random.seed(i + 10)
        corrupt = np.random.randint(2)
        if corrupt:  # To corrupt the image, we'll just copy a patch from somewhere else
            pos_x = np.random.randint(96-16)
            pos_y = np.random.randint(96-16)
            x[:, pos_x:pos_x+16, pos_y:pos_y+16] = 1
        xs.append(x)
        labels.append(corrupt)

    if display_labels == True:
        print(labels)

    return np.array(xs), np.array(labels)


class ConvPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1a = nn.Conv2d(3, 32, 5, padding=2)
        self.p1 = nn.MaxPool2d(2)
        self.c2a = nn.Conv2d(32, 32, 5, padding=2)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(32, 32, 5, padding=2)
        self.bn1a = nn.BatchNorm2d(32)
        self.bn2a = nn.BatchNorm2d(32)

    def forward(self, x):
        z = self.bn1a(F.leaky_relu(self.c1a(x)))
        z = self.p1(z)
        z = self.bn2a(F.leaky_relu(self.c2a(z)))
        z = self.p2(z)
        z = self.c3(z)
        return z
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvPart()
        self.final = nn.Linear(32, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        z = self.conv(x)
        z = z.mean(3).mean(2)
        p = torch.sigmoid(self.final(z))[:, 0]
        return p, ''

class NetMultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvPart()
        self.attn1 = MultiHeadAttention(n_head=4, d_model=32, d_k=8, d_v=8)
        self.final = nn.Linear(32, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        z = self.conv(x)
        q = torch.reshape(z, (z.size(0), -1 , z.size(1)))
        q, w = self.attn1(q, q, q)
        q = torch.reshape(q, (z.size(0), z.size(1), z.size(2), z.size(3)))
        z = q.mean(3).mean(2)
        p = torch.sigmoid(self.final(z))[:, 0]
        return p, q


def plot_without_attention(tr_err, ts_err, tr_acc, ts_acc, img):
    plt.clf()
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].plot(tr_err, label='tr_err')
    axs[0].plot(ts_err, label='ts_err')
    axs[0].legend()
    axs[1].plot(tr_acc, label='tr_err')
    axs[1].plot(ts_acc, label='ts_err')
    axs[1].legend()
    axs[2].axis('off')
    axs[3].axis('off')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.01)

def plot_with_attention(tr_err, ts_err, tr_acc, ts_acc, img, att_out, no_images=6):
    plt.clf()
    fig, axs = plt.subplots(1+no_images, 4, figsize=(20, (no_images+1)*5))
    axs[0, 0].plot(tr_err, label='tr_err')
    axs[0, 0].plot(ts_err, label='ts_err')
    axs[0, 0].legend()
    axs[0, 1].plot(tr_acc, label='tr_err')
    axs[0, 1].plot(ts_acc, label='ts_err')
    axs[0, 1].legend()
    axs[0, 2].axis('off')
    axs[0, 3].axis('off')
    for img_no in range(6):
        im = img[img_no].cpu().detach().numpy().transpose(1, 2, 0)*0.5 + 0.5
        axs[img_no+1, 0].imshow(im)
        for i in range(3):
            att_out_img = att_out[img_no, i+1].cpu().detach().numpy()
            axs[img_no+1, i+1].imshow(att_out_img)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.01)

def train(model, att_flag=False):
    net = model.to(device)
    tr_err, ts_err = [], []
    tr_acc, ts_acc = [], []
    for epoch in range(5):
        errs, accs = [], []
        net.train()
        for i in range(4000//BATCH_SIZE):
            net.optim.zero_grad()
            x, y = getBatch(BATCH_SIZE, i*BATCH_SIZE)
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).to(device)
            p, q = net.forward(x)
            loss = -torch.mean(y*torch.log(p+1e-8) + (1-y)*torch.log(1-p+1e-8))
            loss.backward()
            errs.append(loss.cpu().detach().item())
            pred = torch.round(p)
            accs.append(torch.sum(pred == y).cpu().detach().item()/BATCH_SIZE)
            net.optim.step()    
        tr_err.append(np.mean(errs))
        tr_acc.append(np.mean(accs))

        errs, accs = [], []
        net.eval()
        for i in range(1000//BATCH_SIZE):
            x, y = getBatch(BATCH_SIZE, i*BATCH_SIZE+4000)
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).to(device)
            p, q = net.forward(x)            
            loss = -torch.mean(y*torch.log(p+1e-8) + (1-y)*torch.log(1-p+1e-8))
            errs.append(loss.cpu().detach().item())
            pred = torch.round(p)
            accs.append(torch.sum(pred == y).cpu().detach().item()/BATCH_SIZE)
        ts_err.append(np.mean(errs))  
        ts_acc.append(np.mean(accs))
        
        if att_flag == False:
            plot_without_attention(tr_err, ts_err, tr_acc, ts_acc, x[0])
        else:
            plot_with_attention(tr_err, ts_err, tr_acc, ts_acc, x, q)
        
        print(f'Min train error: {np.min(tr_err)}')
        print(f'Min test error: {np.min(ts_err)}')



model = Net()
train(model, att_flag=False)