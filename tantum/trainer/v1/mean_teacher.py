from __future__ import print_function
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms


class CFG:
    batch_size = 64
    test_batch_size = 100
    epochs = 10
    lr = 0.01
    momentum = 0.05
    seed = 1
    log_interval = 100
    device='GPU'


class Net(nn.Module):

    def __init__(self):
        
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, mean_teacher, device, train_loader, test_loader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        with torch.no_grad():
            mean_t_output = mean_teacher(data)
        

        const_loss = F.mse_loss(output, mean_t_output)

        weight = 0.2
        loss = F.nll_loss(output, target) + weight * const_loss
        loss.backward()
        optimizer.step()

        alpha = 0.95
        for mean_param, param in zip(mean_teacher.parameters(), model.parameters()):
            mean_param.data.mul_(alpha).add_(1-alpha, param.data)
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            test(args, model, device, test_loader, 'original')
            test(args, mean_teacher, device, test_loader, 'mean_teacher')

def test(args, model, device, test_loader, model_type):
    
    model.eval()
    test_loss = 0 
    correct = 0 

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {} {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        model_type, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    model.train()


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(CFG.seed)
    kwargs = {'num_workers': 0, 'pin_memory': True} if CFG.device == 'GPU' else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=CFG.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=CFG.test_batch_size, shuffle=True, **kwargs)
    
    model = Net().to(device)
    mean_teacher = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=CFG.lr, momentum=CFG.momentum)

    for epoch in range(1, CFG.epochs + 1):
        train(CFG, model, mean_teacher, device, train_loader, test_loader, optimizer, epoch)

    if (CFG.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
if __name__ == '__main__':
    main()