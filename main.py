# Source:
# wget https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py

from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import AADL


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        layers = [
            nn.Conv2d(1, 128, 3, 1, padding="same"),
            nn.ReLU()
        ]
        for i in range(0, 5):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, 1, padding="same"),
                    nn.ReLU()
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.reduction = nn.Sequential(
            nn.Conv2d(128, 10, 3, 1, padding="same"),
            nn.AdaptiveMaxPool2d(1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        x = self.reduction(x)
        x = torch.flatten(x, 1)
        x = F.log_softmax(x, dim=1)
        return x


train_step = 0
def train(args, model, device, train_loader, optimizer, epoch, writer):
    global train_step
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)
            train_step += 1
            if args.dry_run:
                break


test_step = 0
def test(model, device, test_loader, writer):
    global test_step
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    writer.add_scalar("test_loss_avg", test_loss, test_step)
    test_step += 1


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--AADL_gpu', action='store_true', default=False,
                        help='Use AADL gpu')

    parser.add_argument('--AADL_cpu', action='store_true', default=False,
                        help='Use AADL cpu')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    assert not (args.AADL_gpu and args.AADL_cpu), "Should not set both AADL_gpu and AADL_cpu"

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 8,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    print(f"Device: {device}")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('dataset', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('dataset', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_dir = f"runs/{device}{int(time.time())}"

    if args.AADL_gpu or args.AADL_cpu:
        # Parameters for Anderson acceleration
        relaxation = 0.5
        wait_iterations = 0
        history_depth = 10
        store_each_nth = 10
        frequency = store_each_nth
        reg_acc = 0.0
        average = True
        AADL_device = "cpu"
        if args.AADL_gpu:
            AADL_device = "cuda"
        AADL.accelerate(
            optimizer, "anderson", relaxation, wait_iterations, history_depth, store_each_nth, frequency, reg_acc, average,
            compute_device=AADL_device,
            history_device=AADL_device)
        log_dir = f"runs/AADL_{AADL_device}{int(time.time())}"
        print(f"AADL Device: {AADL_device}")

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(model, device, test_loader, writer)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
