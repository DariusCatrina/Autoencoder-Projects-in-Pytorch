import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import numpy as np

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

train_MNIST = datasets.MNIST(root='./', train=True, download=True, transform=transform)
test_MNIST = datasets.MNIST(root='./', train=False, download=True, transform=transform)


train_loader = torch.utils.data.DataLoader(train_MNIST,batch_size=16)
test_loader = torch.utils.data.DataLoader(test_MNIST,batch_size=16)