import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        pass


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self):
        pass


class Trainer():
    def __init__(self):
        super(Trainer, self).__init__()

    def train(self):
        pass


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    