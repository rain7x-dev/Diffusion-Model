import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

class DoubleConv(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_ch, out_ch, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace = True)
    )

  def forward(self , x):
    return self.layers(x)

class Down(nn.Module):
  def __init__(self , in_ch , out_ch):
    super().__init__()
    self.down = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_ch , out_ch)
    )

  def forward(self , x):
    return self.down(x)

class Up(nn.Module):
  def __init__(self , in_ch , out_ch):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_ch , in_ch//2 , kernel_size = 2, stride = 2)
    self.conv = DoubleConv(in_ch , out_ch)

  def forward(self , x , skip):
    x = self.up(x)
    x = torch.cat([skip , x] , dim = 1)
    return self.conv(x)
