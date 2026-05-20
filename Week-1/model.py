import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.Linear(784, 128)
        self.l3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
