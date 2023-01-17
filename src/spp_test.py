from torch import nn
import torch


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=7, stride=1, padding=7 // 2)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        return torch.cat([x, x1, x2, x3], dim=1)


x = torch.rand((2, 512, 13, 13))
f = SPP()
print(f(x).shape)
