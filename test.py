import torch
from torch import nn

pool = nn.AvgPool1d(kernel_size=4)
y = torch.randn(64, 379, 32)
z = pool(y)

print(z.shape)
