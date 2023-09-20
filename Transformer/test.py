import torch
from torch import nn

a = torch.tensor([[1, 2],
                  [3, 4]])
b = torch.tensor([2, 2])
y = a + b

print(y)
