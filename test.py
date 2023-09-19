import torch
from torch import nn

model = nn.RNN(input_size=32, hidden_size=64, num_layers=3, nonlinearity='relu', batch_first=True)
x = torch.randn(64, 2, 32)
y = torch.randn(3, 64, 64)
z1, z2 = model(x, y)
print(z1)
print(z2)
print(z1
      .shape)
print(z2.shape)

