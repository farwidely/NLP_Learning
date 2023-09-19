# import torch
# from torch import nn
#
# pool = nn.AvgPool1d(kernel_size=4)
# y = torch.randn(64, 379, 32)
# z = pool(y)
#
# print(z.shape)
import jieba
import numpy as np

# l1 = [1, 2, 3, 4]
# print(l1)
#
# np.save('l1', l1)
data = np.load('text_classification/agnews_number_dataset.npz')
print(type(data))
print(type(data['train_text']))
print(data['train_text'].shape)
