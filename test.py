import numpy as np

# 假设A和B是你要拼接的两个矩阵
A = np.random.rand(3, 4)
B = np.random.rand(3, 4)

print(A)
print(B)

# 使用numpy.concatenate
C = np.concatenate((A, B), axis=1)
print(C)
print(np.zeros((7, 10)))