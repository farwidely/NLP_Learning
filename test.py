import torch
import torch.nn.functional as F
# 创建一个嵌入层
embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=6)

# 输入整数索引
indices = torch.tensor([1, 3, 5, 7])

# 获取嵌入向量表示
embedded = embedding(indices)
print(embedded)

embedded = embedded.transpose(1, 0).unsqueeze(0)
print(embedded)

embedded = F.avg_pool1d(embedded, kernel_size=2)
print(embedded)

embedded = embedded[0].transpose(1, 0)
print(embedded)