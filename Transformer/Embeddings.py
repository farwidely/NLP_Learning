import math
import torch
from torch import nn


# 定义Embedding层
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        # 调用torch.nn.Embedding(), vocab为word容量，d_model为用于表示单个字母的向量的位数，原文中设定为512
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embedded = self.embedding(x)
        return embedded * math.sqrt(self.d_model)


# 测试Embedding层
if __name__ == '__main__':
    batch_size = 32
    seq_length = 50
    vocab_size = 10000
    embedding_dim = 512

    # 创建一个形状为 (batch_size, seq_length) 的输入张量
    input = torch.randint(0, vocab_size, (batch_size, seq_length))
    print(input.size())

    embedding = Embeddings(vocab_size, embedding_dim)

    output = embedding(input)
    print(output)
    print(output.size())
