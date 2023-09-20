import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 注意下面代码的计算方式与公式中给出的是不同的，但是是等价的，你可以尝试简单推导证明一下。
        # 这样计算是为了避免中间的数值计算结果超出float的范围，
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        print(pe.shape)
        # 注册一个不需要求导的张量，使其成为模型的一部分。在模型保存和加载过程中，这些张量也会被保存和加载。
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


if __name__ == '__main__':
    batch_size = 32
    seq_length = 50
    vocab_size = 10000
    embedding_dim = 512

    # 创建一个形状为 (batch_size, seq_length) 的输入张量
    input = torch.randint(0, vocab_size, (batch_size, seq_length))

    # 创建一个 Embedding 层
    embedding = nn.Embedding(vocab_size, embedding_dim)

    # 对输入张量进行 Embedding 操作
    x_embedded = embedding(input)

    # 检查 Embedding 层的输出
    print("Embedding 层的输出形状：", x_embedded.shape)

    PE = PositionalEncoding(d_model=512, dropout=0.1)
    output = PE(x_embedded)
    print(output)
    print(output.size())
