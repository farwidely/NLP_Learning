import torch.nn as nn
import torch.nn.functional as F
import torch

batch_size = 1


class TextSentiment(nn.Module):
    # 文本分类模型
    # vocab_size是词表大小，embed_dim是嵌入维度，num_class是类别数量
    def __init__(self, embed_dim):
        super(TextSentiment, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(30522, embed_dim)

        # 平均池化层
        self.pool = nn.AvgPool1d(kernel_size=32)

        self.flatten = nn.Flatten()

        # 全连接层
        self.fc = nn.Linear(379, 4)

    def forward(self, x):
        # 词嵌入
        x = self.embedding(x)
        # 平均池化
        x = self.pool(x)
        x = self.flatten(x)

        # 全连接层
        x = self.fc(x)

        return x


if __name__ == '__main__':
    model = TextSentiment(vocab_size=25485, embed_dim=32)
    x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]])
    x1 = torch.randint(0, 1000, (64, 379))
    print(x1)
    print("输入尺寸：", x1.shape)
    y = model(x1)
    print(y)
    print(y.shape)
