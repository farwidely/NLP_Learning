import torch
import torch.nn as nn


class GRU1(nn.Module):
    # 使用公式法实现GRU，搭建AG_NEWS数据集文本分类模型，该数据集的句子进行补零后每个句子都很长，输入至GRU1可能会导致梯度爆炸
    # 需使用合适的batch_size模型训练才能收敛，这里设置batch_size=64
    def __init__(self, input_size, hidden_size, num_layer=1):
        super(GRU1, self).__init__()
        self.input_size = input_size
        self.embedding = nn.Embedding(30522, input_size)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        # print(x.shape)
        # print(x[:, -1, :].shape)
        x = self.fc(x[:, -1, :])
        return x


if __name__ == '__main__':
    input = torch.randint(0, 30522, (64, 379))
    print(input.shape)
    model = GRU1(input_size=32, hidden_size=512)
    output = model(input)
    print(output)
    print(output.shape)
