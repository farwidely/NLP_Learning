import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN1(nn.Module):
    # 使用公式法实现RNN，搭建AG_NEWS数据集文本分类模型，该数据集的句子进行补零后每个句子都很长，输入至RNN会导致梯度爆炸
    def __init__(self, input_size, hidden_size, num_layer=1, nonlinearity='relu'):
        super(RNN1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.embedding = nn.Embedding(30522, input_size)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layer, nonlinearity=nonlinearity, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.randn(self.num_layer, x.shape[0], self.hidden_size).to(device)
        x, _ = self.rnn(x, h0)
        # print(x.shape)
        # print(x[:, -1, :].shape)
        x = self.fc(x[:, -1, :])
        return x


if __name__ == '__main__':
    input = torch.randint(0, 30522, (64, 379))
    print(input.shape)
    model = RNN1(input_size=32, hidden_size=512)
    output = model(input)
    print(output)
    print(output.shape)
