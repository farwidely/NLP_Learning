import torch.nn as nn
import torch.nn.functional as F


batch_size = 64

class TextSentiment(nn.Module):
    # 文本分类模型
    # vocab_size是词表大小，embed_dim是嵌入维度，num_class是类别数量
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextSentiment, self).__init__()
        # sparse=True表示使用稀疏张量进行嵌入,该层求解梯度时只更新部分权重
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        # 指定初始化权重的取值范围
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        # text是经过数值映射后的文本
        embedded = self.embedding(text)
        # 将形状为(m，embed_dim)的embedded转化为(batch_size, embedded)，其中m是一个batch_size中文本的不同的词汇数量
        # 用m除以batch_size，m中共包含c个batch_size
        c = embedded.size(0) // batch_size
        # 截取embedded
        embedded = embedded[:batch_size*c]
        # 进行平均池化，因为avg_pool作用在行内，需要转化为作用在行之间
        embedded = embedded.transpose(1, 0).unsqueeze(0)
        embedded = F.avg_pool1d(embedded, kernel_size=c)
        embedded = embedded[0].transpose(1, 0)
        return self.fc(embedded)

