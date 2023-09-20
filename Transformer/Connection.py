import copy
from torch import nn

from Norm import LayerNorm


def clones(module, N):
    """
    复制编码层，原文中N=6
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    实现子层连接结构的类（Transformer_Learning.jpg中的每一个“Add & Norm”与其前面的小方框的连接）
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        sublayer_out = sublayer(x)
        x_norm = self.norm(x + self.dropout(sublayer_out))
        return x_norm
