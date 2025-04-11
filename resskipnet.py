
import torch as th
from torch import nn

class ResSkipNet(nn.Module):
    def __init__(
        self, dim,
        layers: list[nn.Module],
    ):
        super().__init__()
        self.norms = nn.ModuleList([ nn.GroupNorm(1, dim) for _ in layers ])
        self.layers = nn.ModuleList(layers)
        self.post_norm = nn.GroupNorm(1, dim)

    def forward( self, x, a ):
        o = th.zeros_like(x)
        for norm, layer in zip(self.norms, self.layers):
            res, skip = layer(norm(x), a).chunk(2, dim=1)
            x = x + res     # 继续参与训练的部分
            o = o + skip    # 不参与训练的部分，累加了每一轮的输出
        return self.post_norm(o)
