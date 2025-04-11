import torch as torch
from torch import nn
from einops import rearrange

class Cond:
    def __init__(self, cond_dim, cond_proj_depth):
        self.cond_dim = cond_dim
        self.cond_proj_depth = cond_proj_depth
        self.cond_v = None

    def set_cond(self, cond_v):
        self.cond_v = cond_v

class ModConv(nn.Module):
    def __init__(
        self,
        cond: Cond,
        conv: nn.modules.conv._ConvNd,      # target conv
    ):
        super().__init__()

        self.conv = conv
        self.groups = self.conv.groups
        self.cond = cond

        self.proj_cond = nn.Sequential(
            *(
                block for _ in range(cond.cond_proj_depth)
                for block in [ nn.Linear(cond.cond_dim, cond.cond_dim), nn.SiLU() ]
            ),
            nn.Linear(cond.cond_dim, self.conv.weight.size(1)),
        )

    def forward(self, X):
        """
            :param    X: 输入特征 [B, in_cnl, ...]
            :return:     输出特征 [B, out_cnl, ...]
        """

        assert self.cond is not None

        cond_v = self.cond.cond_v
        B = cond_v.size(0)

        scale = self.proj_cond(cond_v)+1

        # bw[b,o,i,k] = weight[o,i,k] × scale[b,i]， cond_v 第 i 个通道的元素，和所有卷积核的第i个通道的所有元素相乘
        bw = torch.einsum('oi...,bi->boi...', self.conv.weight, scale) 

        # demod[b,o] = 1 / sqrt(sum(bw[b,o,i,k...]^2) + eps) 每一个输出通道 o 的权重来自于该通道下所有元素的平方和的平方根的倒数
        demod = torch.rsqrt(torch.sum(bw.flatten(2) ** 2, dim=2) + 1e-8)

        # 每一个o和其下的所有i相乘，最终每个o下所有元素平方和为1
        bw = torch.einsum('boi...,bo->boi...', bw, demod)

        w = rearrange(bw, 'b o i ... -> (b o) i ...')
        x = rearrange(X, 'b i ... -> 1 (b i) ...')     # Batch变成1，输入通道 bd
        
        # 每个 Batch 独立卷积
        self.conv.groups = B * self.groups
        # (1, b*o, ..)
        # 偏置不再适配
        o = self.conv._conv_forward(x, w, None)
        bo = rearrange(o, '1 (b d) ... -> b d ...', b=B)

        # 每个输出通道一个偏置，所有样本共享
        if self.conv.bias is not None:
            bo = bo + self.conv.bias[None,:,None]

        return bo
