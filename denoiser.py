
from dataclasses import dataclass
import torch as torch
from torch import nn
from ..data.prepare_map import NUM_LABELS
from ..modules.modconv import ModConv, Cond
from ..modules.resskipnet import ResSkipNet
from ..modules.gated_layer import GatedLayer

@dataclass
class DenoiserArgs:
    h_dim: int
    depth: int
    expand: int

    cond_dim: int
    cond_depth: int

    mod_depth: int

class Denoiser(nn.Module):
    def __init__( self, dim, a_dim, args: DenoiserArgs ):
        super().__init__()

        c, d = args.cond_dim, args.mod_depth

        # sigma, label  -->  Conditional Vector
        self.proj_cond = nn.Sequential()
        self.proj_cond.extend([nn.Linear(1 + NUM_LABELS, c),
                               nn.SiLU()])
        for i in range(args.cond_depth - 1):
            self.proj_cond.extend([nn.Linear(c, c),
                                   nn.SiLU()])

        self.cond = Cond(c, d)

        self.proj_in = ModConv(self.cond, nn.Conv1d(dim, args.h_dim, 1))

        self.gru_net = ResSkipNet(args.h_dim, [ GatedLayer(self.cond, args.h_dim, a_dim, args.expand) for _ in range(args.depth)])

        # adapt to [B, X, L]
        self.proj_out = nn.Conv1d(args.h_dim, dim, 1)

        torch.nn.init.zeros_(self.proj_out.weight)
        torch.nn.init.zeros_(self.proj_out.bias)

    def forward( self, a, label, x, t ):
        """
            :param a:       [B, A_dim, L]   音频
            :param label:   [B, L_num]      超参
            :param x:       [B, X_dim, L]   噪声图谱
            :param t:       [B]             噪声方差
            :return:        [B, X_dim, L]   预测纯净图谱
        """

        # sigma + label  -->  c
        c = self.proj_cond(torch.cat([t[:,None],label], dim=1))       # (B, Cond_dim)

        self.cond.set_cond(c)

        h = self.proj_in(x)         # 待降噪的噪声图谱映射到嵌入层空间维度    (B, h, L)

        h = self.gru_net(h,a)       # 待降噪图谱 + 音频                   (B, h, L), (B, A, L)

        return self.proj_out(h)     # (B, X, L)