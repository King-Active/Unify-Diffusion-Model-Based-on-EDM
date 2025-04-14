import torch
import torch.nn.functional as F
import torch.nn as nn
from .modconv import Cond, ModConv
from .mingru import minGRU

class GatedLayer(nn.Module):
    def __init__(self, cond: Cond, h_dim, a_dim, expand):
        super().__init__()
        self.cond = cond
        H = h_dim * expand

        self.exp = nn.Sequential(
            ModConv(self.cond, nn.Conv1d(h_dim + a_dim, H * 2, 1)),
            nn.SiLU(),
            ModConv(self.cond, nn.Conv1d(H * 2, H * 2, 1)),
        )
        
        self.gru = nn.Sequential(
            ModConv(self.cond, nn.Conv1d(H, H, 3, 1, 1, groups=H)),
            nn.SiLU(),
            ModConv(self.cond, nn.Conv1d(H, H * 2, 1)),
            minGRU(bi_dir=True),
        )
        self.out = ModConv(self.cond, nn.Conv1d(H, h_dim * 2, 1))

    def forward(self, X, A):
        """
            :param X:   噪声图谱  (B, h_dim, L)
            :param A:   经过 label 和 t 调制后的音频 (B, A, L)
        """
        h, g = self.exp(torch.cat([X, A], dim=1)).chunk(2, dim=1)
        h = self.gru(h) * F.silu(g) + self.gru(g) * F.silu(h)
        return self.out(h)
