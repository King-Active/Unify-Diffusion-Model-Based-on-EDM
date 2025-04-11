import torch
import torch.nn as nn
import torch.nn.functional as F

def min_gru( h, g ):
    """
        h: [B, H, L]    生成 候选隐状态
        g: [B, H, L]    生成 门控信号
    """

    log_h_tilde = torch.where(h < 0, -F.softplus(-h), torch.log(h + 0.5))
    log_1_z = -F.softplus(g)
    log_z = -F.softplus(-g)

    # heinsen scan
    decrease = (log_1_z).cumsum(dim=-1)
    update = (log_z + log_h_tilde - decrease).logcumsumexp(dim=-1)
    return torch.exp(decrease + update)

class minGRU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
            [B, H, L] --> [B, H/2 ... L]
        """

        # suppose 不同特征通道遵循相同分布
        # so do   因此沿着特征拆分，一半作为门控 g，一半作为候选隐状态 h
        fore, back = x.chunk(2, dim=1)      # (B,H,L) --> (B,H/2,L)
        return torch.cat([
            min_gru(*fore.chunk(2, dim=1)),                  #  forward   (B,H/2,L) --> (B,H/4,L)
            min_gru(*back.flip(2).chunk(2, dim=1)).flip(2),  #  backward  (B,H/2,L) --> (B,H/4,L)
        ], dim=1)
