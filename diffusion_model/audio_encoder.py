from torch import nn
from ..modules.mingru import minGRU
from dataclasses import dataclass

@dataclass
class AudioFeatureArgs:
    scales: list[int]
    conv_expand: int
    seq_depth: int
    seq_expand: int

class AudioFeatEncoder(nn.Module):
    def __init__(
        self,
        dim,
        args: AudioFeatureArgs,
    ):
        super().__init__()

        dim_0 = dim // 2**len(args.scales)
        assert 2**len(args.scales) * dim_0 == dim

        self.feat_expand =  nn.Conv2d(1, dim_0, 1)

        self.freq_cnl_net = nn.Sequential()
        dim_cur = dim_0
        for s in args.scales:
            dim_hid = dim_cur * 2
            self.freq_cnl_net.extend([
                nn.Conv2d(dim_cur, dim_hid, 1),     # (B, dim_hid, D_A, L)
                nn.Conv2d(dim_hid, dim_hid, kernel_size=(5, 1), padding=(2, 0), groups=dim_hid),    # along the D_A
                nn.ReLU(),
                nn.Conv2d(dim_hid, dim_cur, 1),     # (B, dim_cur, D_A, L)
                nn.ReLU(),
                nn.MaxPool2d((s,1), (s,1)),   # (B, dim_cur, D_A/s, L)
                nn.Conv2d(dim_cur, dim_cur*2, 1),   # (B, dim_cur*2, D_A/s, L)
            ])
            dim_cur *= 2

        self.time_cnl_net = nn.Sequential()
        class TimeCnlResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    # (B, D, L)
                    nn.InstanceNorm1d(dim_cur),
                    nn.Conv1d(dim_cur, dim_cur, 3,1,1, groups=dim_cur),  # Time Seq Fusing
                    nn.ReLU(),
                    nn.Conv1d(dim_cur, dim_cur * args.seq_expand*2, 1),               # for h & g
                    minGRU(bi_dir=True),    # 翻转序列捕捉反向特征
                    nn.Conv1d(dim_cur*args.seq_expand, dim_cur, 1),
                )

            def forward(self, x):
                return x + self.net(x)

        self.time_cnl_net.extend([TimeCnlResNet() for _ in range(args.seq_depth)])

    def forward(self, audio):
        """
            :param audio: [B, D_Audio, L]
            :return: [B, D_hid, L]
        """
        B,_,L = audio.shape
        x = audio.unsqueeze(1)       # [B, 1, D_A, L]
        x = self.feat_expand(x)      # [B, dim_0, D_A, L]
        x = self.freq_cnl_net(x)     # [B, dim, D_A', L]
        x = x.view(B, -1, L)         # [B, dim*D_A', L]
        x = self.time_cnl_net(x)

        return x
