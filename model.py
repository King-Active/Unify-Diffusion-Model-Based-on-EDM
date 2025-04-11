from functools import partial
import torch
from einops import repeat, rearrange

import pytorch_lightning as pl

from ..data.dataset import Batch
from ..data.beatmap.encode import X_DIM, CursorSignals

from .diffusion import UnifyDiffusionFramework, DiffusionArgs, Mode
from .denoiser import Denoiser, DenoiserArgs
from .audio_encoder import AudioFeatEncoder, AudioFeatureArgs

    
class Model(pl.LightningModule):
    def __init__(
        self,
        val_batches,
        val_steps,
        lr,
        audio_features,
        diffusion_args: DiffusionArgs,
        mode: Mode,
        denoiser_args: DenoiserArgs,
        audio_feature_args: AudioFeatureArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.audio_encoder = AudioFeatEncoder(audio_features, audio_feature_args)
        self.diffusion = UnifyDiffusionFramework(diffusion_args, mode)
        self.denoiser = Denoiser(X_DIM, audio_features, denoiser_args)

        self.val_batches = val_batches
        self.val_steps = val_steps
        self.lr = lr
    

    def forward(self, audio, chart, labels):
        """
        :param audio: [B, A_dim, L]
        :param chart: [B, X_dim, L]
        :param labels: [B, N]
        """
        # set audio feat encoder + labels
        denoiser = partial(self.denoiser, a=self.audio_encoder(audio), label=labels)

        pred, lambda_sigma = self.diffusion.training_sample(denoiser, chart)

        # --- Loss Function ---
        # time
        time_loss = (lambda_sigma * (pred - chart) ** 2).mean()

        # space
        true_diff = chart[:, CursorSignals, 1:] - chart[:, CursorSignals, :-1]
        pred_diff = pred[:, CursorSignals, 1:] - pred[:, CursorSignals, :-1]
        cd_map = lambda diff: torch.tanh(diff * 20)
        space_loss = (lambda_sigma * (cd_map(true_diff) - cd_map(pred_diff)) ** 2).mean()

        loss = time_loss + 0.01 * space_loss

        print("[Loss] ", loss.item())

        return loss, {
            "loss": loss.detach(),
            "time": time_loss.detach(),
            "space": space_loss.detach(),
        }
    

    @torch.no_grad()
    def sample( self, audio, labels, num_steps, **kwargs):

        assert num_steps > 0

        num_samples = labels.size(0)
        audio = audio.unsqueeze(0).repeat(num_samples, 1, 1)
        
        # initial noise map
        x_n = torch.randn(num_samples, X_DIM, audio.size(-1), device=audio.device)

        denoiser = partial(self.denoiser,self.audio_features(audio),labels)

        return self.diffusion.sample(
            denoiser, 
            num_steps,
            x_n,
            **kwargs,
        ).clamp(min=-1, max=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def training_step(self, batch: Batch, batch_idx):
        loss, log_dict = self(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss

    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        with torch.no_grad():
            # 音频； 真实图谱； 超参数
            a,x,l = batch
            # 将原长序列切分为 val_batches 段 a.size(-1) // self.val_batches 大小的子序列，同时剔除剩余部分
            bL = self.val_batches * (a.size(-1) // self.val_batches)
            # 转换成小batch
            a = rearrange(a[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            x = rearrange(x[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            # 共享超参
            l = repeat(l, '1 d -> b d', b = self.val_batches)
            _, log_dict = self(a,x,l)   # 运行模型
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })
