"""
    zwzengi@Outlook.com
"""
from collections.abc import Callable
from jaxtyping import Float
from dataclasses import dataclass
import torch

T = Float[torch.Tensor, "B 1 1"]  # diffusion step
X = Float[torch.Tensor, "B D N"]  # sequence
Denoiser = Callable[[X, T], X]    # p(x0 | xt, t)

@dataclass
class DiffusionArgs:
    log_snr_scale: float
    log_snr_bound: float
    std_data: float

@dataclass
class Mode:
    flow: str                   # { PFODE, SDE, SS }    SS means stochastic sampler
    model: str                  # { EDM, VP, VE }
    diff: str                   # { Euler, Heun }
    sche: str                   # { LAPLACE, EDM, VP, VE }

class UnifyDiffusionFramework:

    def __init__(
            self,
            args: DiffusionArgs,
            mode: Mode
    ):

        super().__init__()

        def sigma_func(t):
            d = t.device

            if mode.model == 'EDM':
                return t

            elif mode.model == 'VP':
                beta_d = torch.tensor(19.9, device=d)
                beta_min = torch.tensor(0.1, device=d)
                return torch.sqrt(torch.exp(0.5 * beta_d * (t ** 2) + beta_min * t ) - 1)

            elif mode.model == 'VE':
                sigma_min = torch.tensor(0.02,device=d)
                sigma_max = torch.tensor(100.,device=d)

                # below is the original formula of VP noise schedule function both in sampling and training
                return sigma_min * (sigma_max / sigma_min)**t

                # below is the formula given by EDM through ODE inference in sampling (training is the same as above)
                return torch.sqrt(t)

        def sigma_sche_train_fun(B):
            d = B.device

            # 不同 B 采用不同训练调度采样值
            if mode.sche == 'EDM':
                """
                采用基于 sigma 的均匀分布调度 
                """
                P_mean = -1.2
                P_std = 1.2
                ln_sigma = torch.normal(mean=P_mean, std=P_std, size=(B, 1, 1), device=d)
                sigma = torch.exp(ln_sigma).clamp(0.002, 80)
                return sigma, sigma    # sigma(t) = t

            elif mode.sche == 'LAPLACE':
                t = torch.rand(B, 1, 1, device=d)
                t = 0.5 + (t - 0.5) * (1 - args.log_snr_bound * 2)  # [0,1] -> [b,1-b]
                log_snr = args.log_snr_scale * torch.sign(0.5 - t) * torch.log(1 - 2 * torch.abs(0.5 - t))
                sigma = args.std_data * torch.exp(-0.5 * log_snr)
                return sigma, sigma    # sigma(t) = t

            elif mode.sche == 'VP':
                """
                采用基于 t 的均匀分布调度，需要转化为 sigma
                """
                epsilon_t = torch.tensor(10 ** (-5), device = d)
                t = epsilon_t + (1 - epsilon_t) * torch.rand(B, 1, 1, device=d)
                sigma = sigma_func(t)
                return t, sigma

            elif mode.sche == 'VE':
                """
                采用基于 sigma 的均匀分布调度 
                """
                sigma_min = torch.tensor(0.02, device=d)
                sigma_max = torch.tensor(50.0, device=d)
                ln_sigma_min = torch.log(sigma_min)
                ln_sigma_max = torch.log(sigma_max)
                ln_sigma = ln_sigma_min + (ln_sigma_max - ln_sigma_min) * torch.rand(B, 1, 1, device=d)
                sigma = torch.exp(ln_sigma).clamp(0.02, 100)           # EDM Prefer : (0.002, 80)
                t = torch.log(sigma/sigma_min)/torch.log(sigma_max/sigma_min)   # 后续不会使用
                return t, sigma

        def sigma_sche_sample_fun(steps):
            d = steps.device

            # 所有 B 共享同一采样调度
            s = torch.linspace(0, 1, steps, device=d)
            if mode.sche == 'EDM':
                sigma_max, sigma_min = torch.tensor(80.0, device=d), torch.tensor(0.002, device=d)
                rou = torch.tensor(7, device=d)
                t = (sigma_max  **  (1 / rou) + s * (sigma_min  **  (1 / rou) - sigma_max  **  ( 1 / rou)))  **  rou
                return t, sigma_func(t)

            elif mode.sche == 'LAPLACE':
                t = 0.5 + (s - 0.5) * (1 - args.log_snr_bound * 2)  # [0,1] -> [b,1-b]
                log_snr = args.log_snr_scale * torch.sign(0.5 - t) * torch.log(1 - 2 * torch.abs(0.5 - t))
                t = args.std_data * torch.exp(-0.5 * log_snr)
                return t, sigma_func(t)

            elif mode.sche == 'VP':
                epsilon_s = torch.tensor(10 ** -3, device = d)
                t = 1 + s * (epsilon_s - 1)
                return t, sigma_func(t)

            elif mode.sche == 'VE':
                sigma_max = torch.tensor(50.0, device=d)
                sigma_min = torch.tensor(0.02, device=d)
                t = sigma_max  **  2 * (sigma_min  **  2 / sigma_max  **  2)  **  s
                return t, sigma_func(t)

        self.sigma_sche_train = sigma_sche_train_fun
        self.sigma_sche_sample = sigma_sche_sample_fun

        self.sigma_data = args.std_data
        self.mode = mode

    def d_model(self, model: Denoiser, x_hat: X, sigma: T, t):
        d = x_hat.device

        c_skip, c_out, c_in, c_noise  =  None, None, None, None

        if self.mode.model == 'EDM':
            power_sum = sigma ** 2 + self.sigma_data ** 2
            sqrt_sum = power_sum.sqrt()

            c_skip = self.sigma_data ** 2 / power_sum
            c_out = sigma * self.sigma_data / sqrt_sum
            c_in = 1 / sqrt_sum
            c_noise = 0.25 * torch.log(sigma)[:, 0, 0]

        elif self.mode.model == 'VP':
            M = 1000
            c_skip = 1
            c_out = -sigma
            c_in = 1 / torch.sqrt(sigma**2 + 1)
            c_noise = (M-1) * t[:, 0, 0]

        elif self.mode.model == 'VE':
            c_skip = 1
            c_out = sigma
            c_in = 1
            c_noise = torch.log(0.5 * sigma)[:, 0, 0]

        # model --> F_theta
        pred_x0 = c_skip * x_hat + c_out * model( x = c_in * x_hat, t = c_noise )
        return pred_x0

    def training_sample(self, model: Denoiser, x0: X):

        # 当前轮叠加噪声方差
        t, sigma = self.sigma_sche_train(torch.tensor(x0.size(0), device=x0.device))

        # 损失权重
        lambda_sigma = torch.tensor(0,device=x0.device)
        if self.mode.model == 'EDM':
            lambda_sigma = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        elif self.mode.model == 'VP':
            lambda_sigma = 1 / sigma ** 2
        elif self.mode.model == 'VE':
            lambda_sigma = 1 / sigma ** 2

        x_hat = x0 + torch.randn_like(x0) * sigma

        return self.d_model(model, x_hat, sigma, t), lambda_sigma

    def inferring_sample(self, model: Denoiser, x_hat: X, sigma: T, t) -> X:
        return self.d_model(model, x_hat, sigma, t)

    @torch.no_grad()
    def sample( self, denoiser: Denoiser, num_steps, x_n: X ):

        #  x_n 随机初始化噪声
        B = x_n.size(0)
        d = x_n.device

        # 所有 Batch 共享
        ts, sigmas = self.sigma_sche_sample(num_steps)
        sigmas = torch.tensor([*sigmas.tolist(), 0], device=d)
        sigmas = sigmas[:, None, None, None].repeat(1, B, 1, 1)
        loop = zip(sigmas[:-1], sigmas[1:])         # decrease

        def dx_dt_edm(x_hat: X, sigma: T):
            x_0 = self.inferring_sample(denoiser, x_hat, sigma, sigma)
            # point to high probability direction
            return (x_hat - x_0) / sigma  # 指向 噪声

        if self.mode.model == 'EDM' and self.mode.flow == 'SS':
            """
                EDM风格随机性采样：对比一阶 Euler 和 二阶 Heun
                不同于 欧拉丸山 离散化方法，其严格遵循 SDE 反向，先降噪后加噪
                EDM风格先加噪，后遵循 ODE 风格的降噪
            """

            # x_cur = x_n * (sigmas[0, 0, 0, 0] ** 2 + self.sigma_data ** 2) ** .5
            x_cur = x_n * sigmas[0, 0, 0, 0]

            for t_cur, t_nxt in loop:
                # t_nxt < t_cur
                # 避免细节缺失 及 过饱和
                S_churn = 40.
                S_tmin = 1e-1
                S_tmax = 1e1
                S_noise = 1.003

                gamma = min(S_churn / num_steps, 2 ** .5 - 1) if S_tmin <= t_cur[0, 0, 0] <= S_tmax else 0
                t_hat = t_cur + gamma * t_cur   # time nosie
                x_hat = x_cur + torch.sqrt(t_hat ** 2 - t_cur ** 2) * S_noise * torch.randn_like(x_cur)     # chart noise

                # Euler
                d_i = dx_dt_edm(x_hat, t_hat)                   # 指向噪声
                x_nxt = x_hat + (t_nxt - t_hat) * d_i       # 反向： 指向纯净

                # Heun
                if self.mode.diff == "Heun" and t_nxt[0, 0, 0] > 0:
                    d_i_ = dx_dt_edm(x_nxt, t_nxt)
                    x_nxt = x_hat + (t_nxt - t_hat) * 0.5 * (d_i + d_i_)

                x_cur = x_nxt

            return x_cur

        elif self.mode.model == 'EDM' and self.mode.flow == 'PFODE':
            """
                通式
                EDM风格确定性采样：对比一阶 Euler 和 二阶 Heun
                通过适配 s(t) sigma(t) 可实现 VP VE 在本函数下的确定性采样
            """

            # x_cur = x_n * (sigmas[0, 0, 0, 0] ** 2 + self.sigma_data ** 2) ** .5
            x_cur = x_n * sigmas[0, 0, 0, 0]

            for t_cur, t_nxt in loop:
                # t_nxt < t_cur
                d_i = dx_dt_edm(x_cur, t_cur)                   # 指向噪声

                # Euler
                x_nxt = x_cur + (t_nxt - t_cur) * d_i       # 反向： 指向纯净

                # Heun
                if self.mode.diff == "Heun" and t_nxt[0, 0, 0] > 0:
                    d_i_ = dx_dt_edm(x_nxt, t_nxt)
                    x_nxt = x_cur + (t_nxt - t_cur) * 0.5 * (d_i + d_i_)

                x_cur = x_nxt

            return x_cur

        elif self.mode.model == 'VP':
            """
                VP风格随机采样
                欧拉丸山风格，先利用SDE降噪，再添加随机项
            """

            x_t = x_n * sigmas[0, 0, 0, 0]

            for sigma_t, t in zip(sigmas, ts):
                s_t = 1 / torch.sqrt(1 + sigma_t ** 2)      # scale
                beta_max = 20.0
                beta_min = 0.1
                beta_t = beta_min + (beta_max-beta_min) * t

                pred = self.inferring_sample(denoiser, x_t/s_t, sigma_t, t)
                determine = (1 / torch.sqrt(1-beta_t)) * (x_t + beta_t * ( (pred - x_t/s_t) / s_t * (sigma_t**2) ) )
                random = torch.sqrt(beta_t) * torch.randn_like(x_t)

                x_t = determine + random

            return x_t

        elif self.mode.model == 'VE':
            """
                VE风格随机采样
                欧拉丸山风格，先利用SDE降噪，再添加随机项
            """

            x_t = x_n * sigmas[0, 0, 0, 0]

            for sigma_cur, sigma_pre in loop:

                pred = self.inferring_sample(denoiser, x_t, sigma_cur, 0)
                determine = x_t - ( (sigma_cur**2 - sigma_pre**2) / sigma_cur**2 ) * (pred - x_t)
                random = torch.sqrt(sigma_cur**2 - sigma_pre**2) * torch.randn_like(x_t)

                x_t = determine + random

            return x_t
