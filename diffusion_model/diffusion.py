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
    loss: str                   # { Residual, Noise }

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
                return ((0.5 * beta_d * (t ** 2) + beta_min * t ).exp() - 1).sqrt()

            elif mode.model == 'VE':
                # sigma_min = torch.tensor(0.02,device=d)
                # sigma_max = torch.tensor(100.,device=d)

                # below is the original formula of VP noise schedule function both in sampling and training
                # return sigma_min * (sigma_max / sigma_min)**t

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
                sigma_max = torch.tensor(100.0, device=d)
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
                sigma_max = torch.tensor(100.0, device=d)
                sigma_min = torch.tensor(0.02, device=d)
                t = (sigma_max  **  2) * (sigma_min  **  2 / sigma_max  **  2)  **  s
                return t, sigma_func(t)

        self.sigma_sche_train = sigma_sche_train_fun
        self.sigma_sche_sample = sigma_sche_sample_fun

        self.sigma_data = args.std_data
        self.mode = mode

    def d_model(self, model: Denoiser, x_hat: X, sigma: T, t):

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
        if self.mode.loss == 'Residual':
            pred_x0 = c_skip * x_hat + c_out * model( x = c_in * x_hat, t = c_noise )
            return pred_x0
        else:
            return model( x = c_in * x_hat, t = c_noise ), c_out, c_skip

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

        n = torch.randn_like(x0) * sigma
        x_hat = x0 + n

        if self.mode.loss == 'Residual':
            return self.d_model(model, x_hat, sigma, t), lambda_sigma
        else:
            f_theta, c_out, c_skip = self.d_model(model, x_hat, sigma, t)
            return f_theta, c_out, c_skip, lambda_sigma, n

    def inferring_sample(self, model: Denoiser, x_hat: X, sigma: T, t) -> X:
        return self.d_model(model, x_hat, sigma, t)

    @torch.no_grad()
    def sample( self, denoiser: Denoiser, num_steps, x_n: X ):
        """
            Adapt from https://github.com/NVlabs/edm/blob/main/generate.py
        """
        d = x_n.device
        B = x_n.shape[0]

        beta_min = 0.1
        beta_d = 19.9
        rho = 7
        epsilon_s=0.003

        S_churn = 40.
        S_min = 1e-1
        S_max = 1e1
        S_noise = 1.003

        # Unify time steps
        sigma_min = {'VP': None, 'VE': 0.02, 'EDM': 0.002}[self.mode.model]
        sigma_max = {'VP': None, 'VE': 100, 'EDM': 80}[self.mode.model]
        steps = torch.arange(num_steps, device=d)
        if self.mode.model == 'VP':
            t_steps = 1 + steps / (num_steps - 1) * (epsilon_s - 1)
        elif self.mode.model == 'VE':
            t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (steps / (num_steps - 1)))
        elif self.mode.model == 'EDM':
            t_steps = (sigma_max ** (1 / rho) + steps / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        else:
            t_steps = steps /  num_steps
            t_steps = 0.5 + (t_steps - 0.5) * (1 - (5e-2) * 2)
            log_snr = (5e-2) * torch.sign(0.5 - t_steps) * torch.log(1 - 2 * torch.abs(0.5 - t_steps))
            t_steps = 0.67 * torch.exp(-0.5 * log_snr)

        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        # Unify Sigma(t)
        if self.mode.sche == 'VP':
            sigma = lambda t: (torch.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
            sigma_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
            sigma_inv = lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        elif self.mode.sche == 'VE':
            sigma = lambda t: t.sqrt()
            sigma_deriv = lambda t: 0.5 / t.sqrt()
            sigma_inv = lambda sigma: sigma ** 2
        elif self.mode.sche == "EDM":
            sigma = lambda t: t
            sigma_deriv = lambda t: 1
            sigma_inv = lambda sigma: sigma

        # Unify S(t)
        if self.mode.model == 'VP':
            s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
            s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
        else:
            s = lambda t: 1
            s_deriv = lambda t: 0

        # Unify Sample Process
        t_next = t_steps[0]
        x_next = x_n * sigma(t_next) * s(t_next)        # x_hat * s --> original x

        def dx_dt(x_hat, t_hat):
            pred = self.inferring_sample(denoiser, x_hat / s(t_hat), sigma(t_hat).view(B, 1, 1), t_hat.view(B, 1, 1))
            dxdt = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(
                t_hat) / sigma(t_hat) * pred
            return dxdt

        if self.mode.flow == 'SDE':
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                x_cur = x_next

                # Only adding extra noise in limited range of sigma
                gamma = min(S_churn / num_steps, 2**0.5 - 1) if S_min <= sigma(t_cur) <= S_max else 0
                # back to pre-noise level
                sigma_new = sigma(t_cur) + gamma * sigma(t_cur)
                # get the timestamp related to pre-noise level
                t_hat = sigma_inv(sigma_new)
                # 1 - update the scale s(t_cur)
                # 2 - add the noise
                x_hat = ((x_cur/s(t_cur)) * s(t_hat) +
                             S_noise * torch.randn_like(x_cur) *                                # normalized noise with standard deviation 'S_noise'
                             ( sigma(t_hat) ** 2 - sigma(t_cur) ** 2 ).clip(min=0).sqrt() *     # coefficient
                             s(t_hat) )                                                         # scaling

                # Euler step
                d_0 = dx_dt(x_hat, t_hat)
                # x_new <- dx + x_hat
                x_mid = x_hat + (t_next - t_hat) * d_0

                # Heun Step
                if self.mode.diff == 'Euler' or i == num_steps - 1:
                    x_next = x_mid
                else:
                    assert self.mode.diff == 'Heun'
                    d_1 = dx_dt(x_mid, t_next)
                    x_next = x_hat + (t_next - t_hat) * 0.5 * (d_0 + d_1)

        else:
            assert self.mode.flow == 'PFODE'

            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                x_cur = x_next

                # Euler step
                d_0 = dx_dt(x_cur, t_cur)
                # x_new <- dx + x_hat
                x_mid = x_cur + (t_next - t_cur) * d_0

                # Heun Step
                if self.mode.diff == 'Euler' or i == num_steps - 1:
                    x_next = x_mid
                else:
                    assert self.mode.diff == 'Heun'
                    d_1 = dx_dt(x_mid, t_next)
                    x_next = x_cur + (t_next - t_cur) * 0.5 * (d_0 + d_1)

        return x_next
