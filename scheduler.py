import torch
import torch.nn.functional as F
import math

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, schedule="linear",
                 beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.device = device

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            betas = self._cosine_betas(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bars = alpha_bars.to(device)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars).to(device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars).to(device)

    @staticmethod
    def _cosine_betas(num_timesteps, s=0.008):

        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps) / num_timesteps
        f = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bars = f / f[0]
        betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
        return torch.clamp(betas, 0.0001, 0.999)

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * noise
        return x_t, noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_timesteps, (batch_size,),
                             device=self.device).long()
