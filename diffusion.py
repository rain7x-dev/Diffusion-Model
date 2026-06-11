import torch.nn.functional as F

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

class Diffusion:
    def __init__(self, num_timesteps=1000, schedule="linear", device="cuda"):
        self.device = device
        self.T = num_timesteps
        self.scheduler = NoiseScheduler(num_timesteps, schedule=schedule,
                                        device=device)
        s = self.scheduler
        self.betas = s.betas
        self.alphas = s.alphas
        self.alpha_bars = s.alpha_bars
        self.sqrt_recip_alphas = torch.sqrt(1.0 / s.alphas)
        self.posterior_var = s.betas

    def training_loss(self, model, x_0):
        b = x_0.shape[0]
        t = self.scheduler.sample_timesteps(b)
        x_t, noise = self.scheduler.add_noise(x_0, t)
        pred_noise = model(x_t, t)
        return F.smooth_l1_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, model, n, img_size=28, channels=1):
        model.eval()
        x = torch.randn(n, channels, img_size, img_size, device=self.device)

        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device).long()
            pred_noise = model(x, t)

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            beta = self.betas[i]

            coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            mean = self.sqrt_recip_alphas[i] * (x - coef * pred_noise)

            if i > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean

        model.train()
        return x.clamp(-1, 1)
