import torch

class DDIMSampler:

    def __init__(self, diffusion, ddim_steps=50, eta=0.0):
        self.diffusion = diffusion
        self.T = diffusion.T
        self.alpha_bars = diffusion.alpha_bars
        self.eta = eta

        step_ratio = self.T // ddim_steps
        self.timesteps = list(range(0, self.T, step_ratio))[:ddim_steps]
        self.timesteps = list(reversed(self.timesteps))

    @torch.no_grad()
    def sample(self, model, n, img_size=28, channels=1, device="cuda"):
        model.eval()
        x = torch.randn(n, channels, img_size, img_size, device=device)

        for i, t_cur in enumerate(self.timesteps):
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else -1

            t_batch = torch.full((n,), t_cur, device=device).long()
            pred_noise = model(x, t_batch)

            alpha_bar_t = self.alpha_bars[t_cur]
            alpha_bar_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1, 1)

            sigma = self.eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise

            if t_prev >= 0:
                noise = torch.randn_like(x) if self.eta > 0 else 0
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                x = pred_x0

        model.train()
        return x.clamp(-1, 1)
