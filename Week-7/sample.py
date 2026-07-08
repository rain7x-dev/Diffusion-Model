import math
import yaml
import torch
import matplotlib.pyplot as plt

from model import UNet
from train import Diffusion


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
    def sample(self, model, n, img_size=64, channels=3, device="cuda", generator=None):
        model.eval()
        x = torch.randn(n, channels, img_size, img_size, device=device, generator=generator)

        for i, t_cur in enumerate(self.timesteps):
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else -1
            t_batch = torch.full((n,), t_cur, device=device).long()
            pred_noise = model(x, t_batch)

            alpha_bar_t = self.alpha_bars[t_cur]
            alpha_bar_prev = (self.alpha_bars[t_prev] if t_prev >= 0
                              else torch.tensor(1.0, device=device))

            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1, 1)

            sigma = self.eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise

            if t_prev >= 0:
                noise = (torch.randn(x.shape, device=device, generator=generator)
                         if self.eta > 0 else 0)
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                x = pred_x0

        return x.clamp(-1, 1)


def save_grid(imgs, path, n=64):
    imgs = ((imgs + 1) / 2).cpu().permute(0, 2, 3, 1)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for k, ax in enumerate(axes.flatten()):
        if k < n:
            ax.imshow(imgs[k])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT = "/content/drive/MyDrive/butterfly_v3_ckpts/final_ema.pt"

    model = UNet(in_ch=cfg["channels"], base=cfg["base"]).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    diffusion = Diffusion(cfg["num_timesteps"], cfg["schedule"], DEVICE)

    sampler = DDIMSampler(diffusion, ddim_steps=50, eta=0.0)
    gen = torch.Generator(device=DEVICE).manual_seed(1234)
    imgs = sampler.sample(model, n=64, img_size=cfg["img_size"],
                          channels=cfg["channels"], device=DEVICE, generator=gen)
    save_grid(imgs, "/content/drive/MyDrive/butterfly_v3_results/final_samples_64.png", n=64)
