import os
import copy
import math
import yaml
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

from model import UNet
from scheduler import NoiseScheduler
from dataset import get_dataloader


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


class Diffusion:
    def __init__(self, num_timesteps=1000, schedule="cosine", device="cuda"):
        self.device = device
        self.T = num_timesteps
        self.scheduler = NoiseScheduler(num_timesteps, schedule=schedule, device=device)
        s = self.scheduler
        self.betas = s.betas
        self.alphas = s.alphas
        self.alpha_bars = s.alpha_bars
        self.sqrt_recip_alphas = torch.sqrt(1.0 / s.alphas)

    def training_loss(self, model, x_0):
        b = x_0.shape[0]
        t = self.scheduler.sample_timesteps(b)
        x_t, noise = self.scheduler.add_noise(x_0, t)
        pred_noise = model(x_t, t)
        return F.smooth_l1_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, model, n, img_size=64, channels=3, generator=None):
        model.eval()
        x = torch.randn(n, channels, img_size, img_size,
                        device=self.device, generator=generator)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device).long()
            pred_noise = model(x, t)
            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            beta = self.betas[i]
            coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            mean = self.sqrt_recip_alphas[i] * (x - coef * pred_noise)
            if i > 0:
                noise = torch.randn(x.shape, device=self.device, generator=generator)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
        return x.clamp(-1, 1)


def save_grid(imgs, path, n=16):
    imgs = ((imgs + 1) / 2).cpu().permute(0, 2, 3, 1)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    for k, ax in enumerate(axes.flatten()):
        if k < n:
            ax.imshow(imgs[k])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_DIR = "/content/drive/MyDrive/butterfly_v3_ckpts"
    RESULTS_DIR = "/content/drive/MyDrive/butterfly_v3_results"
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    wandb.init(project=cfg["wandb_project"], config=cfg)

    loader = get_dataloader(batch_size=cfg["batch_size"])
    model = UNet(in_ch=cfg["channels"], base=cfg["base"]).to(DEVICE)
    diffusion = Diffusion(cfg["num_timesteps"], cfg["schedule"], DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    ema = EMA(model, decay=cfg["ema_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"])

    warmup = cfg["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda step: min(1.0, (step + 1) / warmup))

    start_epoch = 1
    resume_path = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    ema.shadow.to(DEVICE)

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        for x_0 in loader:
            x_0 = x_0.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=cfg["amp"]):
                loss = diffusion.training_loss(model, x_0)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            running += loss.item()

        avg = running / len(loader)
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}")
        wandb.log({"loss": avg, "lr": sched.get_last_lr()[0], "epoch": epoch})

        if epoch % cfg["sample_every"] == 0 or epoch == cfg["epochs"]:
            gen = torch.Generator(device=DEVICE).manual_seed(1234)
            grid_path = os.path.join(RESULTS_DIR, f"samples_epoch_{epoch}.png")
            imgs = diffusion.sample(ema.shadow, n=16, img_size=cfg["img_size"],
                                    channels=cfg["channels"], generator=gen)
            save_grid(imgs, grid_path)
            wandb.log({"samples": wandb.Image(grid_path), "epoch": epoch})

        if epoch % cfg["checkpoint_every"] == 0 or epoch == cfg["epochs"]:
            state = {"model": model.state_dict(),
                     "ema": ema.shadow.state_dict(),
                     "opt": opt.state_dict(),
                     "epoch": epoch}
            torch.save(state, os.path.join(CKPT_DIR, "latest.pt"))
            torch.save(state, os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch}.pt"))

    torch.save(ema.shadow.state_dict(), os.path.join(CKPT_DIR, "final_ema.pt"))
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "final_raw.pt"))
    wandb.finish()
    print("Training done.")
