import math
import torch
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Tshirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


@torch.no_grad()
def sample_cfg(model, diffusion, n, labels, w=3.0,
               img_size=28, channels=1, device="cuda"):
    model.eval()
    x = torch.randn(n, channels, img_size, img_size, device=device)
    null = torch.full((n,), model.null_token, device=device).long()
    labels = labels.to(device).long()

    for i in reversed(range(diffusion.T)):
        t = torch.full((n,), i, device=device).long()

        eps_cond = model(x, t, labels)
        eps_uncond = model(x, t, null)
        pred_noise = eps_uncond + w * (eps_cond - eps_uncond)

        alpha = diffusion.alphas[i]
        alpha_bar = diffusion.alpha_bars[i]
        beta = diffusion.betas[i]

        coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        mean = diffusion.sqrt_recip_alphas[i] * (x - coef * pred_noise)

        if i > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta) * noise
        else:
            x = mean

    model.train()
    return x.clamp(-1, 1)


def save_grid(imgs, path, titles=None, ncol=None):
    imgs = ((imgs + 1) / 2).cpu()
    n = imgs.shape[0]
    ncol = ncol or int(math.ceil(math.sqrt(n)))
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow + 0.3))
    axes = axes.flatten()
    for k, ax in enumerate(axes):
        if k < n:
            ax.imshow(imgs[k].squeeze(0), cmap="gray")
            if titles:
                ax.set_title(titles[k], fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    model = UNet(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/cond_final.pt", map_location=DEVICE))
    diffusion = Diffusion(device=DEVICE, num_classes=10)

    torch.manual_seed(0)
    labels = torch.arange(10, device=DEVICE)
    imgs = sample_cfg(model, diffusion, n=10, labels=labels, w=3.0, device=DEVICE)
    save_grid(imgs, "results/conditional_grid.png", titles=CLASS_NAMES, ncol=10)

    rows = []
    for w in [1, 3, 5, 10]:
        torch.manual_seed(42)
        labels = torch.full((8,), 9, device=DEVICE)
        imgs = sample_cfg(model, diffusion, n=8, labels=labels, w=float(w), device=DEVICE)
        rows.append(imgs)
    all_imgs = torch.cat(rows, dim=0)
    titles = [f"w={w}" if k % 8 == 0 else "" for w in [1, 3, 5, 10] for k in range(8)]
    save_grid(all_imgs, "results/guidance_scale_sweep.png", titles=titles, ncol=8)

    torch.manual_seed(7)
    labels = torch.full((8,), 9, device=DEVICE)
    uncond = sample_cfg(model, diffusion, n=8, labels=labels, w=0.0, device=DEVICE)
    torch.manual_seed(7)
    cfg = sample_cfg(model, diffusion, n=8, labels=labels, w=5.0, device=DEVICE)
    save_grid(torch.cat([uncond, cfg], dim=0),
              "results/unconditional_vs_cfg.png",
              titles=["uncond"] * 8 + ["w=5"] * 8, ncol=8)
