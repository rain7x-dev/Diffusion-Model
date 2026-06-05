import matplotlib.pyplot as plt

def to_displayable(x):
    return (x.clamp(-1, 1) + 1) / 2

def visualize_noising(image, scheduler, timesteps=(0, 100, 250, 500, 750, 999),
                      save_path="results/noising_trajectory.png"):

    fig, axes = plt.subplots(1, len(timesteps), figsize=(3 * len(timesteps), 3))

    for ax, t_val in zip(axes, timesteps):
        t = torch.tensor([t_val], device=scheduler.device).long()
        x_t, _ = scheduler.add_noise(image.unsqueeze(0).to(scheduler.device), t)
        img = to_displayable(x_t.squeeze(0)).cpu()

        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"t = {t_val}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved noising trajectory to {save_path}")


def plot_snr_comparison(num_timesteps=1000,
                        save_path="results/linear_vs_cosine.png"):

    linear = NoiseScheduler(num_timesteps, schedule="linear")
    cosine = NoiseScheduler(num_timesteps, schedule="cosine")

    def snr(sched):
        ab = sched.alpha_bars
        return (ab / (1 - ab)).cpu()

    t = torch.arange(num_timesteps)
    plt.figure(figsize=(8, 5))
    plt.plot(t, snr(linear), label="linear")
    plt.plot(t, snr(cosine), label="cosine")
    plt.yscale("log")
    plt.xlabel("timestep t")
    plt.ylabel("SNR  =  ᾱ_t / (1 - ᾱ_t)   (log scale)")
    plt.title("Signal-to-Noise Ratio: linear vs cosine")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SNR comparison to {save_path}")


def plot_final_noise_distribution(image, scheduler,
                                  save_path="results/final_noise_distribution.png"):

    import numpy as np

    t = torch.tensor([scheduler.num_timesteps - 1],
                     device=scheduler.device).long()
    x_T, _ = scheduler.add_noise(image.unsqueeze(0).to(scheduler.device), t)
    vals = x_T.flatten().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=60, density=True, alpha=0.6,
             label=f"x_T pixels (mean={vals.mean():.3f}, std={vals.std():.3f})")

    xs = np.linspace(-4, 4, 200)
    gaussian = np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi)
    plt.plot(xs, gaussian, "r--", linewidth=2, label="N(0, 1)")

    plt.xlabel("pixel value")
    plt.ylabel("density")
    plt.title("Distribution of x_T  vs  standard normal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved final noise distribution to {save_path}")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    img, _ = ds[0]

    sched = NoiseScheduler(num_timesteps=1000, schedule="linear")
    visualize_noising(img, sched)
    plot_snr_comparison()
    plot_final_noise_distribution(img, sched)
