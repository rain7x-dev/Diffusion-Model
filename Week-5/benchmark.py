import time
import math
import matplotlib.pyplot as plt
import torch

def save_grid(imgs, path, n=16):
    imgs = (imgs + 1) / 2
    imgs = imgs.cpu()
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = axes.flatten() if n > 1 else [axes]
    for k, ax in enumerate(axes):
        if k < n:
            ax.imshow(imgs[k].squeeze(0), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def benchmark_ddpm(model, diffusion, n=16):
    start = time.time()
    imgs = diffusion.sample(model, n)
    elapsed = time.time() - start
    save_grid(imgs, "week5/results/ddpm_1000_steps.png", n)
    print(f"DDPM (1000 steps): {elapsed:.2f}s")
    return elapsed


def benchmark_ddim(model, diffusion, steps, n=16, device="cuda"):
    sampler = DDIMSampler(diffusion, ddim_steps=steps)
    start = time.time()
    imgs = sampler.sample(model, n, device=device)
    elapsed = time.time() - start
    save_grid(imgs, f"week5/results/ddim_{steps}_steps.png", n)
    print(f"DDIM ({steps} steps): {elapsed:.2f}s")
    return elapsed


def run_full_benchmark(model, diffusion, device="cuda"):
    results = {}
    results["DDPM (1000)"] = benchmark_ddpm(model, diffusion)

    for steps in [10, 25, 50, 100]:
        results[f"DDIM ({steps})"] = benchmark_ddim(model, diffusion, steps, device=device)

    labels = list(results.keys())
    times = list(results.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, times, color=["#cc4444"] + ["#4477cc"] * 4)
    plt.ylabel("Time (seconds)")
    plt.title("DDPM vs DDIM Sampling Speed")
    plt.xticks(rotation=20)
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{t:.1f}s", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("week5/results/timing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results

