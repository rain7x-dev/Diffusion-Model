def save_sample_grid(model, diffusion, path="results/samples.png", n=16):
    imgs = diffusion.sample(model, n)
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
    print(f"Saved samples to {path}")
