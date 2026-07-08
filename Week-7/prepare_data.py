from datasets import load_dataset
import matplotlib.pyplot as plt

ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
print(f"Dataset size: {len(ds)}")

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(ds[i]["image"])
    ax.axis("off")
plt.tight_layout()
plt.savefig("raw_samples.png", dpi=120)
print("Saved raw_samples.png")
