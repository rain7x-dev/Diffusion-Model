import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

IMG_SIZE = 64

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def get_dataloader(batch_size=64, num_workers=2):
    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

    def transform_fn(examples):
        examples["pixel_values"] = [train_tf(img.convert("RGB")) for img in examples["image"]]
        return examples

    ds.set_transform(transform_fn)

    def collate(batch):
        return torch.stack([item["pixel_values"] for item in batch])

    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, collate_fn=collate, drop_last=True)


if __name__ == "__main__":
    loader = get_dataloader(batch_size=8)
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")
    print(f"Value range: [{batch.min():.2f}, {batch.max():.2f}]")
