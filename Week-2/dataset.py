class NoisyMNIST(Dataset):
    def __init__(self, root='./data', train=True, sigma=0.3):
        self.sigma = sigma

        transform = transforms.ToTensor()

        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        clean, _ = self.dataset[idx]

        noise = torch.randn_like(clean) * self.sigma

        noisy = clean + noise

        noisy = torch.clamp(noisy, 0.0, 1.0)

        return noisy, clean
