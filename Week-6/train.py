import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Diffusion:
    def __init__(self, num_timesteps=1000, schedule="linear", device="cuda",
                 num_classes=10, p_uncond=0.15):
        self.device = device
        self.T = num_timesteps
        self.scheduler = NoiseScheduler(num_timesteps, schedule=schedule, device=device)
        self.num_classes = num_classes
        self.null_token = num_classes
        self.p_uncond = p_uncond

        s = self.scheduler
        self.betas = s.betas
        self.alphas = s.alphas
        self.alpha_bars = s.alpha_bars
        self.sqrt_recip_alphas = torch.sqrt(1.0 / s.alphas)

    def training_loss(self, model, x_0, labels):
        b = x_0.shape[0]
        t = self.scheduler.sample_timesteps(b)
        x_t, noise = self.scheduler.add_noise(x_0, t)

        drop_mask = torch.rand(b, device=self.device) < self.p_uncond
        labels = labels.clone()
        labels[drop_mask] = self.null_token

        pred_noise = model(x_t, t, labels)
        loss = F.smooth_l1_loss(pred_noise, noise)
        return loss, drop_mask.float().mean().item()


if __name__ == "__main__":
    EPOCHS = 60
    BATCH = 128
    LR = 2e-4
    T = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)

    model = UNet(num_classes=10).to(DEVICE)
    diffusion = Diffusion(num_timesteps=T, schedule="linear", device=DEVICE,
                          num_classes=10, p_uncond=0.15)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, drop_frac = 0.0, 0.0
        for x_0, labels in loader:
            x_0, labels = x_0.to(DEVICE), labels.to(DEVICE)
            loss, df = diffusion.training_loss(model, x_0, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            drop_frac += df

        avg = running / len(loader)
        avg_drop = drop_frac / len(loader)

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg:.4f}  null_label_frac={avg_drop:.3f}")

        if epoch % 20 == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), f"checkpoints/cond_epoch_{epoch}.pt")

    torch.save(model.state_dict(), "checkpoints/cond_final.pt")
    print("Training done.")
