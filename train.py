import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

EPOCHS = 80
BATCH = 128
LR = 2e-4
T = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_EVERY = 10
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    num_workers=2, drop_last=True)

model = UNet().to(DEVICE)
diffusion = Diffusion(num_timesteps=T, schedule="linear", device=DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

loss_history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for x_0, _ in loader:
        x_0 = x_0.to(DEVICE)
        loss = diffusion.training_loss(model, x_0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()

    avg = running / len(loader)
    loss_history.append(avg)
    print(f"Epoch {epoch:3d}/{EPOCHS}  loss = {avg:.4f}")

    if epoch % SAMPLE_EVERY == 0 or epoch == EPOCHS:
        save_sample_grid(model, diffusion,
                         path=f"results/samples_epoch_{epoch}.png", n=16)
        torch.save(model.state_dict(),
                   f"checkpoints/ddpm_epoch_{epoch}.pt")

torch.save(model.state_dict(), "checkpoints/ddpm_mnist.pt")
save_sample_grid(model, diffusion, path="results/samples_final.png", n=64)

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), loss_history)
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Training loss")
plt.grid(True, alpha=0.3)
plt.savefig("results/loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Done. Checkpoint + results saved.")
