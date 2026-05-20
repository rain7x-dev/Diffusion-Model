import torch
import matplotlib.pyplot as plt
import os

def save_checkpoint(model, path='checkpoints/checkpoint.pt'):
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_checkpoint(model, path='checkpoints/checkpoint.pt'):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")

def plot_loss(train_losses, epochs):
    os.makedirs('results', exist_ok=True)
    plt.plot(range(1, epochs+1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('results/loss_curve.png')
    plt.show()
