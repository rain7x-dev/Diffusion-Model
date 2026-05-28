config = {
    'batch_size': 64,
    'epochs': 50,
    'lr': 1e-3,
    'sigma': 0.3,
    'val_split': 0.1,
    'vis_every': 10,
    'save_dir': 'results',
}


def visualize_samples(model, dataset, device, epoch, save_dir, n_samples=8):

    model.eval()
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))

    with torch.no_grad():
        for i in range(n_samples):
            noisy, clean = dataset[i]
            denoised = model(noisy.unsqueeze(0).to(device)).squeeze(0).cpu()

            axes[0, i].imshow(noisy.squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(denoised.squeeze(), cmap='gray')
            axes[1, i].axis('off')
            axes[2, i].imshow(clean.squeeze(), cmap='gray')
            axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Noisy',    fontsize=12)
    axes[1, 0].set_ylabel('Denoised', fontsize=12)
    axes[2, 0].set_ylabel('Clean',    fontsize=12)

    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/samples_epoch_{epoch}.png', dpi=100, bbox_inches='tight')
    plt.close()

    model.train()


def plot_loss_curve(train_losses, val_losses, save_dir):
    """Plot train and validation loss over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/loss_curve.png', dpi=100, bbox_inches='tight')
    plt.close()


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(config['save_dir'], exist_ok=True)

    full_train = NoisyMNIST(sigma=config['sigma'], train=True)
    val_size   = int(len(full_train) * config['val_split'])
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    test_dataset = NoisyMNIST(sigma=config['sigma'], train=False)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'],
                              shuffle=False, num_workers=2)

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples:   {len(val_dataset)}')
    print(f'Test samples:  {len(test_dataset)}')

    model = UNet(in_ch=1, out_ch=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')
    print(f'Starting training for {config["epochs"]} epochs...\n')

    train_losses = []
    val_losses   = []

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        train_loss_sum = 0.0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                output = model(noisy)
                val_loss_sum += criterion(output, clean).item()

        avg_val_loss = val_loss_sum / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch:3d}/{config["epochs"]} | '
              f'Train: {avg_train_loss:.4f} | '
              f'Val: {avg_val_loss:.4f}')

        if epoch == 10 or epoch == 50:
          visualize_samples(model, test_dataset, device, epoch, config['save_dir'])

    plot_loss_curve(train_losses, val_losses, config['save_dir'])
    torch.save(model.state_dict(), 'unet_denoiser.pth')

    print('\nDone!')
    print(f'Visualizations saved in: {config["save_dir"]}/')
    print('Model weights saved as: unet_denoiser.pth')


if __name__ == '__main__':
    train()
