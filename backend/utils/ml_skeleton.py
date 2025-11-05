import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ImageGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28), depth=64):
        super().__init__()
        c, h, w = img_shape
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, depth * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(depth * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(depth * 2, depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),
            nn.ConvTranspose2d(depth, c, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class ImageDiscriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), depth=64):
        super().__init__()
        c, h, w = img_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, depth, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(depth, depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(depth * 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)


def train_image_gan(
    generator,
    discriminator,
    dataset,
    latent_dim=100,
    lr=2e-4,
    betas=(0.5, 0.999),
    epochs=10,
    batch_size=64,
    device=None
):
    """
    Trains a simple DCGAN on the given dataset.

    Args:
        generator (nn.Module): Generator network.
        discriminator (nn.Module): Discriminator network.
        dataset (torch.utils.data.Dataset): Image dataset.
        latent_dim (int): Latent space dimensionality.
        lr (float): Learning rate.
        betas (tuple): Adam optimizer betas.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        device (str): Device to train on ('cuda' or 'cpu').
    """
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    G, D = generator.to(device), discriminator.to(device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=betas)
    criterion = nn.BCELoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"🧠 Training started on {device.upper()} for {epochs} epochs.")

    for epoch in range(epochs):
        for imgs, _ in loader:
            imgs = imgs.to(device)
            bs = imgs.size(0)

            opt_D.zero_grad()

            real_output = D(imgs)
            real_output = real_output.view(bs, -1).mean(dim=1)  # flatten, handle any output shape
            real_labels = torch.ones_like(real_output, device=device)
            real_loss = criterion(real_output, real_labels)

            z = torch.randn(bs, latent_dim, 1, 1, device=device)
            fake_imgs = G(z)
            fake_output = D(fake_imgs.detach())
            fake_output = fake_output.view(bs, -1).mean(dim=1)
            fake_labels = torch.zeros_like(fake_output, device=device)
            fake_loss = criterion(fake_output, fake_labels)

            D_loss = (real_loss + fake_loss) / 2
            D_loss.backward()
            opt_D.step()

            opt_G.zero_grad()
            z = torch.randn(bs, latent_dim, 1, 1, device=device)
            gen_imgs = G(z)
            gen_output = D(gen_imgs)
            gen_output = gen_output.view(bs, -1).mean(dim=1)
            real_labels = torch.ones_like(gen_output, device=device)
            G_loss = criterion(gen_output, real_labels)

            G_loss.backward()
            opt_G.step()

        print(f"[ImageGAN] Epoch {epoch+1}/{epochs} | D_loss={D_loss.item():.4f} | G_loss={G_loss.item():.4f}")

    print("✅ Training complete.")


def generate_images(generator, n_samples, latent_dim=100, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(n_samples, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        imgs = generator.to(device)(z)
    return imgs.cpu()


class AudioGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_len=16384, depth=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, depth * 8, 16, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose1d(depth * 8, depth * 4, 16, 4, 6),
            nn.ReLU(True),
            nn.ConvTranspose1d(depth * 4, depth * 2, 16, 4, 6),
            nn.ReLU(True),
            nn.ConvTranspose1d(depth * 2, 1, 16, 4, 6),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class AudioDiscriminator(nn.Module):
    def __init__(self, input_len=16384, depth=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, depth, 16, 4, 6),
            nn.LeakyReLU(0.2),
            nn.Conv1d(depth, depth * 2, 16, 4, 6),
            nn.LeakyReLU(0.2),
            nn.Conv1d(depth * 2, depth * 4, 16, 4, 6),
            nn.LeakyReLU(0.2),
            nn.Conv1d(depth * 4, 1, 16, 4, 6),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def train_audio_gan(generator, discriminator, dataset, latent_dim=100, lr=2e-4, betas=(0.5, 0.999),
                    epochs=10, batch_size=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    G, D = generator.to(device), discriminator.to(device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=betas)
    criterion = nn.BCELoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for waveforms, _ in loader:
            waveforms = waveforms.to(device)
            bs = waveforms.size(0)
            real_labels = torch.ones(bs, device=device)
            fake_labels = torch.zeros(bs, device=device)

            # Discriminator
            opt_D.zero_grad()
            real_loss = criterion(D(waveforms), real_labels)
            z = torch.randn(bs, latent_dim, 1, device=device)
            fake_audio = G(z)
            fake_loss = criterion(D(fake_audio.detach()), fake_labels)
            D_loss = real_loss + fake_loss
            D_loss.backward()
            opt_D.step()

            # Generator
            opt_G.zero_grad()
            G_loss = criterion(D(fake_audio), real_labels)
            G_loss.backward()
            opt_G.step()

        print(f"[AudioGAN] Epoch {epoch+1}/{epochs} | D_loss={D_loss.item():.4f} | G_loss={G_loss.item():.4f}")


def generate_audio(generator, n_samples, latent_dim=100, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(n_samples, latent_dim, 1, device=device)
    with torch.no_grad():
        audio = generator.to(device)(z)
    return audio.cpu()


class TabularGenerator(nn.Module):
    def __init__(self, input_dim, latent_dim=64, depth=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, depth),
            nn.ReLU(True),
            nn.Linear(depth, depth),
            nn.ReLU(True),
            nn.Linear(depth, input_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class TabularDiscriminator(nn.Module):
    def __init__(self, input_dim, depth=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, depth),
            nn.LeakyReLU(0.2),
            nn.Linear(depth, depth // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(depth // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def train_tabular_gan(generator, discriminator, dataframe, latent_dim=64, lr=2e-4, betas=(0.5, 0.999),
                      epochs=20, batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    G, D = generator.to(device), discriminator.to(device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=betas)
    criterion = nn.BCELoss()
    X = torch.tensor(dataframe.values, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for (x_batch,) in loader:
            bs = x_batch.size(0)
            real_labels = torch.ones(bs, device=device)
            fake_labels = torch.zeros(bs, device=device)

            # Discriminator
            opt_D.zero_grad()
            real_loss = criterion(D(x_batch), real_labels)
            z = torch.randn(bs, latent_dim, device=device)
            fake_data = G(z)
            fake_loss = criterion(D(fake_data.detach()), fake_labels)
            D_loss = real_loss + fake_loss
            D_loss.backward()
            opt_D.step()

            # Generator
            opt_G.zero_grad()
            G_loss = criterion(D(fake_data), real_labels)
            G_loss.backward()
            opt_G.step()

        print(f"[TabularGAN] Epoch {epoch+1}/{epochs} | D_loss={D_loss.item():.4f} | G_loss={G_loss.item():.4f}")


def generate_tabular(generator, n_samples, latent_dim=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(n_samples, latent_dim, device=device)
    with torch.no_grad():
        synthetic = generator.to(device)(z)
    return synthetic.cpu().numpy()
