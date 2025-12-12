# universal_gan.py
# A universal, parameterized GAN framework for image, audio, and tabular data.
# FIXED VERSION: Resolves resolution calculation bugs and architecture issues

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple, Optional

def get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu": return nn.ReLU(inplace=True)
    if name == "leakyrelu": return nn.LeakyReLU(0.2, inplace=True)
    if name == "gelu": return nn.GELU()
    if name == "tanh": return nn.Tanh()
    if name == "sigmoid": return nn.Sigmoid()
    if name == "none": return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")

def get_norm2d(norm_type: str, num_features: int) -> Optional[nn.Module]:
    nt = (norm_type or "batch").lower()
    if nt == "batch": return nn.BatchNorm2d(num_features)
    if nt == "instance": return nn.InstanceNorm2d(num_features, affine=True)
    if nt == "layer": return nn.GroupNorm(1, num_features)
    if nt in ("spectral", "none"): return None
    raise ValueError(f"Unsupported 2D normalization: {norm_type}")

def get_norm1d(norm_type: str, num_features: int) -> Optional[nn.Module]:
    nt = (norm_type or "batch").lower()
    if nt == "batch": return nn.BatchNorm1d(num_features)
    if nt == "instance": return nn.InstanceNorm1d(num_features, affine=True)
    if nt == "layer": return nn.GroupNorm(1, num_features)
    if nt in ("spectral", "none"): return None
    raise ValueError(f"Unsupported 1D normalization: {norm_type}")

def apply_spectral_if_needed(module: nn.Module, spectral: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if spectral else module

def make_optimizer(params, opt_type: str, lr: float, betas: Tuple[float, float], weight_decay: float = 0.0):
    o = (opt_type or "adam").lower()
    if o == "adam": return optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if o == "adamw": return optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if o == "rmsprop": return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if o == "sgd": return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unsupported optimizer_type: {opt_type}")

# -----------------------------------------------------------------------------
# Base Interfaces
# -----------------------------------------------------------------------------
class BaseGenerator(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class BaseDiscriminator(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# -----------------------------------------------------------------------------
# FIXED Image Models (2D)
# -----------------------------------------------------------------------------
class ImageGenerator(BaseGenerator):
    """
    Fixed image generator with stable resolution handling.
    Always starts at 4x4 and progressively upsamples using Upsample + Conv.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        C, H, W = tuple(config.get("img_shape", (3, 64, 64)))
        latent_dim = int(config.get("latent_dim", 100))
        base = int(config.get("base_channels", 64))
        depth = int(config.get("depth", 4))
        norm_type = config.get("norm_type", "batch")
        act_g = config.get("activation_g", "relu")
        out_act = config.get("output_activation", "tanh")

        # Always start at 4x4 for stability
        self.init_size = 4
        self.latent_dim = latent_dim
        self.target_h = H
        self.target_w = W

        # Project latent to initial spatial size
        init_channels = base * (2 ** (depth - 1))
        self.project = nn.Sequential(
            nn.Linear(latent_dim, init_channels * self.init_size * self.init_size),
            get_activation(act_g)
        )

        # Build upsampling blocks
        layers = []
        in_ch = init_channels

        for i in range(depth):
            out_ch = max(base, in_ch // 2)

            # Upsample + Conv (more stable than ConvTranspose2d)
            layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                get_norm2d(norm_type, out_ch),
                get_activation(act_g)
            ])
            in_ch = out_ch

        # Final conv to output channels
        layers.append(nn.Conv2d(in_ch, C, 3, 1, 1))
        layers.append(get_activation(out_act))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Handle (N, latent_dim) or (N, latent_dim, 1, 1) or (N, latent_dim, H, W)
        if z.dim() == 2:
            bs = z.size(0)
            x = self.project(z)
            x = x.view(bs, -1, self.init_size, self.init_size)
        elif z.dim() == 4:
            bs, ld = z.size(0), z.size(1)
            z_flat = z.view(bs, ld)
            x = self.project(z_flat)
            x = x.view(bs, -1, self.init_size, self.init_size)
        else:
            raise ValueError(f"Unsupported latent shape: {z.shape}")

        # Pass through network
        x = self.net(x)

        # Ensure exact output size (handles non-power-of-2 resolutions)
        if x.shape[2] != self.target_h or x.shape[3] != self.target_w:
            x = F.interpolate(x, size=(self.target_h, self.target_w),
                            mode='bilinear', align_corners=False)

        return x


class ImageDiscriminator(BaseDiscriminator):
    """
    Fixed image discriminator with stable downsampling.
    Uses global pooling to avoid dimension collapse.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        C, H, W = tuple(config.get("img_shape"))
        base = int(config.get("base_channels", 64))
        depth = int(config.get("depth", 4))
        norm_type = config.get("norm_type", "batch")
        act_d = config.get("activation_d", "leakyrelu")
        spectral = bool(config.get("spectral_norm", False))
        final_act = config.get("final_activation", "sigmoid").lower()

        def maybe_spectral(layer):
            return nn.utils.spectral_norm(layer) if spectral else layer

        layers = []
        in_ch = C
        ch = base

        # Build depth layers of downsampling
        for i in range(depth):
            # Use stride=2 for clean downsampling
            layers.append(maybe_spectral(
                nn.Conv2d(in_ch, ch, 4, 2, 1, bias=False)
            ))

            # Skip norm on first layer (standard practice)
            if i != 0:
                norm = get_norm2d(norm_type, ch)
                if norm:
                    layers.append(norm)

            layers.append(get_activation(act_d))

            in_ch = ch
            ch = min(ch * 2, base * 8)  # Cap at 8x base to prevent explosion

        # Use global pooling instead of final conv that might collapse dimensions
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(maybe_spectral(nn.Linear(in_ch, 1)))

        if final_act != "none":
            layers.append(get_activation(final_act))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(1) if out.dim() > 1 else out

# -----------------------------------------------------------------------------
# Audio Models (1D)
# -----------------------------------------------------------------------------
class AudioGenerator(BaseGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        latent_dim = int(config.get("latent_dim", 100))
        base = int(config.get("base_channels", 64))
        depth = int(config.get("depth", 4))
        up_mode = config.get("upsample_mode_1d", "convtranspose1d").lower()
        k = int(config.get("kernel_size_1d", 16))
        s = int(config.get("stride_1d", 4))
        p = int(config.get("padding_1d", 6))
        norm_type = config.get("norm_type", "batch")
        act_g = config.get("activation_g", "relu")
        out_act = config.get("output_activation", "tanh")

        layers = []
        in_ch = latent_dim
        ch = base * (2 ** (depth - 1))

        for i in range(depth):
            if up_mode == "convtranspose1d":
                block = [nn.ConvTranspose1d(in_ch, ch, k, s, p, bias=False)]
            elif up_mode == "upsample_conv1d":
                block = [nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv1d(in_ch, ch, k, stride=1, padding=p, bias=False)]
            else:
                raise ValueError(f"Invalid upsample_mode_1d: {up_mode}")

            layers.extend(block)
            norm = get_norm1d(norm_type, ch)
            if norm:
                layers.append(norm)
            layers.append(get_activation(act_g))

            in_ch = ch
            ch = max(base, ch // 2)

        layers.append(nn.Conv1d(in_ch, 1, 3, 1, 1, bias=False))
        layers.append(get_activation(out_act))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AudioDiscriminator(BaseDiscriminator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        base = int(config.get("base_channels", 64))
        depth = int(config.get("depth", 4))
        k = int(config.get("kernel_size_1d", 16))
        s = int(config.get("stride_1d", 4))
        p = int(config.get("padding_1d", 6))
        norm_type = config.get("norm_type", "batch")
        act_d = config.get("activation_d", "leakyrelu")
        spectral = bool(config.get("spectral_norm", False))
        final_act = config.get("final_activation", "sigmoid").lower()

        layers = []
        in_ch = 1
        ch = base

        for i in range(depth):
            conv = nn.Conv1d(in_ch, ch, k, s, p, bias=False)
            conv = apply_spectral_if_needed(conv, spectral)
            layers.append(conv)

            if i != 0:
                norm = get_norm1d(norm_type, ch)
                if norm:
                    layers.append(norm)

            layers.append(get_activation(act_d))

            in_ch = ch
            ch *= 2

        layers.append(apply_spectral_if_needed(nn.Conv1d(in_ch, 1, 4, 1, 0, bias=False), spectral))

        if final_act != "none":
            layers.append(get_activation(final_act))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1).squeeze(1)

# -----------------------------------------------------------------------------
# Tabular Models (MLP)
# -----------------------------------------------------------------------------
class TabularGenerator(BaseGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        input_dim = int(config["input_dim"])
        latent_dim = int(config.get("latent_dim", 64))
        hidden_dim = int(config.get("hidden_dim", 256))
        n_layers = int(config.get("num_hidden_layers", 2))
        act_g = config.get("activation_g", "relu")
        out_act = config.get("output_activation", "tanh")

        layers = [nn.Linear(latent_dim, hidden_dim), get_activation(act_g)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), get_activation(act_g)])
        layers.append(nn.Linear(hidden_dim, input_dim))
        layers.append(get_activation(out_act))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TabularDiscriminator(BaseDiscriminator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        input_dim = int(config["input_dim"])
        hidden_dim = int(config.get("hidden_dim", 256))
        n_layers = int(config.get("num_hidden_layers", 2))
        act_d = config.get("activation_d", "leakyrelu")
        final_act = config.get("final_activation", "sigmoid").lower()

        layers = [nn.Linear(input_dim, hidden_dim), get_activation(act_d)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), get_activation(act_d)])
        layers.append(nn.Linear(hidden_dim, 1))

        if final_act != "none":
            layers.append(get_activation(final_act))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1).squeeze(1)

# -----------------------------------------------------------------------------
# Universal Trainer
# -----------------------------------------------------------------------------
CLASS_MAP = {
    "image": (ImageGenerator, ImageDiscriminator),
    "audio": (AudioGenerator, AudioDiscriminator),
    "tabular": (TabularGenerator, TabularDiscriminator)
}

class UniversalGAN:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.modality = self.cfg.get("modality", "image").lower()

        if self.modality not in CLASS_MAP:
            raise ValueError(f"Unsupported modality: {self.modality}")

        self.device = self.cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        G_cls, D_cls = CLASS_MAP[self.modality]
        self.G = G_cls(self.cfg).to(self.device)
        self.D = D_cls(self.cfg).to(self.device)

        self.latent_dim = int(self.cfg.get("latent_dim", 100))
        self.opt_type = self.cfg.get("optimizer_type", "adam")
        self.lr_g = float(self.cfg.get("lr_g", 2e-4))
        self.lr_d = float(self.cfg.get("lr_d", 2e-4))
        self.betas = tuple(self.cfg.get("betas", (0.5, 0.999)))
        self.weight_decay = float(self.cfg.get("weight_decay", 0.0))

        self.loss_type = self.cfg.get("loss_type", "bce").lower()
        self.lambda_gp = float(self.cfg.get("lambda_gp", 10.0))
        self.n_critic = int(self.cfg.get("n_critic", 1 if self.loss_type in ("bce", "bce_logits") else 5))

        self.use_amp = bool(self.cfg.get("amp", True))
        self.grad_clip = float(self.cfg.get("grad_clip", 0.0))

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.opt_G = make_optimizer(self.G.parameters(), self.opt_type, self.lr_g, self.betas, self.weight_decay)
        self.opt_D = make_optimizer(self.D.parameters(), self.opt_type, self.lr_d, self.betas, self.weight_decay)

        if self.loss_type == "bce":
            self.criterion = nn.BCELoss()
        elif self.loss_type == "bce_logits":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = None

    def sample_latent(self, bs: int) -> torch.Tensor:
        if self.modality == "image":
            return torch.randn(bs, self.latent_dim, 1, 1, device=self.device)
        if self.modality == "audio":
            return torch.randn(bs, self.latent_dim, 1, device=self.device)
        if self.modality == "tabular":
            return torch.randn(bs, self.latent_dim, device=self.device)
        raise RuntimeError("Unknown modality")

    def _grad_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        bs = real.size(0)
        eps = torch.rand(bs, *(1 for _ in range(real.dim() - 1)), device=self.device)
        x_hat = eps * real + (1 - eps) * fake
        x_hat.requires_grad_(True)

        d_hat = self.D(x_hat)
        grads = torch.autograd.grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=torch.ones_like(d_hat),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        grads = grads.view(bs, -1)
        return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()

    def train(self, dataset):
        batch_size = int(self.cfg.get("batch_size", 64))
        epochs = int(self.cfg.get("epochs", 10))

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        print(f"Training {self.modality.upper()} GAN on {self.device.upper()} | loss={self.loss_type}")

        self.G.train()
        self.D.train()

        for epoch in range(epochs):
            for step, batch in enumerate(loader):
                # Prepare real data
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(self.device)
                bs = x.size(0)

                # Train Discriminator
                self.D.requires_grad_(True)
                self.G.requires_grad_(False)
                self.opt_D.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    real_out = self.D(x)

                    # Real loss
                    if self.loss_type == "bce":
                        smooth_real = float(self.cfg.get("smooth_real", 0.9))
                        real_targets = torch.full_like(real_out, smooth_real)
                        real_loss = self.criterion(torch.sigmoid(real_out), real_targets)
                    elif self.loss_type == "bce_logits":
                        real_targets = torch.ones_like(real_out)
                        real_loss = self.criterion(real_out, real_targets)
                    elif self.loss_type == "hinge":
                        real_loss = torch.mean(torch.relu(1.0 - real_out))
                    elif self.loss_type == "wgan_gp":
                        real_loss = -real_out.mean()

                    # Fake loss
                    z = self.sample_latent(bs)
                    fake = self.G(z).detach()
                    fake_out = self.D(fake)

                    if self.loss_type == "bce":
                        fake_loss = self.criterion(torch.sigmoid(fake_out), torch.zeros_like(fake_out))
                        D_loss = 0.5 * (real_loss + fake_loss)
                    elif self.loss_type == "bce_logits":
                        fake_loss = self.criterion(fake_out, torch.zeros_like(fake_out))
                        D_loss = 0.5 * (real_loss + fake_loss)
                    elif self.loss_type == "hinge":
                        fake_loss = torch.mean(torch.relu(1.0 + fake_out))
                        D_loss = real_loss + fake_loss
                    elif self.loss_type == "wgan_gp":
                        fake_for_gp = self.G(self.sample_latent(bs))
                        gp = self._grad_penalty(x, fake_for_gp)
                        D_loss = real_loss + fake_out.mean() + self.lambda_gp * gp

                self.scaler.scale(D_loss).backward()

                if self.grad_clip > 0:
                    self.scaler.unscale_(self.opt_D)
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)

                self.scaler.step(self.opt_D)
                self.scaler.update()

                # Only update generator every n_critic steps
                if (step % self.n_critic) != 0:
                    continue

                # Train Generator
                self.D.requires_grad_(False)
                self.G.requires_grad_(True)
                self.opt_G.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    z = self.sample_latent(bs)
                    gen = self.G(z)
                    gen_out = self.D(gen)

                    if self.loss_type in ("bce", "bce_logits"):
                        G_loss = -torch.log(torch.sigmoid(gen_out) + 1e-8).mean()
                    elif self.loss_type in ("hinge", "wgan_gp"):
                        G_loss = -gen_out.mean()

                self.scaler.scale(G_loss).backward()

                if self.grad_clip > 0:
                    self.scaler.unscale_(self.opt_G)
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.grad_clip)

                self.scaler.step(self.opt_G)
                self.scaler.update()

            print(f"[{epoch+1}/{epochs}] D={D_loss.item():.4f} | G={G_loss.item():.4f}")

        print("✅ Training complete.")
        return self.G, self.D

    def generate(self, n_samples: int) -> torch.Tensor:
        self.G.eval()
        z = self.sample_latent(n_samples)
        with torch.no_grad():
            out = self.G(z)
        return out.detach().cpu()