"""
Baseline GAN implementations for comparison with Quantum GAN.
All baselines use equivalent architecture (same cond_dim, attention, etc.) for fair comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np


# ============================================================================
# Shared Components (import from main notebook or redefine here)
# ============================================================================

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)

    def forward(self, labels):
        return self.embed(labels)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=7):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        attn = torch.softmax(Q @ K.transpose(-1, -2) * self.scale, dim=-1)
        out = (attn @ V).reshape(batch_size, -1)
        return self.out_proj(out) + x


# ============================================================================
# 1. Vanilla GAN (with conditional embedding + attention for fair comparison)
# ============================================================================

class ClassicalBottleneck(nn.Module):
    """Classical bottleneck layer matching quantum encoder capacity."""
    def __init__(self, input_dim, bottleneck_dim, output_dim, n_layers=2):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        # Pre-encoding
        self.pre_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim),
            nn.Tanh()
        )

        # Classical bottleneck layers (matching quantum variational layers)
        classical_layers = []
        for _ in range(n_layers):
            classical_layers.extend([
                nn.Linear(bottleneck_dim, bottleneck_dim),
                nn.Tanh()
            ])
        self.bottleneck = nn.Sequential(*classical_layers)

        # Decoding
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.pre_encoder(x)
        z = self.bottleneck(z)
        return self.decoder(z)


class VanillaGenerator(nn.Module):
    """Generator with classical bottleneck + attention (fair comparison to quantum)."""
    def __init__(self, latent_dim, feature_dim, cond_dim, bottleneck_dim=16, n_layers=2):
        super().__init__()
        self.cond_emb = nn.Linear(cond_dim, latent_dim)
        self.classical_encoder = ClassicalBottleneck(latent_dim, bottleneck_dim, feature_dim, n_layers)
        # Refinement layers - same as quantum generator
        self.attn1 = MultiHeadSelfAttention(feature_dim, num_heads=7)
        self.attn2 = MultiHeadSelfAttention(feature_dim, num_heads=7)

    def forward(self, z, cond):
        z = z + self.cond_emb(cond)
        out = self.classical_encoder(z)
        out = self.attn1(out)
        out = self.attn2(out)
        return out


class VanillaDiscriminator(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim + cond_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        return self.model(torch.cat([x, cond], dim=1))


def train_vanilla_gan(X_train, y_train, config, device):
    """Train Vanilla GAN with classical bottleneck (fair comparison to quantum)."""
    latent_dim = config['latent_dim']
    input_dim = config['input_dim']
    cond_dim = config['cond_dim']
    batch_size = config['batch_size']
    epochs = config.get('epochs', 8)
    lr = config.get('lr', 1e-4)
    bottleneck_dim = config.get('bottleneck_dim', 16)  # Matched to quantum dim

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )

    cond_embed = ConditionalEmbedding(2, cond_dim).to(device)
    G = VanillaGenerator(latent_dim, input_dim, cond_dim, bottleneck_dim=bottleneck_dim).to(device)
    D = VanillaDiscriminator(input_dim, cond_dim).to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for real_samples, labels in tqdm(train_loader, desc=f"Vanilla Epoch {epoch+1}"):
            real_samples = real_samples.to(device)
            labels = labels.to(device)
            cond = cond_embed(labels)
            batch_size = real_samples.size(0)

            # Train D
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = G(z, cond).detach()
            D_real = D(real_samples, cond)
            D_fake = D(fake, cond)
            loss_D = criterion(D_real, torch.ones_like(D_real)) + criterion(D_fake, torch.zeros_like(D_fake))
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Train G
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = G(z, cond)
            D_fake = D(fake, cond)
            loss_G = criterion(D_fake, torch.ones_like(D_fake))
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        print(f"Vanilla Epoch {epoch+1} | D: {loss_D.item():.4f}, G: {loss_G.item():.4f}")

    return G, D, cond_embed


# ============================================================================
# 2. WGAN (with conditional embedding for fair comparison)
# ============================================================================

class WGANCritic(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim + cond_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)  # No sigmoid
        )

    def forward(self, x, cond):
        return self.model(torch.cat([x, cond], dim=1))


def train_wgan(X_train, y_train, config, device):
    """Train WGAN with weight clipping."""
    latent_dim = config['latent_dim']
    input_dim = config['input_dim']
    cond_dim = config['cond_dim']
    batch_size = config['batch_size']
    epochs = config.get('epochs', 8)
    lr = config.get('lr', 5e-5)
    n_critic = config.get('n_critic', 5)
    clip_value = config.get('clip_value', 0.01)
    bottleneck_dim = config.get('bottleneck_dim', 16)  # Matched to quantum dim

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )

    cond_embed = ConditionalEmbedding(2, cond_dim).to(device)
    G = VanillaGenerator(latent_dim, input_dim, cond_dim, bottleneck_dim=bottleneck_dim).to(device)
    D = WGANCritic(input_dim, cond_dim).to(device)

    optimizer_G = optim.RMSprop(G.parameters(), lr=lr)
    optimizer_D = optim.RMSprop(D.parameters(), lr=lr)

    for epoch in range(epochs):
        for real_samples, labels in tqdm(train_loader, desc=f"WGAN Epoch {epoch+1}"):
            real_samples = real_samples.to(device)
            labels = labels.to(device)
            cond = cond_embed(labels)
            batch_size = real_samples.size(0)

            # Train Critic
            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim, device=device)
                fake = G(z, cond).detach()
                D_real = D(real_samples, cond)
                D_fake = D(fake, cond)
                loss_D = -torch.mean(D_real) + torch.mean(D_fake)
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

                # Weight clipping
                for p in D.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Train Generator
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = G(z, cond)
            D_fake = D(fake, cond)
            loss_G = -torch.mean(D_fake)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        print(f"WGAN Epoch {epoch+1} | D: {loss_D.item():.4f}, G: {loss_G.item():.4f}")

    return G, D, cond_embed


# ============================================================================
# 3. WGAN-GP (with conditional embedding for fair comparison)
# ============================================================================

def compute_gradient_penalty(D, real, fake, cond):
    """Compute gradient penalty for WGAN-GP."""
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates, cond)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(X_train, y_train, config, device):
    """Train WGAN-GP."""
    latent_dim = config['latent_dim']
    input_dim = config['input_dim']
    cond_dim = config['cond_dim']
    batch_size = config['batch_size']
    epochs = config.get('epochs', 8)
    lr = config.get('lr', 1e-4)
    n_critic = config.get('n_critic', 5)
    lambda_gp = config.get('lambda_gp', 10)
    bottleneck_dim = config.get('bottleneck_dim', 16)  # Matched to quantum dim

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )

    cond_embed = ConditionalEmbedding(2, cond_dim).to(device)
    G = VanillaGenerator(latent_dim, input_dim, cond_dim, bottleneck_dim=bottleneck_dim).to(device)
    D = WGANCritic(input_dim, cond_dim).to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    for epoch in range(epochs):
        for real_samples, labels in tqdm(train_loader, desc=f"WGAN-GP Epoch {epoch+1}"):
            real_samples = real_samples.to(device)
            labels = labels.to(device)
            cond = cond_embed(labels)
            batch_size = real_samples.size(0)

            # Train Critic
            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim, device=device)
                fake = G(z, cond).detach()
                D_real = D(real_samples, cond)
                D_fake = D(fake, cond)
                gp = compute_gradient_penalty(D, real_samples, fake, cond)
                loss_D = -torch.mean(D_real) + torch.mean(D_fake) + lambda_gp * gp
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            # Train Generator
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = G(z, cond)
            D_fake = D(fake, cond)
            loss_G = -torch.mean(D_fake)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        print(f"WGAN-GP Epoch {epoch+1} | D: {loss_D.item():.4f}, G: {loss_G.item():.4f}")

    return G, D, cond_embed


# ============================================================================
# Classifier for evaluating generated data
# ============================================================================

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_and_evaluate_classifier(X_train_balanced, y_train_balanced, X_test, y_test,
                                   config, device, model_name="Model"):
    """Train classifier on balanced data and evaluate."""
    input_dim = config['input_dim']
    batch_size = config.get('batch_size', 64)
    epochs = config.get('clf_epochs', 5)
    lr = config.get('lr', 1e-4)

    clf = SimpleClassifier(input_dim).to(device)

    # Class-weighted loss
    num_pos = (y_train_balanced == 1).sum().item()
    num_neg = (y_train_balanced == 0).sum().item()
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(clf.parameters(), lr=lr, betas=(0.9, 0.999))

    train_loader = DataLoader(
        TensorDataset(X_train_balanced, y_train_balanced.float()),
        batch_size=batch_size, shuffle=True
    )

    for ep in range(epochs):
        clf.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = criterion(clf(Xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    clf.eval()
    y_scores = []
    y_true = []

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            y_true.append(yb.cpu().numpy())
            y_scores.append(torch.sigmoid(clf(Xb)).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)
    y_pred = (y_scores >= 0.5).astype(int)

    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    print(f"\n[{model_name}] Results:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")

    return clf, results
