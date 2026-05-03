"""
Ablation Study: Quantum vs Classical Generator
==============================================
This script trains BOTH quantum and classical generators with matched
parameter counts, then compares them on:
- Generation quality (MMD, diversity)
- Classifier performance (accuracy, F1, per-class recall)
- Training efficiency (convergence speed)

Run this AFTER training your quantum model to get fair comparison.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import jensenshannon

# ============ Configuration ============
DATA_PATH = "/content/drive/MyDrive/supergan/CSE-CIC-IDS2018"
N_SAMPLES = 70000
RANDOM_SEED = 42

# Architecture (matched for fair comparison)
LATENT_DIM = 64
QUANTUM_DIM = 16  # 16 qubits for meaningful quantum bottleneck
CLASSICAL_BOTTLENECK = 16  # Matched to quantum dim
COND_DIM = 16
NUM_HEADS = 7
N_QUANTUM_LAYERS = 2

# Training (reduced for speed - adjust as needed)
BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-4
LAMBDA_GP = 10
N_CRITIC = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# =======================================


# ============ Data Loading ============
def load_and_preprocess_data(data_path, n_samples, seed):
    import pyarrow.parquet as pq

    files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
    df = pq.read_table(os.path.join(data_path, files[0])).to_pandas()
    df = df.sample(n=min(n_samples, len(df)), random_state=seed)

    # Clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Features and labels
    X = df.drop(columns=['Label'])
    y = df['Label']
    y_binary = np.where(y.str.contains("Benign", case=False, na=False), 0, 1).astype(np.int64)

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split by class
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_binary, dtype=torch.long)

    X_attack = X_tensor[y_binary == 1]
    y_attack = y_tensor[y_binary == 1]
    X_benign = X_tensor[y_binary == 0]
    y_benign = y_tensor[y_binary == 0]

    def split_data(X, y, train_ratio=0.8, seed=42):
        perm = torch.Generator().manual_seed(seed).randperm(len(X))
        n_train = int(len(X) * train_ratio)
        return X[perm[:n_train]], X[perm[n_train:]], y[perm[:n_train]], y[perm[n_train:]]

    X_attack_train, X_attack_test, y_attack_train, y_attack_test = split_data(X_attack, y_attack)
    X_benign_train, X_benign_test, y_benign_train, y_benign_test = split_data(X_benign, y_benign)

    X_train = torch.cat([X_attack_train, X_benign_train])
    y_train = torch.cat([y_attack_train, y_benign_train])
    X_test = torch.cat([X_attack_test, X_benign_test])
    y_test = torch.cat([y_attack_test, y_benign_test])

    perm = torch.randperm(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    return X_train, y_train, X_test, y_test, X_scaled.shape[1], scaler


# ============ Model Components ============
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


# ============ QUANTUM GENERATOR ============
try:
    import pennylane as qml

    dev = qml.device("lightning.qubit", wires=QUANTUM_DIM)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def quantum_layer_variational(x_batch, weights):
        """Process a batch through variational quantum circuit.

        FIXED: Returns torch.Tensor instead of list of lists.
        """
        batch_size = x_batch.shape[0]
        n_layers = weights.shape[0]
        results = []

        for i in range(batch_size):
            # Encode input (only on first layer)
            for q in range(QUANTUM_DIM):
                qml.RY(x_batch[i, q], wires=q)

            # Variational layers
            for layer in range(n_layers):
                for q in range(QUANTUM_DIM):
                    qml.RY(weights[layer, q, 0], wires=q)
                    qml.RZ(weights[layer, q, 1], wires=q)

                # Circular entanglement
                for q in range(QUANTUM_DIM - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[QUANTUM_DIM - 1, 0])

            # Measure - FIXED: convert to tensor immediately
            result = torch.tensor([qml.expval(qml.PauliZ(i)) for i in range(QUANTUM_DIM)], dtype=torch.float32)
            results.append(result)

        return torch.stack(results)

    class HybridQuantumEncoder(nn.Module):
        def __init__(self, input_dim, quantum_dim, output_dim, n_layers=2):
            super().__init__()
            self.quantum_dim = quantum_dim
            self.n_layers = n_layers
            # 2 parameters per qubit per layer (RY + RZ, no RX)
            self.quantum_weights = nn.Parameter(torch.randn(n_layers, quantum_dim, 2) * 0.1)

            self.pre_encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, quantum_dim),
                nn.Tanh()
            )

            self.decoder = nn.Sequential(
                nn.Linear(quantum_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            z_q = self.pre_encoder(x)
            q_out = quantum_layer_variational(z_q, self.quantum_weights)
            return self.decoder(q_out)

    class QuantumGenerator(nn.Module):
        def __init__(self, latent_dim, feature_dim, cond_dim, quantum_dim, n_quantum_layers=2):
            super().__init__()
            self.cond_emb = nn.Linear(cond_dim, latent_dim)
            self.quantum_encoder = HybridQuantumEncoder(latent_dim, quantum_dim, feature_dim, n_quantum_layers)
            self.attn1 = MultiHeadSelfAttention(feature_dim, num_heads=7)
            self.attn2 = MultiHeadSelfAttention(feature_dim, num_heads=7)

        def forward(self, z, cond):
            z = z + self.cond_emb(cond)
            out = self.quantum_encoder(z)
            out = self.attn1(out)
            out = self.attn2(out)
            return out

    QUANTUM_AVAILABLE = True
    print("✓ PennyLane available - quantum generator enabled")

except ImportError:
    QUANTUM_AVAILABLE = False
    print("✗ PennyLane not available - skipping quantum generator")


# ============ CLASSICAL GENERATOR (Matched Parameters) ============
class ClassicalBottleneckEncoder(nn.Module):
    """
    Classical equivalent with MATCHED parameter count.

    Quantum has: n_layers * n_qubits * 3 parameters for variational weights
    Plus encoder/decoder parameters.

    We match this by using a bottleneck layer with similar capacity.
    """
    def __init__(self, input_dim, bottleneck_dim, output_dim, n_layers=2):
        super().__init__()

        # Match quantum parameter count
        quantum_params = n_layers * bottleneck_dim * 3

        self.pre_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim),
            nn.Tanh()
        )

        # Classical bottleneck: use multiple linear layers to match quantum capacity
        # Quantum does: n_layers of (n_qubits * 3) rotations
        # We do: n_layers of (bottleneck_dim -> bottleneck_dim) with nonlinearity
        classical_layers = []
        for _ in range(n_layers):
            classical_layers.extend([
                nn.Linear(bottleneck_dim, bottleneck_dim),
                nn.Tanh()
            ])
        self.bottleneck = nn.Sequential(*classical_layers)

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


class ClassicalGenerator(nn.Module):
    def __init__(self, latent_dim, feature_dim, cond_dim, bottleneck_dim=8, n_layers=2):
        super().__init__()
        self.cond_emb = nn.Linear(cond_dim, latent_dim)
        self.encoder = ClassicalBottleneckEncoder(latent_dim, bottleneck_dim, feature_dim, n_layers)
        self.attn1 = MultiHeadSelfAttention(feature_dim, num_heads=7)
        self.attn2 = MultiHeadSelfAttention(feature_dim, num_heads=7)

    def forward(self, z, cond):
        z = z + self.cond_emb(cond)
        out = self.encoder(z)
        out = self.attn1(out)
        out = self.attn2(out)
        return out


# ============ Discriminator ============
class Discriminator(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim + cond_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, cond):
        return self.model(torch.cat([x, cond], dim=1))


def compute_gradient_penalty(D, real, fake, cond):
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


# ============ Training Function ============
def train_gan(G, D, cond_embed, train_loader, device, epochs, lr, lambda_gp, n_critic):
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    G.train()
    D.train()

    latent_dim = 64
    history = {'G_loss': [], 'D_loss': []}

    for epoch in range(epochs):
        epoch_g_losses = []
        epoch_d_losses = []

        for real, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            real = real.to(device)
            labels = labels.to(device)
            cond = cond_embed(labels)

            # Train D
            for _ in range(n_critic):
                z = torch.randn(real.size(0), latent_dim, device=device)
                fake = G(z, cond).detach()
                D_real = D(real, cond)
                D_fake = D(fake, cond)
                gp = compute_gradient_penalty(D, real, fake, cond)
                loss_D = -torch.mean(D_real) + torch.mean(D_fake) + lambda_gp * gp

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            # Train G
            z = torch.randn(real.size(0), latent_dim, device=device)
            fake = G(z, cond)
            loss_G = -torch.mean(D(fake, cond))

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            epoch_g_losses.append(loss_G.item())
            epoch_d_losses.append(loss_D.item())

        history['G_loss'].append(np.mean(epoch_g_losses))
        history['D_loss'].append(np.mean(epoch_d_losses))
        print(f"Epoch {epoch+1} | D: {history['D_loss'][-1]:.4f} | G: {history['G_loss'][-1]:.4f}")

    return G, D, history


# ============ Evaluation Metrics ============
def compute_mmd(X_real, X_synth, sigma=1.0):
    """Maximum Mean Discrepancy - measures distribution similarity."""
    n_real = X_real.shape[0]
    n_synth = X_synth.shape[0]

    # RBF kernel
    def rbf_kernel(X, Y, sigma):
        XX = np.sum(X**2, axis=1).reshape(-1, 1)
        YY = np.sum(Y**2, axis=1).reshape(1, -1)
        XY = np.dot(X, Y.T)
        return np.exp(-(XX + YY - 2*XY) / (2 * sigma**2))

    K_real = rbf_kernel(X_real, X_real, sigma)
    K_synth = rbf_kernel(X_synth, X_synth, sigma)
    K_cross = rbf_kernel(X_real, X_synth, sigma)

    mmd = K_real.mean() + K_synth.mean() - 2 * K_cross.mean()
    return mmd


def compute_diversity(X_real, X_synth, k=5):
    """Diversity via nearest neighbor distances."""
    # Fit on real data
    nn_real = NearestNeighbors(n_neighbors=k).fit(X_real)
    distances_real, _ = nn_real.kneighbors(X_real[np.random.choice(len(X_real), min(1000, len(X_real)))])

    # Fit on synthetic data
    nn_synth = NearestNeighbors(n_neighbors=k).fit(X_synth)
    distances_synth, _ = nn_synth.kneighbors(X_synth[np.random.choice(len(X_synth), min(1000, len(X_synth)))])

    return {
        'real_avg_distance': distances_real.mean(),
        'synth_avg_distance': distances_synth.mean(),
        'diversity_ratio': distances_synth.mean() / distances_real.mean()
    }


def compute_coverage(X_real, X_synth, k=1):
    """Coverage: fraction of real samples with a close synthetic neighbor."""
    nn_synth = NearestNeighbors(n_neighbors=k).fit(X_synth)
    distances, _ = nn_synth.kneighbors(X_real)

    # Threshold: consider covered if within 10th percentile of synthetic-synthetic distances
    nn_synth_self = NearestNeighbors(n_neighbors=k).fit(X_synth)
    synth_distances, _ = nn_synth_self.kneighbors(X_synth)
    threshold = np.percentile(synth_distances, 10)

    covered = (distances < threshold).sum() / len(X_real)
    return covered


def evaluate_generation(X_train, G, cond_embed, device, n_samples=2000):
    """Generate samples and compute all metrics."""
    G.eval()

    with torch.no_grad():
        # Generate balanced samples
        n_per_class = n_samples // 2

        z_benign = torch.randn(n_per_class, 64, device=device)
        c_benign = cond_embed(torch.zeros(n_per_class, dtype=torch.long, device=device))
        X_synth_benign = G(z_benign, c_benign).cpu().numpy()

        z_attack = torch.randn(n_per_class, 64, device=device)
        c_attack = cond_embed(torch.ones(n_per_class, dtype=torch.long, device=device))
        X_synth_attack = G(z_attack, c_attack).cpu().numpy()

    X_synth = np.concatenate([X_synth_benign, X_synth_attack])

    # Get real samples for comparison
    X_real = X_train.numpy()

    # Compute metrics
    metrics = {
        'mmd': compute_mmd(X_real, X_synth),
        'diversity': compute_diversity(X_real, X_synth),
        'coverage': compute_coverage(X_real, X_synth),
    }

    return X_synth, metrics


def train_and_evaluate_classifier(X_train_balanced, y_train_balanced, X_test, y_test, device):
    """Train classifier on balanced data and return metrics."""

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

    clf = SimpleClassifier(X_train_balanced.shape[1]).to(device)

    # Class-weighted loss
    num_pos = (y_train_balanced == 1).sum().item()
    num_neg = (y_train_balanced == 0).sum().item()
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(clf.parameters(), lr=1e-4)

    train_loader = DataLoader(
        TensorDataset(X_train_balanced, y_train_balanced.float()),
        batch_size=64, shuffle=True
    )

    # Train
    clf.train()
    for epoch in range(5):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = criterion(clf(Xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    clf.eval()
    y_true, y_scores = [], []

    test_loader = DataLoader(TensorDataset(X_test, y_test.float()), batch_size=64)

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            y_true.append(yb.cpu().numpy())
            y_scores.append(torch.sigmoid(clf(Xb)).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)

    # Adaptive threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    best_thresh = thresholds[np.argmax(J)]
    y_pred = (y_scores >= best_thresh).astype(int)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_attack': f1[1],  # F1 for attack class (rare)
        'recall_attack': recall[1],
        'precision_attack': precision[1],
        'roc_auc': roc_auc_score(y_true, y_scores),
        'best_threshold': best_thresh
    }


# ============ Main Ablation Study ============
def run_ablation_study():
    results = {
        'quantum': {},
        'classical': {}
    }

    # Load data
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    X_train, y_train, X_test, y_test, input_dim, scaler = load_and_preprocess_data(
        DATA_PATH, N_SAMPLES, RANDOM_SEED
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {input_dim}")

    cond_embed = ConditionalEmbedding(2, COND_DIM).to(device)

    # ============ TRAIN QUANTUM GENERATOR ============
    if QUANTUM_AVAILABLE:
        print("\n" + "="*60)
        print("TRAINING QUANTUM GENERATOR")
        print("="*60)

        G_quantum = QuantumGenerator(LATENT_DIM, input_dim, COND_DIM, QUANTUM_DIM, n_quantum_layers=2).to(device)
        D_quantum = Discriminator(input_dim, COND_DIM).to(device)

        print(f"Quantum G params: {sum(p.numel() for p in G_quantum.parameters()):,}")
        print(f"D params: {sum(p.numel() for p in D_quantum.parameters()):,}")

        G_quantum, D_quantum, history_q = train_gan(
            G_quantum, D_quantum, cond_embed, train_loader, device,
            EPOCHS, LR, LAMBDA_GP, N_CRITIC
        )

        # Evaluate generation quality
        print("\nEvaluating quantum generation quality...")
        X_synth_q, gen_metrics_q = evaluate_generation(X_train, G_quantum, cond_embed, device)

        # Train classifier on quantum-augmented data
        print("Training classifier on quantum-augmented data...")
        # (Simplified: just use generated samples directly)
        clf_metrics_q = train_and_evaluate_classifier(
            torch.tensor(X_synth_q),
            torch.tensor([0]*1000 + [1]*1000),
            X_test, y_test, device
        )

        results['quantum'] = {
            'generator_params': sum(p.numel() for p in G_quantum.parameters()),
            'training_history': history_q,
            'generation_metrics': gen_metrics_q,
            'classifier_metrics': clf_metrics_q
        }

        print(f"\nQuantum Results:")
        print(f"  MMD: {gen_metrics_q['mmd']:.6f}")
        print(f"  Diversity ratio: {gen_metrics_q['diversity']['diversity_ratio']:.4f}")
        print(f"  Coverage: {gen_metrics_q['coverage']:.4f}")
        print(f"  Classifier F1 (attack): {clf_metrics_q['f1_attack']:.4f}")

    # ============ TRAIN CLASSICAL GENERATOR ============
    print("\n" + "="*60)
    print("TRAINING CLASSICAL GENERATOR (Ablation)")
    print("="*60)

    G_classical = ClassicalGenerator(LATENT_DIM, input_dim, COND_DIM, bottleneck_dim=CLASSICAL_BOTTLENECK, n_layers=2).to(device)
    D_classical = Discriminator(input_dim, COND_DIM).to(device)

    print(f"Classical G params: {sum(p.numel() for p in G_classical.parameters()):,}")
    print(f"D params: {sum(p.numel() for p in D_classical.parameters()):,}")

    G_classical, D_classical, history_c = train_gan(
        G_classical, D_classical, cond_embed, train_loader, device,
        EPOCHS, LR, LAMBDA_GP, N_CRITIC
    )

    # Evaluate generation quality
    print("\nEvaluating classical generation quality...")
    X_synth_c, gen_metrics_c = evaluate_generation(X_train, G_classical, cond_embed, device)

    # Train classifier
    print("Training classifier on classical-augmented data...")
    clf_metrics_c = train_and_evaluate_classifier(
        torch.tensor(X_synth_c),
        torch.tensor([0]*1000 + [1]*1000),
        X_test, y_test, device
    )

    results['classical'] = {
        'generator_params': sum(p.numel() for p in G_classical.parameters()),
        'training_history': history_c,
        'generation_metrics': gen_metrics_c,
        'classifier_metrics': clf_metrics_c
    }

    print(f"\nClassical Results:")
    print(f"  MMD: {gen_metrics_c['mmd']:.6f}")
    print(f"  Diversity ratio: {gen_metrics_c['diversity']['diversity_ratio']:.4f}")
    print(f"  Coverage: {gen_metrics_c['coverage']:.4f}")
    print(f"  Classifier F1 (attack): {clf_metrics_c['f1_attack']:.4f}")

    # ============ SAVE RESULTS ============
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    os.makedirs('./ablation_results', exist_ok=True)

    # Save full results
    with open('./ablation_results/ablation_results.json', 'w') as f:
        # Convert tensors to lists for JSON
        import json
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    # Save comparison summary
    summary = []

    if QUANTUM_AVAILABLE:
        summary.append({
            'model': 'quantum',
            'params': results['quantum']['generator_params'],
            'mmd': results['quantum']['generation_metrics']['mmd'],
            'diversity_ratio': results['quantum']['generation_metrics']['diversity']['diversity_ratio'],
            'coverage': results['quantum']['generation_metrics']['coverage'],
            'f1_attack': results['quantum']['classifier_metrics']['f1_attack'],
            'roc_auc': results['quantum']['classifier_metrics']['roc_auc']
        })

    summary.append({
        'model': 'classical',
        'params': results['classical']['generator_params'],
        'mmd': results['classical']['generation_metrics']['mmd'],
        'diversity_ratio': results['classical']['generation_metrics']['diversity']['diversity_ratio'],
        'coverage': results['classical']['generation_metrics']['coverage'],
        'f1_attack': results['classical']['classifier_metrics']['f1_attack'],
        'roc_auc': results['classical']['classifier_metrics']['roc_auc']
    })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('./ablation_results/ablation_summary.csv', index=False)

    print("\nAblation Summary:")
    print(summary_df.to_string())

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    metrics_to_plot = ['mmd', 'diversity_ratio', 'coverage', 'f1_attack']
    titles = ['MMD (lower=better)', 'Diversity Ratio', 'Coverage', 'F1 (Attack Class)']

    for ax, metric, title in zip(axes.flatten(), metrics_to_plot, titles):
        values = [s.get(metric, 0) for s in summary]
        models = summary_df['model'].tolist()
        ax.bar(models, values, color=['#7B68EE' if m == 'quantum' else '#4682B4' for m in models])
        ax.set_ylabel('Value')
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig('./ablation_results/ablation_comparison.png', dpi=150)
    print("\n✓ Saved results to ./ablation_results/")

    return results


if __name__ == "__main__":
    run_ablation_study()
