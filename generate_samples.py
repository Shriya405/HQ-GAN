"""
Generate and Export Synthetic Network Traffic Samples
======================================================
This script loads a trained HQ-GAN model and generates synthetic
network intrusion detection samples (both attacks and benign traffic).

Usage:
    python generate_samples.py --checkpoint path/to/checkpoint.pth --n_samples 5000

For Colab:
    - Place checkpoint in Google Drive
    - Update CHECKPOINT_PATH below
    - Run: python generate_samples.py
"""

import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import pyarrow.parquet as pq
import pickle

# ============ Configuration ============
CHECKPOINT_PATH = "/content/drive/MyDrive/supergan/FULL_CHECKPOINT.pth"  # Update for your setup
DATA_PATH = "/content/drive/MyDrive/supergan/CSE-CIC-IDS2018"
OUTPUT_DIR = "./synthetic_samples"

# Architecture hyperparameters (must match training)
LATENT_DIM = 64
COND_DIM = 16
QUANTUM_DIM = 16  # Upgraded from 4 to 16 qubits
NUM_HEADS = 7
NUM_CLASSES = 2
N_QUANTUM_LAYERS = 2

# Generation settings
N_SYNTH_ATTACK = 5000
N_SYNTH_BENIGN = 5000
# =======================================


# ============ Model Definitions (must match training) ============
import pennylane as qml

quantum_dim = QUANTUM_DIM
dev = qml.device("lightning.qubit", wires=quantum_dim)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_layer_batched(x_batch, weights):
    """Process a batch of samples through the variational quantum circuit.

    FIXED: Returns torch.Tensor instead of list of lists.
    """
    batch_size = x_batch.shape[0]
    n_layers = weights.shape[0]
    results = []

    for i in range(batch_size):
        # Encode input
        for q in range(quantum_dim):
            qml.RY(x_batch[i, q], wires=q)

        # Variational layers
        for layer in range(n_layers):
            for q in range(quantum_dim):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            # Circular entanglement
            for q in range(quantum_dim - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[quantum_dim - 1, 0])

        # Measure - FIXED: convert to tensor immediately
        result = torch.tensor([qml.expval(qml.PauliZ(i)) for i in range(quantum_dim)], dtype=torch.float32)
        results.append(result)

    return torch.stack(results)


class QuantumEncoder(nn.Module):
    def __init__(self, input_dim, quantum_dim, output_dim, n_quantum_layers=2):
        super().__init__()
        self.quantum_dim = quantum_dim
        self.n_quantum_layers = n_quantum_layers
        self.weights = nn.Parameter(torch.randn(n_quantum_layers, quantum_dim, 2) * 0.1)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
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
        z_q = self.encoder(x)
        q_out = quantum_layer_batched(z_q, self.weights)
        return self.decoder(q_out)


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


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)

    def forward(self, labels):
        return self.embed(labels)


class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim, cond_dim, quantum_dim, n_quantum_layers=2):
        super().__init__()
        self.cond_emb = nn.Linear(cond_dim, latent_dim)
        self.quantum_encoder = QuantumEncoder(latent_dim, quantum_dim, feature_dim, n_quantum_layers)
        self.attn1 = MultiHeadSelfAttention(feature_dim, num_heads=7)
        self.attn2 = MultiHeadSelfAttention(feature_dim, num_heads=7)

    def forward(self, z, cond):
        z = z + self.cond_emb(cond)
        out = self.quantum_encoder(z)
        out = self.attn1(out)
        out = self.attn2(out)
        return out
# ================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic network traffic samples")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to trained checkpoint")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to original dataset (for feature names)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--n_attack", type=int, default=N_SYNTH_ATTACK, help="Number of attack samples")
    parser.add_argument("--n_benign", type=int, default=N_SYNTH_BENIGN, help="Number of benign samples")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    input_dim = checkpoint.get('input_dim', 77)

    # Reconstruct models
    cond_embed = ConditionalEmbedding(NUM_CLASSES, COND_DIM).to(device)
    G = Generator(LATENT_DIM, input_dim, COND_DIM, QUANTUM_DIM, n_quantum_layers=N_QUANTUM_LAYERS).to(device)

    G.load_state_dict(checkpoint['G_state_dict'])
    cond_embed.load_state_dict(checkpoint['cond_embed_state_dict'])

    G.eval()
    print("Models loaded successfully")

    # Load feature names from original dataset
    print(f"Loading feature names from {args.data_path}...")
    files = [f for f in os.listdir(args.data_path) if f.endswith('.parquet')]
    df_sample = pq.read_table(os.path.join(args.data_path, files[0])).to_pandas()
    feature_names = df_sample.drop(columns=['Label']).columns.tolist()
    print(f"Found {len(feature_names)} features")

    # Generate synthetic samples
    print(f"\nGenerating {args.n_attack} attack and {args.n_benign} benign samples...")

    with torch.no_grad():
        # Attack samples
        z_attack = torch.randn(args.n_attack, LATENT_DIM, device=device)
        c_attack = cond_embed(torch.ones(args.n_attack, dtype=torch.long, device=device))
        X_synth_attack = G(z_attack, c_attack).cpu().numpy()

        # Benign samples
        z_benign = torch.randn(args.n_benign, LATENT_DIM, device=device)
        c_benign = cond_embed(torch.zeros(args.n_benign, dtype=torch.long, device=device))
        X_synth_benign = G(z_benign, c_benign).cpu().numpy()

    # Create DataFrames
    df_synth_attack = pd.DataFrame(X_synth_attack, columns=feature_names)
    df_synth_attack['Label'] = 'Attack'

    df_synth_benign = pd.DataFrame(X_synth_benign, columns=feature_names)
    df_synth_benign['Label'] = 'Benign'

    # Export
    os.makedirs(args.output_dir, exist_ok=True)

    attack_csv = os.path.join(args.output_dir, 'synthetic_attacks.csv')
    benign_csv = os.path.join(args.output_dir, 'synthetic_benign.csv')
    combined_csv = os.path.join(args.output_dir, 'synthetic_samples_combined.csv')

    df_synth_attack.to_csv(attack_csv, index=False)
    df_synth_benign.to_csv(benign_csv, index=False)
    pd.concat([df_synth_attack, df_synth_benign]).to_csv(combined_csv, index=False)

    # Parquet format
    df_synth_attack.to_parquet(attack_csv.replace('.csv', '.parquet'), index=False)
    df_synth_benign.to_parquet(benign_csv.replace('.csv', '.parquet'), index=False)

    print("\n" + "="*60)
    print("✓ EXPORT COMPLETE")
    print("="*60)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"\nCSV Files:")
    print(f"  - Attacks:  {attack_csv} ({len(df_synth_attack)} samples)")
    print(f"  - Benign:   {benign_csv} ({len(df_synth_benign)} samples)")
    print(f"  - Combined: {combined_csv}")
    print(f"\nParquet Files:")
    print(f"  - {attack_csv.replace('.csv', '.parquet')}")
    print(f"  - {benign_csv.replace('.csv', '.parquet')}")

    # Statistics
    print(f"\n=== Data Statistics ===")
    print(f"Attack value range: [{df_synth_attack.iloc[:, :-1].values.min():.4f}, {df_synth_attack.iloc[:, :-1].values.max():.4f}]")
    print(f"Benign value range: [{df_synth_benign.iloc[:, :-1].values.min():.4f}, {df_synth_benign.iloc[:, :-1].values.max():.4f}]")

    print("\n✓ Files ready for download/use!")


if __name__ == "__main__":
    main()
