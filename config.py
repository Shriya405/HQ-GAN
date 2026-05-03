"""
Configuration file for HQ-GAN project.
Edit these settings to customize your experiment.
"""

# ============ Paths ============
# For Colab: Set to your Google Drive path
# For Local: Set to your local data directory
DATA_PATH = "/content/drive/MyDrive/supergan/CSE-CIC-IDS2018"
CHECKPOINT_DIR = "/content/drive/MyDrive/supergan"
LOCAL_CHECKPOINT_DIR = "./checkpoints"  # Fallback for local execution

# ============ Data Settings ============
N_SAMPLES = 70000  # Number of samples to load from dataset
TEST_SIZE = 0.2  # Train/test split ratio
RANDOM_SEED = 42

# ============ Model Architecture ============
LATENT_DIM = 64
FEATURE_DIM = 77  # Will be overwritten by actual data dimension
NUM_CLASSES = 2
COND_DIM = 16
QUANTUM_DIM = 16  # Number of qubits (upgraded from 4 for more expressive quantum bottleneck)
NUM_HEADS = 7  # For self-attention (FEATURE_DIM must be divisible by this)

# ============ Training Hyperparameters ============
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
BETAS = (0.5, 0.9)
EPOCHS_GAN = 12
EPOCHS_CLF = 5
N_CRITIC = 5  # For WGAN-style training
LAMBDA_GP = 10  # Gradient penalty coefficient

# ============ Synthetic Data Generation ============
N_SYNTHETIC_TOTAL = 40000  # Total synthetic samples to generate

# ============ Device ============
DEVICE = "cuda"  # Will fallback to CPU if CUDA unavailable
