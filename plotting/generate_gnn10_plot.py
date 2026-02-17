import os
import sys
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

# Add subdirectories to path
sys.path.append('gnn-scripts')
from gnn_estranet import GNNEstraNet
from evaluation_utils import compute_key_rank

# --- Configuration ---
CHECKPOINT = "checkpoints/gnn_checkpoints_archive2/gnn_ASCAD-10"
DATA_PATH = "data/ASCAD.h5" # Desync 0
INPUT_LENGTH = 700
MAX_TRACES = 10000
BATCH_SIZE = 256

def generate_plot():
    print(f"📥 Loading dataset: {DATA_PATH}...")
    with h5py.File(DATA_PATH, 'r') as f:
        traces = f['Attack_traces']['traces'][:MAX_TRACES, :INPUT_LENGTH].astype(np.float32)
        metadata = f['Attack_traces']['metadata'][:MAX_TRACES]
        plaintexts = metadata['plaintext'][:, 2].astype(np.uint8)
        keys = metadata['key'][:, 2].astype(np.uint8)

    print("🏗️ Building GNN Model...")
    model = GNNEstraNet(
        n_gcn_layers=2, d_model=128, k_neighbors=5, graph_pooling='mean',
        d_head_softmax=16, n_head_softmax=8, dropout=0.05, n_classes=256,
        conv_kernel_size=3, n_conv_layer=2, pool_size=2,
        beta_hat_2=150, model_normalization='preLC', softmax_attn=True, output_attn=False
    )
    
    # Initialize weights
    _ = model(tf.zeros((1, INPUT_LENGTH)), training=False)
    
    print(f"📂 Restoring Checkpoint: {CHECKPOINT}...")
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(CHECKPOINT).expect_partial()

    print(f"🚀 Running Inference on {MAX_TRACES} traces...")
    preds = model.predict(traces, batch_size=BATCH_SIZE, verbose=1)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    print("📊 Computing Rank Progression...")
    # compute_key_rank in evaluation_utils.py returns ranks for 1..N traces
    ranks = compute_key_rank(preds, plaintexts, keys)
    
    # Save NPY
    np.save("rank_gnn10.npy", ranks)
    print("✅ Saved rank_gnn10.npy")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, label=f'GNN-ASCAD-10 (Final Rank: {ranks[-1]:.2f})', color='purple', linewidth=2)
    
    plt.title('GNN-10 Attack on ASCAD desync0 (10k Traces)')
    plt.xlabel('Number of Traces')
    plt.ylabel('Key Rank (Log Scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    output_file = "plot_gnn10.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ Saved {output_file}")

if __name__ == "__main__":
    generate_plot()
