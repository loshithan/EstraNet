# ============================================================================
# EVALUATE RANK VS TRACES (GENERALIZED)
# ============================================================================
# This script creates a standalone tool `evaluate_rank_progression.py`
# that can test ANY model (GNN, Transformer, Mamba) and any Checkpoint.
# It computes the rank every N traces (e.g. 100).

import argparse
import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from evaluation_utils import compute_key_rank

# Import Models
from gnn_estranet import GNNEstraNet
try:
    from estranet import Transformer  # Adjust import based on your file structure
except ImportError:
    try:
        from estranet import Transformer
    except:
        print("⚠️ Warning: Could not import Transformer. GNN still works.")

def load_ascad(n_traces=10000, input_length=700):
    print(f"📥 Loading ASCAD dataset ({n_traces} traces, Length {input_length})...")
    with h5py.File('data/ASCAD.h5', 'r') as f:
        traces = f['Attack_traces']['traces'][:n_traces]
        metadata = f['Attack_traces']['metadata'][:n_traces]
        
        # Slicing and Casting to float32 (Crucial for Conv1D)
        traces = traces[:, :input_length].astype(np.float32)
        
        plaintexts = metadata['plaintext'][:, 2].astype(np.uint8)
        keys = metadata['key'][:, 2].astype(np.uint8)
    return traces, plaintexts, keys

def build_model(model_type, input_length, pool_size=2):
    print(f"🏗️ Building {model_type.upper()} model...")
    
    if model_type == 'gnn':
        model = GNNEstraNet(
            n_gcn_layers=2, d_model=128, k_neighbors=5, graph_pooling='mean',
            d_head_softmax=16, n_head_softmax=8, dropout=0.05, n_classes=256,
            conv_kernel_size=3, n_conv_layer=2, pool_size=pool_size,
            beta_hat_2=150, model_normalization='preLC', softmax_attn=True, output_attn=False
        )
    elif model_type == 'transformer':
        # Standard EstraNet Transformer config
        model = Transformer(
            n_layer=4, d_model=128, d_head=16, n_head=8, d_inner=512,
            n_head_softmax=8, d_head_softmax=16, dropout=0.1, n_classes=256,
            conv_kernel_size=3, n_conv_layer=2, pool_size=pool_size,
            d_kernel_map=16, beta_hat_2=150, model_normalization='preLC',
            head_initialization='random', softmax_attn=True, output_attn=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Dummy build
    _ = model(tf.zeros((1, input_length)), training=False)
    return model

def evaluate(args):
    # 1. Load Data
    traces, plaintexts, keys = load_ascad(n_traces=args.max_traces, input_length=args.input_length)

    # 2. Build Model
    model = build_model(args.model_type, args.input_length, args.pool_size)

    # 3. Restore Checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path.endswith('.index'):
        checkpoint_path = checkpoint_path[:-6]  # Strip .index
    
    print(f"📂 Loading Checkpoint: {checkpoint_path}")
    try:
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(checkpoint_path).expect_partial()
        print("✅ Checkpoint Loaded!")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    # 4. Run Inference (Batched)
    print("🧠 Running Inference...")
    predictions = model.predict(traces, batch_size=args.batch_size, verbose=1)

    # Unpack tuple if necessary (GNN returns (logits,))
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]

    # 5. Compute Rank per Step
    print(f"📊 Calculating Rank every {args.step_size} traces...")
    
    rank_history = []
    x_axis = []
    
    # We will simulate "progression" by taking subsets of predictions.
    # Note: Using random subsets or sequential depends on methodology.
    # Usually standard is sequential.
    
    # We compute ONE rank evolution curve for the whole set, then sample points.
    # Actually, compute_key_rank usually returns the convergence over traces.
    # Let's use that if possible, or manually step.
    
    # Manual stepping for clear "N traces" metrics
    for n in range(args.step_size, len(traces) + 1, args.step_size):
        # Subset
        preds_subset = predictions[:n]
        txt_subset = plaintexts[:n]
        key_subset = keys[:n]
        
        # Compute rank for this subset (outcome is final rank of this subset)
        # We handle single key assumption (checking key[0])
        real_key = key_subset[0] 
        
        # Using the provided utility logic (simplified for script)
        # We need the rank of the correct key byte.
        # Log-likelihood accumulation
        log_preds = np.log(preds_subset + 1e-30)
        
        # Get probabilities for only the correct key byte candidates (0..255)
        # We need a key guess model (like Sbox(pt ^ k)). 
        # Since we don't have the attack implementation here, we rely on `compute_key_rank`
        # if available.
        # However, `compute_key_rank` typically returns an array of shape (n_traces,).
        
        ranks = compute_key_rank(preds_subset, txt_subset, key_subset)
        final_rank = ranks[-1]
        
        rank_history.append(final_rank)
        x_axis.append(n)
        
        print(f"   Traces: {n:<5} | Rank: {final_rank:.2f}")
        
        if final_rank <= 0.0:
            print(f"🎉 BROKEN at {n} traces!")
            if not args.full_sweep:
                 break
    
    # 6. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, rank_history, marker='o', label=args.label or args.model_type)
    plt.axhline(0, color='r', linestyle='--', alpha=0.3)
    plt.title(f"Key Rank Progression - {args.model_type.upper()}")
    plt.xlabel("Number of Traces")
    plt.ylabel("Key Rank")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("rank_progression.png")
    print("\n✅ Plot saved to rank_progression.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (prefix or .index)")
    parser.add_argument("--model_type", default="gnn", choices=["gnn", "transformer"], help="Model architecture")
    parser.add_argument("--input_length", type=int, default=700, help="Trace length")
    parser.add_argument("--pool_size", type=int, default=2, help="Pooling size (GNN=2 for 175 nodes)")
    parser.add_argument("--step_size", type=int, default=100, help="Calculate rank every N traces")
    parser.add_argument("--max_traces", type=int, default=2000, help="Max traces to evaluate")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--label", type=str, default="", help="Label for plot")
    parser.add_argument("--full_sweep", action="store_true", help="Continue even after breaking")
    
    args = parser.parse_args()
    evaluate(args)
