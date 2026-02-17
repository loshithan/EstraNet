# ============================================================================
# EVALUATE RANK VS TRACES (COMPARATIVE)
# ============================================================================
# This script tests multiple checkpoints and plots their rank progression on one chart.

import argparse
import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import sys

# Robust Imports
sys.path.append('gnn-scripts')
sys.path.append('mamba-scripts')

# Import Models
from evaluation_utils import compute_key_rank
try:
    from gnn_estranet import GNNEstraNet
except ImportError:
    from gnn_scripts.gnn_estranet import GNNEstraNet

try:
    from estranet import Transformer
except ImportError:
    try:
        from estranet import Transformer
    except:
        Transformer = None

def load_ascad(n_traces=2000, start_index=0, input_length=700):
    print(f"📥 Loading ASCAD dataset (Traces {start_index}-{start_index+n_traces}, Length {input_length})...")
    with h5py.File('data/ASCAD.h5', 'r') as f:
        # Load specific range
        total_available = f['Attack_traces']['traces'].shape[0]
        end_index = min(start_index + n_traces, total_available)
        
        traces = f['Attack_traces']['traces'][start_index:end_index]
        metadata = f['Attack_traces']['metadata'][start_index:end_index]
        
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
        if Transformer is None:
             raise ImportError("Transformer module not found.")
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
    traces, plaintexts, keys = load_ascad(n_traces=args.max_traces, start_index=args.start_index, input_length=args.input_length)

    # 2. Build Model
    model = build_model(args.model_type, args.input_length, args.pool_size)
    ckpt = tf.train.Checkpoint(model=model)
    
    # Setup Plot
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # 3. Loop Checkpoints
    for i, checkpoint_path in enumerate(args.checkpoints):
        # Normalize path
        if checkpoint_path.endswith('.index'):
            checkpoint_path = checkpoint_path[:-6]
            
        label = os.path.basename(checkpoint_path)
        print(f"\n[{i+1}/{len(args.checkpoints)}] Testing {label}...")
        
        try:
            ckpt.restore(checkpoint_path).expect_partial()
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            continue

        # 4. Inference
        predictions = model.predict(traces, batch_size=args.batch_size, verbose=0)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        # 5. Compute Rank per Step
        rank_history = []
        x_axis = []
        
        # Calculate rank every step_size traces
        for n in range(args.step_size, len(traces) + 1, args.step_size):
            preds_subset = predictions[:n]
            txt_subset = plaintexts[:n]
            key_subset = keys[:n]
            
            ranks = compute_key_rank(preds_subset, txt_subset, key_subset)
            final_rank = ranks[-1]
            
            rank_history.append(final_rank)
            x_axis.append(n)
        
        # Plot Line
        color = colors[i % len(colors)]
        plt.plot(x_axis, rank_history, marker='o', label=label, color=color)
        
        # Check Final Status
        end_rank = rank_history[-1]
        print(f"   End Rank: {end_rank:.2f}")

    # 6. Finalize Plot
    plt.axhline(0, color='r', linestyle='--', alpha=0.3, label="Rank 0 (Broken)")
    plt.title(f"Key Rank Progression - {args.model_type.upper()}")
    plt.xlabel("Number of Traces")
    plt.ylabel("Key Rank (Lower is Better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    filename = f"rank_progression_{args.start_index}.png"
    plt.savefig(filename)
    print(f"\n✅ Plot saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs='+', required=True, help="List of checkpoint paths")
    parser.add_argument("--model_type", default="gnn", choices=["gnn", "transformer"], help="Model architecture")
    parser.add_argument("--input_length", type=int, default=700, help="Trace length")
    parser.add_argument("--pool_size", type=int, default=2, help="Pooling size")
    parser.add_argument("--step_size", type=int, default=100, help="Calculate rank every N traces")
    parser.add_argument("--max_traces", type=int, default=2000, help="Max traces to evaluate")
    parser.add_argument("--start_index", type=int, default=0, help="Trace start index")
    parser.add_argument("--batch_size", type=int, default=256)
    
    args = parser.parse_args()
    evaluate(args)
