# ============================================================================
# MASS EVALUATE CHECKPOINTS (GNN)
# ============================================================================
# Evaluates ALL checkpoints in given directories.
# Usage: python evaluate_all_checkpoints.py --checkpoint_dirs "dir1" "dir2" ...

import argparse
import tensorflow as tf
import numpy as np
import h5py
import os
import glob
import sys
import pandas as pd

# Add subdirectories to path for robust imports
sys.path.append('gnn-scripts')
sys.path.append('mamba-scripts')

# Import Models (Robust)
from evaluation_utils import compute_key_rank
try:
    from gnn_estranet import GNNEstraNet
except ImportError:
    from gnn_scripts.gnn_estranet import GNNEstraNet

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

def build_model(input_length, pool_size=2):
    print(f"🏗️ Building GNN model (Pool={pool_size})...")
    model = GNNEstraNet(
        n_gcn_layers=2, d_model=128, k_neighbors=5, graph_pooling='mean',
        d_head_softmax=16, n_head_softmax=8, dropout=0.05, n_classes=256,
        conv_kernel_size=3, n_conv_layer=2, pool_size=pool_size,
        beta_hat_2=150, model_normalization='preLC', softmax_attn=True, output_attn=False
    )
    # Dummy build
    _ = model(tf.zeros((1, input_length)), training=False)
    return model

def evaluate_all(args):
    # 1. Load Data (Range)
    traces, plaintexts, keys = load_ascad(n_traces=args.max_traces, start_index=args.start_index, input_length=args.input_length)
    
    # 2. Build Model (Once)
    model = build_model(args.input_length, args.pool_size)
    ckpt = tf.train.Checkpoint(model=model)

    # 3. Find Checkpoints
    checkpoint_files = []
    for d in args.checkpoint_dirs:
        # Support full paths or glob patterns
        if "*" in d:
             files = glob.glob(d) # Handle wildcards if user passed them
             # Filter for .index
             files = [f for f in files if f.endswith(".index")]
        else:
             # Search directory
             search_path = os.path.join(d, "*.index")
             files = glob.glob(search_path)
             
        # Normalize paths (remove .index suffix for restore)
        for f in files:
            prefix = f.replace(".index", "")
            if prefix not in checkpoint_files:
                checkpoint_files.append(prefix)
    
    if not checkpoint_files:
        print("❌ No checkpoints found!")
        return

    print(f"🔍 Found {len(checkpoint_files)} checkpoints to evaluate.")
    
    results = []
    
    # 4. Loop Evaluate
    for i, prefix in enumerate(checkpoint_files):
        name = os.path.basename(prefix)
        print(f"\n[{i+1}/{len(checkpoint_files)}] Testing {name}...")
        
        # Get Size (Check .data file)
        data_path = prefix + ".data-00000-of-00001"
        size_mb = 0.0
        if os.path.exists(data_path):
            size_mb = os.path.getsize(data_path) / (1024 * 1024)
        
        try:
            ckpt.restore(prefix).expect_partial()
        except Exception as e:
            print(f"   ❌ Load Failed: {e}")
            results.append({'Name': name, 'SizeMB': f"{size_mb:.2f}", 'Rank': 999, 'BrokenAt': '-', 'Status': 'Load Error'})
            continue
            
        # Inference
        preds = model.predict(traces, batch_size=args.batch_size, verbose=0)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
            
        # Compute Rank
        ranks = compute_key_rank(preds, plaintexts, keys)
        final_rank = ranks[-1]
        
        # Check broken
        broken_step = ">" + str(args.max_traces)
        success_idx = np.where(ranks <= 0)[0]
        if len(success_idx) > 0:
            broken_step = success_idx[0] + 1
            print(f"   🏆 BROKEN at {broken_step} traces! (Rank {final_rank:.2f})")
        else:
            print(f"   Rank: {final_rank:.2f}")
            
        results.append({
            'Name': name, 
            'SizeMB': f"{size_mb:.2f}",
            'Rank': final_rank, 
            'BrokenAt': broken_step, 
            'Status': 'OK' if final_rank < 10 else 'Weak'
        })

    # 5. Summary
    print("\n" + "="*60)
    print("FINAL RESULTS (Sorted by Rank)")
    print("="*60)
    
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='Rank')
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_filename = f"evaluation_results_{args.start_index}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n✅ Results saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dirs", nargs='+', required=True, help="List of directories to search")
    parser.add_argument("--input_length", type=int, default=700)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--max_traces", type=int, default=2000)
    parser.add_argument("--start_index", type=int, default=0, help="Trace index to start loading from")
    parser.add_argument("--batch_size", type=int, default=256)
    
    args = parser.parse_args()
    evaluate_all(args)
