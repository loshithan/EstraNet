import os
import sys

# Add parent directory to path to allow importing from models and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow as tf

# Models
# Note: dynamic imports inside functions also need to be fixed or top-level imports used

print("Starting evaluation script...", flush=True)

# -----------------------------------------------------------------------------
# 1. Mamba-GNN Evaluation
# -----------------------------------------------------------------------------

def evaluate_mamba_gnn(dataset_path, checkpoint_path):
    print(f"\n[Mamba-GNN] Evaluating on {dataset_path}...", flush=True)
    
    try:
        from models.mamba_gnn_model import OptimizedMambaGNN
        print("[Mamba-GNN] Imported OptimizedMambaGNN", flush=True)
    except ImportError as e:
        print(f"[Mamba-GNN] Error importing model: {e}", flush=True)
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Mamba-GNN] Using device: {device}", flush=True)

    # Load Data
    try:
        with h5py.File(dataset_path, 'r') as f:
            X_profiling = f['Profiling_traces/traces'][:20000].astype(np.float32)
            X_attack = f['Attack_traces/traces'][:10000].astype(np.float32)
            metadata_attack = f['Attack_traces/metadata'][:10000]
            # real_key = metadata_attack['key'][0, 2] # Key should be fixed
    except Exception as e:
        print(f"[Mamba-GNN] Error loading data: {e}")
        return None

    # Normalization
    print("[Mamba-GNN] Fitting Scaler on profiling data...")
    scaler = StandardScaler()
    scaler.fit(X_profiling)
    del X_profiling
    
    print("[Mamba-GNN] Transforming attack traces...")
    X_attack_scaled = scaler.transform(X_attack)
    
    # Load Model
    # Remove d_state, d_conv, expand as they are not in __init__ of the copied mamba_gnn_model.py
    model = OptimizedMambaGNN(
        d_model=192,
        mamba_layers=4,
        gnn_layers=3, 
        # d_state=16,
        # d_conv=4,
        # expand=2,
        dropout=0.1,
        num_classes=256
    ).to(device)
    
    print(f"[Mamba-GNN] Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # weights_only=False for older PyTorch versions/safe loading
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    model.input_scale = 1.0 # Optimal setting found previously
    
    # Inference
    print("[Mamba-GNN] Running Inference...", flush=True)
    all_preds = []
    batch_size = 100
    with torch.no_grad():
        for i in tqdm(range(0, len(X_attack_scaled), batch_size)):
            batch = X_attack_scaled[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device)
            preds = model(batch_tensor)
            all_preds.append(preds.cpu().numpy())
            
    predictions = np.concatenate(all_preds, axis=0)
    
    # Rank
    return compute_rank(predictions, metadata_attack, "Mamba-GNN")

# -----------------------------------------------------------------------------
# 2. Transformer Evaluation
# -----------------------------------------------------------------------------

def evaluate_transformer(dataset_path, checkpoint_dir, checkpoint_idx):
    print(f"\n[Transformer] Evaluating on {dataset_path}...", flush=True)
    
    # We need to construct the model exactly as in train_trans.py
    from models.transformer import Transformer
    
    # Default Hyperparams from train_trans.py
    n_layer = 6
    d_model = 128
    d_head = 32
    n_head = 4
    d_inner = 256
    
    # UPDATED based on checkpoint mismatch error: (16, 8) for softmax/q_heads
    # Means d_head_softmax=16, n_head_softmax=8
    n_head_softmax = 8
    d_head_softmax = 16
    
    dropout = 0.1
    conv_kernel_size = 3
    n_conv_layer = 1
    pool_size = 2
    d_kernel_map = 128
    beta_hat_2 = 100
    model_normalization = 'preLC'
    head_initialization = 'forward'
    softmax_attn = True
    output_attn = False
    
    n_classes = 256
    
    # Clear session to avoid conflicts
    tf.keras.backend.clear_session()
    
    # Create Model
    model = Transformer(
        n_layer=n_layer, d_model=d_model, d_head=d_head, n_head=n_head, d_inner=d_inner,
        d_head_softmax=d_head_softmax, n_head_softmax=n_head_softmax, dropout=dropout,
        n_classes=n_classes, conv_kernel_size=conv_kernel_size, n_conv_layer=n_conv_layer,
        pool_size=pool_size, d_kernel_map=d_kernel_map, beta_hat_2=beta_hat_2,
        model_normalization=model_normalization, head_initialization=head_initialization,
        softmax_attn=softmax_attn, output_attn=output_attn
    )
    
    # Dummy call to build inputs
    dummy_input = tf.zeros((1, 700))
    model(dummy_input)
    
    # Restore Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"trans_long-{checkpoint_idx}")
    print(f"[Transformer] Restoring from {checkpoint_path}", flush=True)
    
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # LR doesn't matter for inference
    # ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    # Exclude optimizer to avoid shape mismatch (adam/learning_rate)
    ckpt = tf.train.Checkpoint(model=model)
    
    try:
        status = ckpt.restore(checkpoint_path).expect_partial()
        print("[Transformer] Checkpoint restored.", flush=True)
    except Exception as e:
        print(f"[Transformer] Error restoring checkpoint: {e}", flush=True)
        return None
        
    # Load Data (No scaling for Transformer typically? Or check train_trans.py... 
    # train_trans.py uses simple data loading, typically raw trace normalization is done in dataset or model handles it.
    # But usually ASCAD data is centered/standardized.
    # The Transformer model has `LayerScaling`/`LayerCentering` if `preLC` is used (which is default).
    # So we pass RAW traces (float32).
    
    try:
        with h5py.File(dataset_path, 'r') as f:
            X_attack = f['Attack_traces/traces'][:10000].astype(np.float32)
            metadata_attack = f['Attack_traces/metadata'][:10000]
    except Exception as e:
        print(f"[Transformer] Error loading data: {e}")
        return None
        
    # Inference
    print("[Transformer] Running Inference...", flush=True)
    predictions = model.predict(X_attack, batch_size=256)
    
    # Transformer returns [logits] or [logits, attn]
    if isinstance(predictions, list):
        predictions = predictions[0]
        
    return compute_rank(predictions, metadata_attack, "Transformer")

# -----------------------------------------------------------------------------
# Shared Utils
# -----------------------------------------------------------------------------

def compute_rank(predictions, metadata, model_name):
    print(f"[{model_name}] Computing Rank...")
    
    # Sbox
    sbox = np.array([
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    ])
    
    # Initialize key probabilities (log scale)
    key_probabilities = np.zeros(256)
    
    real_key = metadata['key'][0, 2]
    plaintext = metadata['plaintext'][:, 2]
    
    num_traces = len(predictions)
    rank_evolution = np.zeros(num_traces)

    for i in tqdm(range(num_traces), desc=f"[{model_name}] Rank"):
        p_t = plaintext[i]
        preds = predictions[i].flatten() # Ensure 1D array
        
        # Apply softmax to get probabilities
        # exp_preds = np.exp(preds - np.max(preds))
        # probs = exp_preds / np.sum(exp_preds)
        
        # Or faster: LogSoftmax directly?
        # log_probs = preds - scipy.special.logsumexp(preds)
        
        # Simple softmax
        probs = np.exp(preds - np.max(preds))
        probs = probs / np.sum(probs)
        
        # Add log(prob) to key candidate accumulation
        for k in range(256):
            # If key guess is k, then input byte was p_t ^ k.
            # The label (Sbox output) would be sbox[p_t ^ k]
            label = sbox[p_t ^ k]
            prob = probs[label]
            key_probabilities[k] += np.log(max(prob, 1e-40))
            
        # Compute rank of real key
        sorted_keys = np.argsort(key_probabilities)[::-1] # Descending prob
        real_key_rank = np.where(sorted_keys == real_key)[0][0]
        rank_evolution[i] = real_key_rank
        
    print(f"[{model_name}] Final Rank: {rank_evolution[-1]}", flush=True)
    return rank_evolution

def run_comparison():
    print("Entered run_comparison", flush=True)
    # Paths
    desync50_data = "data/ASCAD_desync50.h5"
    mamba_ckpt = "checkpoints/best_mamba_gnn_tunned.pth" # Correct path found
    trans_ckpt_dir = "checkpoints/checkpoints_transformer_desync50"
    trans_ckpt_idx = 5
    
    print(f"Checking data: {desync50_data}", flush=True)
    # Check files
    if not os.path.exists(desync50_data):
        print(f"Data not found: {desync50_data}", flush=True)
        return

    print("Data found. Starting Mamba eval...", flush=True)
    # Evaluate Mamba
    try:
        rank_mamba = evaluate_mamba_gnn(desync50_data, mamba_ckpt)
    except Exception as e:
        print(f"Mamba eval failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        rank_mamba = None
    
    print("Starting Transformer eval...", flush=True)
    # Evaluate Transformer
    try:
        rank_trans = evaluate_transformer(desync50_data, trans_ckpt_dir, trans_ckpt_idx)
    except Exception as e:
        print(f"Transformer eval failed: {e}")
        rank_trans = None
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    if rank_mamba is not None:
        np.save("rank_mamba_desync50.npy", rank_mamba)
        plt.plot(rank_mamba, label=f"Mamba-GNN (Final: {rank_mamba[-1]})", color='green', linewidth=2)
        
    if rank_trans is not None:
        np.save("rank_trans_desync50.npy", rank_trans)
        plt.plot(rank_trans, label=f"Transformer Desync50 (Final: {rank_trans[-1]})", color='orange', linewidth=2)

    plt.yscale('log')
    plt.title("ASCAD Desync50: Mamba-GNN vs Transformer")
    plt.xlabel("Traces")
    plt.ylabel("Key Rank (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("desync50_comparison.png")
    print("Comparison plot saved to desync50_comparison.png", flush=True)

print("Script initialization complete. Calling main...", flush=True)
if __name__ == "__main__":
    run_comparison()
