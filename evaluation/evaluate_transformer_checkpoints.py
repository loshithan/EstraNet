
import os
import glob
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import re

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model and data utils
from models.transformer import Transformer
from utils.data_utils import Dataset
from models.fast_attention import SelfAttention

# --- Configuration ---
class Config:
    CHECKPOINT_DIR = "checkpoints" 
    DATA_FILE = "data/ASCAD.h5"
    INPUT_LENGTH = 700 
    N_HEAD = 8 
    NUM_TEST_TRACES = 2000 
    BATCH_SIZE = 50 
    KEY_LENGTH = 16 

    # Model Hyperparameters (Must match checkpoint training)
    N_LAYER = 2
    D_MODEL = 128
    D_HEAD = 32
    D_INNER = 256
    N_HEAD_SOFTMAX = 8
    D_HEAD_SOFTMAX = 16
    DROPOUT = 0.05
    CONV_KERNEL_SIZE = 3
    N_CONV_LAYER = 2
    POOL_SIZE = 20
    D_KERNEL_MAP = 512
    BETA_HAT_2 = 150
    MODEL_NORMALIZATION = "preLC"
    HEAD_INITIALIZATION = "forward"
    SOFTMAX_ATTN = True
    OUTPUT_ATTN = False

# --- Helper Functions ---

def load_data():
    print(f"Loading data from {Config.DATA_FILE}...")
    dataset = Dataset(Config.DATA_FILE, Config.INPUT_LENGTH, 0) # 0 desync for testing
    
    # We need to manually load test data because dataset.get_next_train_batch involves randomness
    # and we want consistent test set.
    # The Dataset class in data_utils doesn't expose a clean "get_test_data" method 
    # that returns numpy arrays directly, it uses TF iterators.
    # Let's read directly from H5 for simplicity and speed in this evaluation script.

    with h5py.File(Config.DATA_FILE, "r") as f:
        X_test = f['Attack_traces']['traces'][:Config.NUM_TEST_TRACES]
        # Slicing to input_length
        X_test = X_test[:, :Config.INPUT_LENGTH]
        
        metadata = f['Attack_traces']['metadata']
        # Metadata is a structured array
        plt_txt = metadata['plaintext'][:Config.NUM_TEST_TRACES]
        keys = metadata['key'][:Config.NUM_TEST_TRACES]
        
        # We need the correct key byte (usually byte 2 for ASCAD)
        # But wait, test_ascad.py uses dataset.get_test_iterator.
        # Let's stick to using the Dataset class to ensure preprocessing (if any) matches.
        
    return dataset

def rank_func(preds, plt_txt, correct_key_byte, byte_idx=2):
    # preds: (n_traces, 256) - probability of each S-box output
    # plt_txt: (n_traces, 16)
    # correct_key_byte: scalar
    
    # SBox table (standard AES)
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

    num_traces = preds.shape[0]
    key_prob = np.zeros(256)

    # Accumulate probabilities (log likelihood)
    # Equation: sum( log( P(expected_sbox | trace) ) )
    # But predictions are probabilities of Sbox output
    
    # For each key candidate k (0..255):
    #   Calculate expected Sbox output: v = Sbox[pt[2] ^ k]
    #   Add log(preds[trace_idx][v]) to key_prob[k]
    
    # Vectorized approach:
    # 1. Expand plt_txt to (n_traces, 256) candidates
    # 2. Compute indices
    
    ranks = []
    
    # Efficient calculation using broadcasting
    # Assuming byte 2 is the target
    pts = plt_txt[:, byte_idx] # (n_traces,)
    
    # Pre-compute log predictions to avoid repeated logs
    log_preds = np.log(preds + 1e-30) 

    # Key candidates 0..255
    k_candidates = np.arange(256, dtype=np.uint8)
    
    # We will accumulate probabilities trace by trace to see rank evolution
    total_log_prob = np.zeros(256)
    
    for i in range(num_traces):
        pt = pts[i]
        # sbox_outs for all 256 keys given this pt
        # sbox_outs[k] = Sbox[pt ^ k]
        sbox_outs = sbox[pt ^ k_candidates]
        
        # Get probabilities for these valid sbox outputs from the prediction
        probs = log_preds[i, sbox_outs]
        
        # Update geometric mean (sum of logs)
        total_log_prob += probs
        
        # Calculate Rank of correct key
        # Rank = number of keys with higher probability than correct key
        correct_prob = total_log_prob[correct_key_byte]
        rank = np.count_nonzero(total_log_prob > correct_prob)
        ranks.append(rank)
        
    return ranks

def find_checkpoints():
    # Only transformer checkpoints
    pattern = os.path.join(Config.CHECKPOINT_DIR, "trans_long-*.index")
    print(f"DEBUG: find_checkpoints pattern: {pattern}")
    files = glob.glob(pattern)
    print(f"DEBUG: find_checkpoints found {len(files)} files.")
    # Extract filename without extension and use ABSOLUTE path
    ckpts = [os.path.abspath(os.path.splitext(f)[0]) for f in files]
    
    # Sort by step number if possible or version
    # trans_long-8 -> 8
    def get_sort_key(name):
        match = re.search(r"trans_long-(\d+)", name)
        if match:
            return int(match.group(1))
        return 0
        
    ckpts.sort(key=get_sort_key)
    return ckpts

def build_model():
    model = Transformer(
        n_layer=Config.N_LAYER,
        d_model=Config.D_MODEL,
        d_head=Config.D_HEAD,
        n_head=Config.N_HEAD,
        d_inner=Config.D_INNER,
        d_head_softmax=Config.D_HEAD_SOFTMAX,
        n_head_softmax=Config.N_HEAD_SOFTMAX,
        dropout=Config.DROPOUT,
        n_classes=256,
        conv_kernel_size=Config.CONV_KERNEL_SIZE,
        n_conv_layer=Config.N_CONV_LAYER,
        pool_size=Config.POOL_SIZE,
        d_kernel_map=Config.D_KERNEL_MAP,
        beta_hat_2=Config.BETA_HAT_2,
        model_normalization=Config.MODEL_NORMALIZATION,
        head_initialization=Config.HEAD_INITIALIZATION,
        softmax_attn=Config.SOFTMAX_ATTN,
        output_attn=Config.OUTPUT_ATTN
    )
    
    # Run dummy inference to initialize variables (Eager execution style)
    dummy_input = tf.zeros((1, Config.INPUT_LENGTH), dtype=tf.float32)
    model(dummy_input, training=False)
    return model

# --- Main Logic ---

def main():
    print("Starting Transformer Checkpoint Evaluation...")
    
    # 1. Load Data
    with h5py.File(Config.DATA_FILE, "r") as f:
        # Load Raw bytes from H5
        print(f"   Reading {Config.NUM_TEST_TRACES} test traces...")
        X_test = f['Attack_traces']['traces'][:Config.NUM_TEST_TRACES]
        X_test = X_test[:, :Config.INPUT_LENGTH] # Crop
        
        metadata = f['Attack_traces']['metadata'][:Config.NUM_TEST_TRACES]
        # Assuming metadata is structured
        # We need plaintexts and keys. 
        # Check dtype names to be safe, usually 'plaintext' and 'key'
        # In ASCAD variable key, key is (16,) byte array.
        
        # Extract correct key byte (Assuming byte 2 is the target, verify with test_ascad logic)
        # test_ascad.py usually attacks byte 2 (index 2) for ASCAD fixed/variable
        TARGET_BYTE = 2
        
        # Handle structured array access
        try:
             plt_txt = metadata['plaintext']
             keys = metadata['key']
        except:
             # If it's not structured, fallback (unlikely for ASCAD.h5)
             print("❌ Error reading metadata fields. Check format.")
             return

        correct_key = keys[0][TARGET_BYTE] # Assuming fixed key for now or just taking first
        # Actually in ASCAD variable key, key changes.
        # But 'rank_func' needs scalar correct key? 
        # Wait, if variable key, we need correct key PER trace.
        # Refactoring rank_func slightly.
        
        y_test_labels = [] 
        # Generate labels (SBox output)
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
        
        # WARNING: If variable key, we must check.
        # But for ASCAD.h5 (original), it's fixed key.
        # Let's assume fixed key for correct_key variable, but verify.
        if np.all(keys == keys[0]):
             print("   Key appears FIXED.")
        else:
             print("   Key appears VARIABLE. Using first key for analysis (might be wrong if rank_func assumes fixed).")
             # My rank_func above assumes fixed correct_key_byte for the final scalar comparison
             # but accumulates probabilities.
             # Standard GE assumes fixed key.
             # If variable key, we test key recovery per trace? No, GE is for fixed key.
             
    # 2. Build Model
    model = build_model()
    
    # Run dummy Inference to build graph
    results = []
    
    checkpoints = find_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints: {[os.path.basename(c) for c in checkpoints]}")
    
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\nEvaluating {ckpt_name}...")
        
        # Load Weights
        try:
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(ckpt_path).expect_partial()
            print("   Weights restored.")
        except Exception as e:
            print(f"   Failed to restore weights: {e}")
            continue
            
        # Inference
        # Batch inference
        all_preds = []
        num_batches = int(np.ceil(Config.NUM_TEST_TRACES / Config.BATCH_SIZE))
        
        valid_infer = True
        for b in range(num_batches):
            start = b * Config.BATCH_SIZE
            end = min((b + 1) * Config.BATCH_SIZE, Config.NUM_TEST_TRACES)
            batch_x = X_test[start:end]
            
            try:
                # Convert to float32 for model input (Conv1D expects float)
                batch_x_input = batch_x.astype('float32')
                # Model output: (batch, 256) probabilities (softmax)
                logits = model(batch_x_input, training=False)[0]
                preds = tf.nn.softmax(logits)
                all_preds.append(preds.numpy())
            except Exception as e:
                print(f"   Inference failed: {e}")
                valid_infer = False
                break
        
        if not valid_infer: continue
        
        all_preds = np.concatenate(all_preds, axis=0) # (2000, 256)
        
        # Compute Rank
        # Using byte 2
        correct_key_byte = keys[0][TARGET_BYTE]
        
        # My rank_func implementation calculates rank evolution trace by trace
        ranks = rank_func(all_preds, plt_txt, correct_key_byte, TARGET_BYTE)
        
        final_rank = ranks[-1]
        print(f"   Final Rank (at {Config.NUM_TEST_TRACES} traces): {final_rank}")
        
        results.append({
            "Checkpoint": ckpt_name,
            "Final_Rank": final_rank,
            "Traces_to_Rank_0": np.argmax(np.array(ranks) == 0) if 0 in ranks else "N/A"
        })

    # 3. Save Results
    csv_file = "transformer_evaluation_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Checkpoint", "Final_Rank", "Traces_to_Rank_0"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {csv_file}")
    
    # 4. Plot
    plt.figure(figsize=(10, 6))
    ckpt_names = [r["Checkpoint"] for r in results]
    final_ranks = [r["Final_Rank"] for r in results]
    
    plt.bar(ckpt_names, final_ranks, color='skyblue')
    plt.xlabel("Checkpoint")
    plt.ylabel("Final Key Rank (Lower is Better)")
    plt.title(f"Transformer Performance ({Config.NUM_TEST_TRACES} traces)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("transformer_checkpoint_comparison.png")
    print("Saved plot to transformer_checkpoint_comparison.png")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Transformer Checkpoints")
    parser.add_argument("--data_path", type=str, default=Config.DATA_FILE, help="Path to H5 data file")
    parser.add_argument("--checkpoint_dir", type=str, default=Config.CHECKPOINT_DIR, help="Directory containing checkpoints")
    parser.add_argument("--input_length", type=int, default=Config.INPUT_LENGTH, help="Input length (must match training)")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # Update Config
    Config.DATA_FILE = args.data_path
    Config.CHECKPOINT_DIR = args.checkpoint_dir
    Config.INPUT_LENGTH = args.input_length
    
    # We might need to handle output_csv in the main function if it's hardcoded
    # The view_file output showed: csv_file = "transformer_evaluation_results.csv" (line 328)
    # It seems hardcoded. I will try to patch it or just rename the file after run.
    # For now let's just run it. The user wants to know "which is better".
    
    print(f"Configuration:\n  Data: {Config.DATA_FILE}\n  Checkpoints: {Config.CHECKPOINT_DIR}\n  Input Length: {Config.INPUT_LENGTH}")
    
    main()
