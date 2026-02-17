
import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Constants
ASCAD_KEY = np.array([0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6, 0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C], dtype=np.uint8)

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return False
    return True

def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print(f"Error: can't open HDF5 file '{ascad_database_file}'")
        sys.exit(-1)

    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
    Y_profiling = np.array(in_file['Profiling_traces/labels'], dtype=np.int64)

    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
    Y_attack = np.array(in_file['Attack_traces/labels'], dtype=np.int64)
    
    metadata_attack = None
    if load_metadata:
        metadata_attack = in_file['Attack_traces/metadata']

    return (X_profiling, Y_profiling), (X_attack, Y_attack), metadata_attack

def rank(predictions, metadata, key_byte, num_traces=2000):
   
    # Real key for ASCAD is fixed
    real_key = 0x4D # Key byte 2 is 0x4D for ASCAD fixed key dataset (CHECK THIS usually it's known)
    # Actually, let's use the metadata to be safe
    # But wait, checking ASCAD.h5... the key is fixed. 
    # Let's derive it or use the metadata for each trace (which is the same)
    
    # Actually, the user's verify_ascad_labels.py script confirmed Byte 2 matches.
    # The key byte 2 value is: metadata['key'][0, 2]
    
    # Predictions: (num_traces, 256)
    
    min_trace_idx = 0
    max_trace_idx = num_traces
    
    # Pre-calculate log probs
    # predictions += 1e-40 # avoid log(0)
    # log_predictions = np.log(predictions)

    # We need to compute rank evolution.
    # But we can use the existing `rank` function logic from `evaluate_mamba_gnn.py` adapted?
    # Or implement a simple one here.

    # Let's implementation a simple full rank calculation.
    
    real_key = metadata['key'][0, 2]
    plaintext = metadata['plaintext'][:, 2]
    
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

    min_trace_idx = 0
    max_trace_idx = num_traces
    
    step = 1
    rank_evolution = np.zeros(num_traces)
    
    # Initialize key probabilities (log scale)
    key_probabilities = np.zeros(256)
    
    for i in tqdm(range(min_trace_idx, max_trace_idx), desc="Calculating Rank"):
        # For each trace
        p_t = plaintext[i]
        probs = predictions[i]
        
        # Update key probabilities
        # For each key candidate k, label = Sbox[p_t ^ k]
        # P(trace | k) = prob[label]
        # P(k | traces) = product(P(trace | k))
        # log P(k) = sum(log P(trace | k))
        
        for k in range(256):
            label = sbox[p_t ^ k]
            prob = probs[label]
            if prob < 1e-40:
                prob = 1e-40
            key_probabilities[k] += np.log(prob)
            
        # Get rank of real key
        # Sort keys by probability (descending)
        sorted_keys = np.argsort(key_probabilities)[::-1]
        real_key_rank = np.where(sorted_keys == real_key)[0][0]
        rank_evolution[i] = real_key_rank
        
    return rank_evolution

def evaluate_model(model_path, dataset_path="data/ASCAD.h5", output_path=None):
    print(f"Evaluating {model_path}...")
    
    # Load model
    model = None
    try:
        # custom_objects might be needed if custom metrics were used, but compile=False usually helps
        model = load_model(model_path, compile=False)
        print(f"Successfully loaded {model_path}")
    except Exception as e:
        print(f"Error loading model directly: {e}")
        
        # Fallback for MLP
        if "mlp" in model_path.lower() and "node200" in model_path and "layernb6" in model_path:
            print("Attempting to reconstruct MLP model manually...")
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Input
                
                model = Sequential()
                model.add(Input(shape=(700,)))
                # Adjusting to 5 hidden layers based on "found 6 saved layers" (5 hidden + 1 output)
                for _ in range(5):
                    model.add(Dense(200, activation='relu'))
                # Output layer
                model.add(Dense(256, activation='softmax'))
                
                model.load_weights(model_path)
                print("Successfully loaded weights into reconstructed MLP model")
            except Exception as e2:
                print(f"Error reconstructing/loading weights: {e2}")
                return
        else:
            return

    if model is None:
        return

    # Inspect input shape
    input_shape = model.input_shape
    print(f"Model Input Shape: {input_shape}")
    
    required_len = input_shape[1]
    required_channels = input_shape[2] if len(input_shape) > 2 else 1
    
    # Load Data
    (X_profiling, _), (X_attack, Y_attack), metadata = load_ascad(dataset_path, load_metadata=True)
    
    # Preprocessing (StandardScaler)
    # Fit on profiling traces 
    print("Fitting Scaler on Profiling traces...")
    scaler = StandardScaler()
    scaler.fit(X_profiling)
    
    X_attack = scaler.transform(X_attack)
    
    # Reshape if needed
    if len(input_shape) == 3: # CNN usually (N, 700, 1)
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    
    # Limit to 10k traces
    X_attack = X_attack[:10000]
    metadata = metadata[:10000]
    
    # Predictions
    print("Running Inference...")
    predictions = model.predict(X_attack, batch_size=200, verbose=1)
    
    # Rank
    print("Computing Rank...")
    rank_evolution = rank(predictions, metadata, 2, num_traces=10000)
    
    print(f"Final Rank: {rank_evolution[-1]}")
    
    # Save
    if output_path:
        np.save(output_path, rank_evolution)
        print(f"Saved rank to {output_path}")

if __name__ == "__main__":
    checkpoints = [
        "checkpoints/mlp_best_ascad_desync0_node200_layernb6_epochs200_classes256_batchsize100.h5",
        "checkpoints/cnn_best_ascad_desync0_epochs75_classes256_batchsize200.h5"
    ]
    
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            name = os.path.basename(ckpt).replace(".h5", "")
            output_file = f"rank_{name}.npy"
            evaluate_model(ckpt, output_path=output_file)
        else:
            print(f"Checkpoint not found: {ckpt}")
