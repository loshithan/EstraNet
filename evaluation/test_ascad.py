import os
import sys
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing from models and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
try:
    from models.transformer import Transformer
    from models.fast_attention import SelfAttention
    import utils.data_utils as data_utils
    import utils.evaluation_utils as evaluation_utils
except ImportError as e:
    print(f"Error importing one of the project modules: {e}")
    sys.exit(1)

# ==========================================
# 1. CONFIGURATION (Must match training!)
# ==========================================
class Config:
    # Data params
    # 1. Select your target dataset (e.g. ASCAD.h5, ASCAD_desync50.h5)
    DATA_PATH = "data/ASCAD_desync100.h5"     
    
    # 2. Select the folder containing your trained checkpoints
    # CHANGE THIS to your actual checkpoint folder path
    CHECKPOINT_DIR = "checkpoints/checkpoints_transformer/" 
    
    # 3. Select the specific checkpoint (e.g. "trans_long-50")
    # or set to None to us the latest one in the folder.
    CHECKPOINT_FILE = None          
    
    # Model Architecture (Must match train_trans.py exactly)
    # NOTE: These values (e.g. N_HEAD=8) match the 'trans_long-11' checkpoint
    # and override the defaults found in train_trans.py (which uses n_head=4).
    N_LAYER = 2
    D_MODEL = 128
    D_HEAD = 32
    N_HEAD = 8
    D_INNER = 256
    N_HEAD_SOFTMAX = 8
    D_HEAD_SOFTMAX = 16
    DROPOUT = 0.05
    N_CLASSES = 256
    CONV_KERNEL_SIZE = 3
    N_CONV_LAYER = 2
    POOL_SIZE = 20
    D_KERNEL_MAP = 512
    BETA_HAT_2 = 150
    MODEL_NORMALIZATION = 'preLC'
    HEAD_INITIALIZATION = 'forward'
    SOFTMAX_ATTN = True
    OUTPUT_ATTN = False

    # Inference params
    # NOTE: INPUT_LENGTH=700 matches the 'input_length' flag in train_trans.py
    INPUT_LENGTH = 700  # Number of samples per trace
    NUM_TEST_TRACES = 10000 # Number of traces to test (Set to None for all)
    DATA_DESYNC = 0     # 0 for testing
    BATCH_SIZE = 50     # Adjust based on GPU memory

CONFIG = Config()

def check_env():
    # Helper to handle Colab Drive mounting if needed
    if 'google.colab' in sys.modules:
        print("Running in Google Colab")
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
    else:
        print("Running Locally")

def load_data():
    print(f"\nLoading ASCAD dataset from {CONFIG.DATA_PATH}...")
    if not os.path.exists(CONFIG.DATA_PATH):
        print(f"Dataset file not found at {CONFIG.DATA_PATH}")
        sys.exit(1)

    try:
        # Load test set (Attack Traces)
        test_data = data_utils.Dataset(
            CONFIG.DATA_PATH,
            "test",
            CONFIG.INPUT_LENGTH,
            CONFIG.DATA_DESYNC
        )
        print(f"Loaded {test_data.num_samples} test traces.")
        return test_data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def build_model():
    print("\nBuilding EstraNet Model...")
    model = Transformer(
        n_layer=CONFIG.N_LAYER,
        d_model=CONFIG.D_MODEL,
        d_head=CONFIG.D_HEAD,
        n_head=CONFIG.N_HEAD,
        d_inner=CONFIG.D_INNER,
        n_head_softmax=CONFIG.N_HEAD_SOFTMAX,
        d_head_softmax=CONFIG.D_HEAD_SOFTMAX,
        dropout=CONFIG.DROPOUT,
        n_classes=CONFIG.N_CLASSES,
        conv_kernel_size=CONFIG.CONV_KERNEL_SIZE,
        n_conv_layer=CONFIG.N_CONV_LAYER,
        pool_size=CONFIG.POOL_SIZE,
        d_kernel_map=CONFIG.D_KERNEL_MAP,
        beta_hat_2=CONFIG.BETA_HAT_2,
        model_normalization=CONFIG.MODEL_NORMALIZATION,
        head_initialization=CONFIG.HEAD_INITIALIZATION,
        softmax_attn=CONFIG.SOFTMAX_ATTN,
        output_attn=CONFIG.OUTPUT_ATTN
    )
    
    # Run a dummy input to initialize weights (build the graph)
    dummy_input = tf.zeros((1, CONFIG.INPUT_LENGTH))
    _ = model(dummy_input)
    print(" Model built and initialized (empty weights).")
    return model

def load_weights(model):
    print(f"\n Preparing to load weights from {CONFIG.CHECKPOINT_DIR}...")
    
    # Create checkpoint object pointing to our model
    checkpoint = tf.train.Checkpoint(model=model)
    
    # Try 1: Check if specific checkpoint file is defined
    chk_path = None
    if hasattr(CONFIG, 'CHECKPOINT_FILE') and CONFIG.CHECKPOINT_FILE:
        candidate = os.path.join(CONFIG.CHECKPOINT_DIR, CONFIG.CHECKPOINT_FILE)
        # Verify it exists (check for .index or .data file)
        if os.path.exists(candidate + ".index") or os.path.exists(candidate + ".data-00000-of-00001"):
            chk_path = candidate
            print(f" Using specific checkpoint: {chk_path}")

    # Try 2: Standard latest_checkpoint (requires 'checkpoint' file)
    if not chk_path:
        chk_path = tf.train.latest_checkpoint(CONFIG.CHECKPOINT_DIR)
    
    # Try 2: Manual search for prefix
    if not chk_path:
        print("   'checkpoint' file not found, searching for weight files...")
        if os.path.exists(CONFIG.CHECKPOINT_DIR):
            files = os.listdir(CONFIG.CHECKPOINT_DIR)
            # Look for .index files first (most reliable indicator)
            index_files = [f for f in files if f.endswith('.index')]
            if index_files:
                # Use the first one found, strip extension
                chk_path = os.path.join(CONFIG.CHECKPOINT_DIR, index_files[0][:-6])
            else:
                # Look for .data files
                data_files = [f for f in files if f.endswith('.data-00000-of-00001')]
                if data_files:
                    # Use the first one found, strip extension
                    chk_path = os.path.join(CONFIG.CHECKPOINT_DIR, data_files[0][:-20])
    
    if chk_path:
        print(f" Found checkpoint: {chk_path}")
        try:
            # .expect_partial() suppresses warnings about optimizer variables we don't need
            status = checkpoint.restore(chk_path).expect_partial()
            print(" Weights restored successfully!")
            return True
        except Exception as e:
            print(f" Failed to restore weights: {e}")
            return False
    else:
        print(" No checkpoint found! Check your CHECKPOINT_DIR path.")
        if os.path.exists(CONFIG.CHECKPOINT_DIR):
             print(f"Files in {CONFIG.CHECKPOINT_DIR}: {os.listdir(CONFIG.CHECKPOINT_DIR)}")
        return False

def run_inference(model, test_data):
    print(f"\n Running Inference on {test_data.num_samples} traces...")
    
    tf_dataset = test_data.GetTFRecords(CONFIG.BATCH_SIZE, training=False)
    
    all_predictions = []
    num_batches = test_data.num_samples // CONFIG.BATCH_SIZE
    
    # Limit number of batches if configured
    if hasattr(CONFIG, 'NUM_TEST_TRACES') and CONFIG.NUM_TEST_TRACES is not None:
        max_batches = CONFIG.NUM_TEST_TRACES // CONFIG.BATCH_SIZE
        num_batches = min(num_batches, max_batches)
        tf_dataset = tf_dataset.take(num_batches)
        print(f" Limiting inference to {CONFIG.NUM_TEST_TRACES} traces ({num_batches} batches).")

    progbar = tf.keras.utils.Progbar(num_batches)
    
    for i, (batch_traces, batch_labels) in enumerate(tf_dataset):
        # Run prediction
        # model() returns [logits]
        logits = model(inputs=batch_traces, training=False)[0]
        
        # Apply Softmax to get probabilities
        preds = tf.nn.softmax(logits)
        all_predictions.append(preds.numpy())
        progbar.update(i+1)
        
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Limit to number of predictions made (drop_remainder=True drops last partial batch)
    limit = len(all_predictions)
    print(f"\n Inference complete. Predictions shape: {all_predictions.shape}")
    return all_predictions, limit

def compute_rank(predictions, test_data, limit):
    print("\n Computing Key Rank (Averaging over 100 experiments)...")
    
    plaintexts = test_data.plaintexts[:limit]
    keys = test_data.keys[:limit]
    
    num_experiments = 100
    key_rank_list = []
    
    for i in range(num_experiments):
        if hasattr(evaluation_utils, 'compute_key_rank'):
             key_ranks = evaluation_utils.compute_key_rank(predictions, plaintexts, keys)
             key_rank_list.append(key_ranks)
        else:
             print(" Error: evaluation_utils.compute_key_rank not found.")
             return None
             
        if (i+1) % 10 == 0:
            print(f"   Experiment {i+1}/{num_experiments} done", end='\r')
            
    print("") # Newline
    scores = np.stack(key_rank_list, axis=0)
    mean_rank = np.mean(scores, axis=0)
    
    final_rank = mean_rank[-1]
    print(f"\nFinal Mean Rank (at {limit} traces): {final_rank:.4f}")
    
    if final_rank < 1.0:
        print("ATTACK SUCCESSFUL! (Rank < 1)")
    else:
        print("Attack not yet successful.")
        
    return mean_rank

def plot_results(mean_rank):
    plt.figure(figsize=(10, 6))
    plt.plot(mean_rank, label='Mean Rank (100 exps)')
    plt.title(f'EstraNet Attack on ASCAD')
    plt.xlabel('Number of Traces')
    plt.ylabel('Key Rank (Log Scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    
    plot_file = "result_rank_plot.png"
    plt.savefig(plot_file)
    print(f"\nRank plot saved to: {plot_file}")
    
    # Save raw data for later plotting
    npy_file = "result_rank.npy"
    np.save(npy_file, mean_rank)
    print(f"Rank data saved to: {npy_file}")
    # plt.show()

import argparse

if __name__ == "__main__":
    check_env()
    
    # --- Parse CLI Arguments ---
    parser = argparse.ArgumentParser(description="Evaluate ASCAD Transformer")
    parser.add_argument("--data_path", type=str, default=CONFIG.DATA_PATH, help="Path to H5 data file")
    parser.add_argument("--checkpoint_dir", type=str, default=CONFIG.CHECKPOINT_DIR, help="Directory containing checkpoints")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Specific checkpoint file (e.g. trans_long-50)")
    parser.add_argument("--input_length", type=int, default=CONFIG.INPUT_LENGTH, help="Input length (must match training)")
    parser.add_argument("--data_desync", type=int, default=CONFIG.DATA_DESYNC, help="Always 0 for testing unless debugging")
    
    args = parser.parse_args()
    
    # Update Config
    CONFIG.DATA_PATH = args.data_path
    CONFIG.CHECKPOINT_DIR = args.checkpoint_dir
    CONFIG.CHECKPOINT_FILE = args.checkpoint_file
    CONFIG.INPUT_LENGTH = args.input_length
    CONFIG.DATA_DESYNC = args.data_desync
    
    print(f"Configuration:\n  Data: {CONFIG.DATA_PATH}\n  Checkpoints: {CONFIG.CHECKPOINT_DIR}\n  Checkpoint File: {CONFIG.CHECKPOINT_FILE}\n  Input Length: {CONFIG.INPUT_LENGTH}\n  Desync: {CONFIG.DATA_DESYNC}")
    
    # 1. Load Data
    data = load_data()
    
    # 2. Build Model
    model = build_model()
    
    # 3. Load Weights
    if load_weights(model):
        
        # 4. Run Inference
        preds, limit = run_inference(model, data)
        
        # 5. Compute Rank
        mean_rank = compute_rank(preds, data, limit)
        
        # 6. Plot
        if mean_rank is not None:
            plot_results(mean_rank)
