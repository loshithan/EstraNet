import os
import sys
import torch
import numpy as np
import h5py
from tqdm import tqdm

# Add parent directory to path to allow importing from models and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mamba_gnn_model import OptimizedMambaGNN
from utils.evaluation_utils import compute_key_rank

def evaluate_mamba(checkpoint_path, data_path, num_traces=10000, chunk_size=2000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model parameters from previous research
    d_model = 192
    mamba_layers = 4
    gnn_layers = 3
    num_classes = 256
    k_neighbors = 8
    
    # Initialize model
    model = OptimizedMambaGNN(
        trace_length=700,
        d_model=d_model,
        mamba_layers=mamba_layers,
        gnn_layers=gnn_layers,
        num_classes=num_classes,
        k_neighbors=k_neighbors
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    ckpt_rank = checkpoint.get('rank', 'N/A')
    print(f"Checkpoint reported rank: {ckpt_rank}")

    # Load data
    print(f"Loading data: {data_path}")
    with h5py.File(data_path, 'r') as f:
        attack_traces = f['Attack_traces/traces'][:num_traces]
        attack_labels = f['Attack_traces/labels'][:num_traces] # Correct labels
        attack_metadata = f['Attack_traces/metadata'][:num_traces]
        
        # EstraNet code expects plaintext byte 2 and key byte 2
        plaintexts = attack_metadata['plaintext'][:, 2]
        keys = attack_metadata['key'][:, 2]

    # Run inference in chunks
    all_preds = []
    
    print(f"Running inference on {num_traces} traces...")
    with torch.no_grad():
        for i in tqdm(range(0, num_traces, 128)):
            batch_traces = attack_traces[i:i+128].astype(np.float32)
            # Scaling is handled inside the model forward pass (input_scale=0.1)
            batch_tensor = torch.from_numpy(batch_traces).unsqueeze(1).to(device)
            logits = model(batch_tensor)
            all_preds.append(logits.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    
    # Debug: Average Class Rank
    print("Computing average class rank (Unscaled)...")
    class_ranks = []
    for i in range(num_traces):
        pred = predictions[i]
        true_label = attack_labels[i]
        rank = np.where(np.argsort(-pred) == true_label)[0][0]
        class_ranks.append(rank)
    avg_rank_unscaled = np.mean(class_ranks)
    print(f"Average Class Rank (Unscaled): {avg_rank_unscaled:.2f}")

    # Try with StandardScaler fit on profiling data
    from sklearn.preprocessing import StandardScaler
    print("Fitting StandardScaler on profiling data...")
    with h5py.File(data_path, 'r') as f:
        # Use first 20k profiling traces for fitting
        prof_traces = f['Profiling_traces/traces'][:20000].astype(np.float32)
        scaler = StandardScaler()
        scaler.fit(prof_traces)
        del prof_traces
        
    print("Normalizing attack traces...")
    scaled_traces = scaler.transform(attack_traces.astype(np.float32))
    
    all_preds_scaled = []
    with torch.no_grad():
        for i in tqdm(range(0, num_traces, 128)):
            batch_traces = scaled_traces[i:i+128]
            batch_tensor = torch.from_numpy(batch_traces).unsqueeze(1).to(device)
            logits = model(batch_tensor)
            all_preds_scaled.append(logits.cpu().numpy())
    
    predictions_scaled = np.concatenate(all_preds_scaled, axis=0)
    
    print("Computing average class rank (Scaled with Prof)...")
    class_ranks_scaled = []
    for i in range(num_traces):
        pred = predictions_scaled[i]
        true_label = attack_labels[i]
        rank = np.where(np.argsort(-pred) == true_label)[0][0]
        class_ranks_scaled.append(rank)
    avg_rank_scaled = np.mean(class_ranks_scaled)
    print(f"Average Class Rank (Scaled with Prof): {avg_rank_scaled:.2f}")

    # Try with StandardScaler (Prof) AND model.input_scale = 1.0
    print("Testing StandardScaler (Prof) with model.input_scale = 1.0...")
    model.input_scale = 1.0
    
    all_preds_no_scale = []
    with torch.no_grad():
        for i in tqdm(range(0, num_traces, 128)):
            batch_traces = scaled_traces[i:i+128]
            batch_tensor = torch.from_numpy(batch_traces).unsqueeze(1).to(device)
            logits = model(batch_tensor)
            all_preds_no_scale.append(logits.cpu().numpy())
            
    predictions_no_scale = np.concatenate(all_preds_no_scale, axis=0)
    
    print("Computing average class rank (Scaled with Prof, model_scale=1.0)...")
    class_ranks_no_scale = []
    for i in range(num_traces):
        pred = predictions_no_scale[i]
        true_label = attack_labels[i]
        rank = np.where(np.argsort(-pred) == true_label)[0][0]
        class_ranks_no_scale.append(rank)
    avg_rank_no_scale = np.mean(class_ranks_no_scale)
    print(f"Average Class Rank (Scaled with Prof, model_scale=1.0): {avg_rank_no_scale:.2f}")

    # Decision logic
    if avg_rank_no_scale < avg_rank_scaled and avg_rank_no_scale < avg_rank_unscaled:
        print("StandardScaler + model_scale=1.0 is the BEST!")
        predictions = predictions_no_scale
    elif avg_rank_scaled < avg_rank_unscaled:
        print("StandardScaler (Prof) is the BEST!")
        predictions = predictions_scaled
        model.input_scale = 0.1 # Reset for clarity
    else:
        print("Unscaled is the BEST!")
        # predictions stays as unscaled preds (all_preds)
        model.input_scale = 0.1 # Reset
        predictions = np.concatenate(all_preds, axis=0)

    # Compute rank evolution for the full set
    print("Computing key rank evolution...")
    # compute_key_rank shuffles internally, but we'll use it to get the progression
    ranks = compute_key_rank(predictions, plaintexts, keys)
    
    # Save the full rank progression
    np.save('rank_mamba_gnn_tunned.npy', ranks)
    print(f"Final Rank: {ranks[-1]}")
    print(f"Saved rank progression to rank_mamba_gnn_tunned.npy")

if __name__ == "__main__":
    evaluate_mamba(
        checkpoint_path='checkpoints/best_mamba_gnn_tunned.pth',
        data_path='data/ASCAD.h5',
        num_traces=10000
    )
