import numpy as np
import matplotlib.pyplot as plt
import os

def load_rank(path):
    if os.path.exists(path):
        return np.load(path)
    else:
        print(f"⚠️ Warning: {path} not found.")
        return None

def plot_comparison():
    # File paths
    files = {
        "Mamba-GNN (Rank 0)": "rank_mamba_gnn_tunned.npy",
        "CNN Baseline (Rank 2)": "rank_cnn_best_ascad_desync0_epochs75_classes256_batchsize200.npy",
        "MLP Baseline (Rank 12)": "rank_mlp_best_ascad_desync0_node200_layernb6_epochs200_classes256_batchsize100.npy",
        "EstraNet (Rank 13)": "rank_desync0.npy",
        "GNN Archive 3 (Rank 64)": "rank_gnn_archive3_best.npy"
    }

    plt.figure(figsize=(12, 8))
    
    # Colors: Green (Mamba), Red (CNN), Purple (MLP), Orange (EstraNet), Blue (GNN)
    colors = ['#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#1f77b4'] 
    markers = ['o', 'D', 'p', '^', 's']

    for i, (label, path) in enumerate(files.items()):
        rank_data = load_rank(path)
        if rank_data is not None:
            # Check shape
            print(f"{label}: {rank_data.shape} traces")
            
            # Subsample for plotting if too dense (e.g., plot every 100th point)
            steps = np.arange(1, len(rank_data) + 1)
            
            # Plot
            plt.plot(steps, rank_data, label=label, linewidth=2, color=colors[i])

    plt.yscale('log')
    plt.xlabel('Number of Traces')
    plt.ylabel('Mean Key Rank (Log Scale)')
    plt.title('Model Comparison: Rank Progression')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5) # Rank 0 line (though log scale can't show 0)
    plt.ylim(bottom=0.5) # Log scale lower bound

    output_path = "final_model_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Comparison plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()
