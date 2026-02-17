import numpy as np
import matplotlib.pyplot as plt

def compare():
    gnn_rank = np.load('rank_gnn10.npy')
    base_rank = np.load('rank_desync0.npy')
    
    # Truncate to same length if needed
    length = min(len(gnn_rank), len(base_rank))
    gnn_rank = gnn_rank[:length]
    base_rank = base_rank[:length]
    
    # Metrics
    gnn_final = gnn_rank[-1]
    base_final = base_rank[-1]
    
    # Average Rank (over the whole trace set)
    gnn_avg = np.mean(gnn_rank)
    base_avg = np.mean(base_rank)
    
    # Success check
    gnn_success = np.where(gnn_rank == 0)[0]
    base_success = np.where(base_rank == 0)[0]
    
    gnn_min_traces = gnn_success[0] + 1 if len(gnn_success) > 0 else "N/A"
    base_min_traces = base_success[0] + 1 if len(base_success) > 0 else "N/A"

    print(f"--- Comparison (10,000 Traces) ---")
    print(f"Metric              | GNN-10         | EstraNet (Base)")
    print(f"--------------------|----------------|----------------")
    print(f"Final Key Rank      | {gnn_final:<14.2f} | {base_final:<14.2f}")
    print(f"Average Key Rank    | {gnn_avg:<14.2f} | {base_avg:<14.2f}")
    print(f"Traces to Rank 0    | {gnn_min_traces:<14} | {base_min_traces:<14}")
    
    # Plotting for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(gnn_rank, label=f'GNN-10 (Avg: {gnn_avg:.1f})', color='purple', alpha=0.7)
    plt.plot(base_rank, label=f'EstraNet Baseline (Avg: {base_avg:.1f})', color='green', alpha=0.7)
    plt.yscale('log')
    plt.title('Performance Comparison: GNN-10 vs EstraNet Baseline')
    plt.xlabel('Number of Traces')
    plt.ylabel('Key Rank (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.savefig('gnn_vs_estranet_comparison.png', dpi=300)
    print("\n✅ Saved gnn_vs_estranet_comparison.png")

if __name__ == "__main__":
    compare()
