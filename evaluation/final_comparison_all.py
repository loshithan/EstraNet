import numpy as np
import matplotlib.pyplot as plt

def final_compare():
    rank_v2 = np.load('rank_gnn10.npy')
    rank_v3 = np.load('rank_gnn_archive3_best.npy')
    rank_base = np.load('rank_desync0.npy')
    
    length = min(len(rank_v2), len(rank_v3), len(rank_base))
    rank_v2 = rank_v2[:length]
    rank_v3 = rank_v3[:length]
    rank_base = rank_base[:length]
    
    plt.figure(figsize=(12, 7))
    plt.plot(rank_v2, label=f'GNN Archive 2 Best (GNN-10) - Final: {rank_v2[-1]:.1f}', color='purple', alpha=0.7)
    plt.plot(rank_v3, label=f'GNN Archive 3 Best (GNN-3) - Final: {rank_v3[-1]:.1f}', color='orange', alpha=0.7)
    plt.plot(rank_base, label=f'EstraNet Baseline - Final: {rank_base[-1]:.1f}', color='green', alpha=0.7)
    
    plt.yscale('log')
    plt.title('Final GNN vs EstraNet Performance Comparison (10k Traces)')
    plt.xlabel('Number of Traces')
    plt.ylabel('Key Rank (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('gnn_archive_comparison.png', dpi=300)
    print("✅ Saved gnn_archive_comparison.png")
    
    # Text summary
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Model                | Final Rank | Avg Rank")
    print(f"---------------------|------------|----------")
    print(f"Archive 2 Best (10)  | {rank_v2[-1]:<10.2f} | {np.mean(rank_v2):<8.2f}")
    print(f"Archive 3 Best (3)   | {rank_v3[-1]:<10.2f} | {np.mean(rank_v3):<8.2f}")
    print(f"EstraNet Baseline    | {rank_base[-1]:<10.2f} | {np.mean(rank_base):<8.2f}")

if __name__ == "__main__":
    final_compare()
