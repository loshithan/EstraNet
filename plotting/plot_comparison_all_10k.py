
import numpy as np
import matplotlib.pyplot as plt
import os

# Files to load
files = {
    "Desync0 (trans_long-5)": "rank_desync0.npy",
    "Desync50 (trans_long-2)": "rank_desync50.npy",
    "Desync100 (trans_long-21)": "rank_desync100.npy"
}

plt.figure(figsize=(12, 7))

colors = ['green', 'blue', 'red']
styles = ['-', '--', '-.']

for i, (label, filename) in enumerate(files.items()):
    if os.path.exists(filename):
        try:
            rank_data = np.load(filename)
            final_rank = rank_data[-1]
            plt.plot(rank_data, label=f'{label} - Final Rank: {final_rank:.2f}', color=colors[i], linestyle=styles[i], linewidth=2)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    else:
        print(f"File not found: {filename}")

plt.title('EstraNet Attack Comparison (10k Traces)', fontsize=14)
plt.xlabel('Number of Traces', fontsize=12)
plt.ylabel('Mean Key Rank (Log Scale)', fontsize=12)
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

output_file = "comparison_plot_all_10k.png"
plt.savefig(output_file, dpi=300)
print(f"Comparison plot saved to: {output_file}")
