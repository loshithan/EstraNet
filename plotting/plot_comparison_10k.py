
import numpy as np
import matplotlib.pyplot as plt

# Load data
try:
    rank50 = np.load("rank_desync50.npy")
    rank100 = np.load("rank_desync100.npy")
except FileNotFoundError as e:
    print(f"Error loading .npy files: {e}")
    exit(1)

# Create plot
plt.figure(figsize=(10, 6))

# Plot Desync50
plt.plot(rank50, label=f'Desync50 (Final Rank: {rank50[-1]:.2f})', color='blue', alpha=0.8)

# Plot Desync100
plt.plot(rank100, label=f'Desync100 (Final Rank: {rank100[-1]:.2f})', color='red', alpha=0.8)

plt.title('EstraNet Attack Comparison (10k Traces)')
plt.xlabel('Number of Traces')
plt.ylabel('Key Rank (Log Scale)')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

output_file = "comparison_plot_10k.png"
plt.savefig(output_file, dpi=300)
print(f"Comparison plot saved to: {output_file}")
