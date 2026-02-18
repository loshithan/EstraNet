"""
Compare Mamba-GNN and EstraNet evaluation results side-by-side
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def load_results(result_file):
    """Load evaluation results from .txt file"""
    if not os.path.exists(result_file):
        print(f"Warning: {result_file} not found")
        return None, None
    
    with open(result_file, 'r') as f:
        lines = f.readlines()
    
    # Line 1: mean ranks, Line 2: std ranks (if available)
    mean_ranks = np.array([float(x) for x in lines[0].strip().split('\t') if x])
    
    if len(lines) > 1:
        std_ranks = np.array([float(x) for x in lines[1].strip().split('\t') if x])
    else:
        std_ranks = None
    
    return mean_ranks, std_ranks


def plot_comparison(mamba_results, estranet_results, output_path='comparison_plot.png'):
    """Plot guessing entropy comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Unpack results
    mamba_mean, mamba_std = mamba_results
    estranet_mean, estranet_std = estranet_results
    
    # Determine max traces
    max_traces = min(len(mamba_mean), len(estranet_mean))
    traces = np.arange(1, max_traces + 1)
    
    # Plot 1: Full range
    ax1.plot(traces, mamba_mean[:max_traces], 'b-', linewidth=2, label='Mamba-GNN')
    if mamba_std is not None:
        ax1.fill_between(traces, 
                         (mamba_mean - mamba_std)[:max_traces], 
                         (mamba_mean + mamba_std)[:max_traces], 
                         alpha=0.2, color='blue')
    
    ax1.plot(traces, estranet_mean[:max_traces], 'r-', linewidth=2, label='EstraNet')
    if estranet_std is not None:
        ax1.fill_between(traces, 
                         (estranet_mean - estranet_std)[:max_traces], 
                         (estranet_mean + estranet_std)[:max_traces], 
                         alpha=0.2, color='red')
    
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Key Recovered (rank=0)')
    ax1.set_xlabel('Number of Traces', fontsize=12)
    ax1.set_ylabel('Key Rank (Guessing Entropy)', fontsize=12)
    ax1.set_title('Guessing Entropy Comparison (Full Range)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 260)
    
    # Plot 2: Zoomed (first 5000 traces)
    zoom_traces = min(5000, max_traces)
    ax2.plot(traces[:zoom_traces], mamba_mean[:zoom_traces], 'b-', linewidth=2, label='Mamba-GNN')
    if mamba_std is not None:
        ax2.fill_between(traces[:zoom_traces], 
                         (mamba_mean - mamba_std)[:zoom_traces], 
                         (mamba_mean + mamba_std)[:zoom_traces], 
                         alpha=0.2, color='blue')
    
    ax2.plot(traces[:zoom_traces], estranet_mean[:zoom_traces], 'r-', linewidth=2, label='EstraNet')
    if estranet_std is not None:
        ax2.fill_between(traces[:zoom_traces], 
                         (estranet_mean - estranet_std)[:zoom_traces], 
                         (estranet_mean + estranet_std)[:zoom_traces], 
                         alpha=0.2, color='red')
    
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Key Recovered')
    ax2.set_xlabel('Number of Traces', fontsize=12)
    ax2.set_ylabel('Key Rank (Guessing Entropy)', fontsize=12)
    ax2.set_title('Guessing Entropy Comparison (Zoomed: 0-5000 traces)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


def print_comparison_table(mamba_results, estranet_results):
    """Print comparison table"""
    mamba_mean, mamba_std = mamba_results
    estranet_mean, estranet_std = estranet_results
    
    print("\n" + "="*100)
    print("GUESSING ENTROPY COMPARISON")
    print("="*100)
    print(f"\n{'Traces':<12} | {'Mamba-GNN':<25} | {'EstraNet':<25} | {'Winner':<10}")
    print("-" * 100)
    
    trace_counts = [100, 500, 1000, 2000, 5000, 10000]
    
    for n in trace_counts:
        if n <= len(mamba_mean) and n <= len(estranet_mean):
            mamba_val = mamba_mean[n-1]
            estranet_val = estranet_mean[n-1]
            
            mamba_str = f"{mamba_val:.2f}"
            if mamba_std is not None:
                mamba_str += f" ± {mamba_std[n-1]:.2f}"
            
            estranet_str = f"{estranet_val:.2f}"
            if estranet_std is not None:
                estranet_str += f" ± {estranet_std[n-1]:.2f}"
            
            if mamba_val < estranet_val:
                winner = "Mamba-GNN ✓"
            elif estranet_val < mamba_val:
                winner = "EstraNet ✓"
            else:
                winner = "Tie"
            
            print(f"{n:<12} | {mamba_str:<25} | {estranet_str:<25} | {winner:<10}")
    
    print("="*100)
    
    # Find first rank=0
    mamba_recovered = np.where(mamba_mean == 0)[0]
    estranet_recovered = np.where(estranet_mean == 0)[0]
    
    print("\nKey Recovery:")
    if len(mamba_recovered) > 0:
        print(f"  Mamba-GNN:  ✓ Recovered at {mamba_recovered[0]+1} traces")
    else:
        print(f"  Mamba-GNN:  ✗ Not recovered (best rank: {mamba_mean[-1]:.2f})")
    
    if len(estranet_recovered) > 0:
        print(f"  EstraNet:   ✓ Recovered at {estranet_recovered[0]+1} traces")
    else:
        print(f"  EstraNet:   ✗ Not recovered (best rank: {estranet_mean[-1]:.2f})")
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Compare Mamba-GNN and EstraNet results')
    parser.add_argument('--mamba_results', type=str, required=True,
                       help='Path to Mamba-GNN results .txt file')
    parser.add_argument('--estranet_results', type=str, required=True,
                       help='Path to EstraNet results .txt file')
    parser.add_argument('--output', type=str, default='comparison_plot.png',
                       help='Output plot filename')
    
    args = parser.parse_args()
    
    # Load results
    print("Loading Mamba-GNN results...")
    mamba_mean, mamba_std = load_results(args.mamba_results)
    
    print("Loading EstraNet results...")
    estranet_mean, estranet_std = load_results(args.estranet_results)
    
    if mamba_mean is None or estranet_mean is None:
        print("Error: Could not load results. Please check file paths.")
        return
    
    # Print comparison table
    print_comparison_table((mamba_mean, mamba_std), (estranet_mean, estranet_std))
    
    # Plot comparison
    plot_comparison((mamba_mean, mamba_std), (estranet_mean, estranet_std), args.output)


if __name__ == '__main__':
    main()
