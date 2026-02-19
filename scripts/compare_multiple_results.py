"""
Plot multiple Guessing-Entropy result files together (log y-axis option).
Usage example:
  python scripts/compare_multiple_results.py \
      --files results/mamba_gnn-50000.txt results/mamba_gnn-5000.txt results/mamba_gnn-best_model.txt results/mamba_gnn_layernorm_eval.txt \
      --labels "mamba_gnn-50000" "mamba_gnn-5000" "best_model" "layernorm" \
      --output results/comparison_mamba_gnn_vs_others.png --log
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # If two lines: first = mean, second = std. Otherwise treat last line as mean.
    if len(lines) == 0:
        return None, None
    if len(lines) == 1:
        mean = np.array([float(x) for x in lines[0].split('\t') if x != ''])
        std = None
    else:
        # prefer the last line as mean (some TF scripts append mean on last line)
        mean = np.array([float(x) for x in lines[-1].split('\t') if x != ''])
        # try to parse a std line if available (choose the second-to-last when appropriate)
        possible_std = lines[1] if len(lines) >= 2 else None
        try:
            std = np.array([float(x) for x in possible_std.split('\t') if x != '']) if possible_std else None
            if std is not None and len(std) != len(mean):
                std = None
        except Exception:
            std = None
    return mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--files', nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=False)
    p.add_argument('--output', type=str, default='results/comparison_mamba_gnn_vs_others.png')
    p.add_argument('--log', action='store_true')
    args = p.parse_args()

    labels = args.labels if args.labels else [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    if len(labels) != len(args.files):
        raise SystemExit('Number of labels must match number of files')

    results = []
    for fp in args.files:
        mean, std = load_results(fp)
        results.append((mean, std))

    # Determine shortest length across results
    min_len = min(len(r[0]) for r in results)
    x = np.arange(1, min_len + 1)

    plt.figure(figsize=(14, 7))
    cmap = plt.get_cmap('tab10')
    for i, ((mean, std), lab) in enumerate(zip(results, labels)):
        y = mean[:min_len]
        plt.plot(x, y, label=lab, color=cmap(i), linewidth=1.5)
        if std is not None:
            plt.fill_between(x, (mean - std)[:min_len], (mean + std)[:min_len], color=cmap(i), alpha=0.15)

    if args.log:
        plt.yscale('log')
        plt.ylim(1e0, 256)
    else:
        plt.ylim(0, max(r[0][:min_len].max() for r in results) * 1.05)

    plt.xlabel('Number of Traces', fontsize=12)
    plt.ylabel('Key Rank (Guessing Entropy)', fontsize=12)
    plt.title('Mamba-GNN — GE comparison (selected checkpoints)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {args.output}')


if __name__ == '__main__':
    main()
