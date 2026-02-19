"""
Plot multiple GE result files and annotate GE@100, GE@500 and key-recovery (first zero).
Usage example:
  python scripts/plot_ge_compare_with_annotations.py \
      --files results/mamba_gnn-50000.txt results/mamba_gnn_finetune_from_ce-55000.txt results/mamba_gnn_finetune_from_ce-60000.txt results/mamba_gnn_finetune_from_ce-75000.txt \
      --labels "CE-50k" "Focal-55k" "Focal-60k" "Focal-75k" \
      --output plots/ge_compare_focal_vs_ce_75k_annotated.png --log
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
    if len(lines) == 0:
        return None, None
    if len(lines) == 1:
        mean = np.array([float(x) for x in lines[0].split('\t') if x != ''])
        std = None
    else:
        mean = np.array([float(x) for x in lines[-1].split('\t') if x != ''])
        possible_std = lines[1] if len(lines) >= 2 else None
        try:
            std = np.array([float(x) for x in possible_std.split('\t') if x != '']) if possible_std else None
            if std is not None and len(std) != len(mean):
                std = None
        except Exception:
            std = None
    return mean, std


def format_num(x):
    return f"{x:.2f}" if x is not None else "N/A"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--files', nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=False)
    p.add_argument('--output', type=str, default='plots/ge_compare_annotated.png')
    p.add_argument('--log', action='store_true')
    p.add_argument('--annotate_points', nargs='+', type=int, default=[100, 500])
    args = p.parse_args()

    labels = args.labels if args.labels else [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    if len(labels) != len(args.files):
        raise SystemExit('Number of labels must match number of files')

    results = []
    for fp in args.files:
        mean, std = load_results(fp)
        results.append((mean, std, fp))

    min_len = min(len(r[0]) for r in results)
    x = np.arange(1, min_len + 1)

    plt.figure(figsize=(14, 7))
    cmap = plt.get_cmap('tab10')

    summary_lines = []
    for i, ((mean, std, fp), lab) in enumerate(zip(results, labels)):
        y = mean[:min_len]
        plt.plot(x, y, label=lab, color=cmap(i), linewidth=1.6)
        if std is not None:
            plt.fill_between(x, (mean - std)[:min_len], (mean + std)[:min_len], color=cmap(i), alpha=0.12)

        # numeric annotations
        ge100 = y[99] if min_len >= 100 else None
        ge500 = y[499] if min_len >= 500 else None
        # first index where GE==0 (key recovered)
        zero_idx = np.where(y == 0.0)[0]
        key_recov = int(zero_idx[0]) + 1 if zero_idx.size > 0 else None

        summary_lines.append(f"{lab}: GE@100={format_num(ge100)}, GE@500={format_num(ge500)}, key_recovery={key_recov if key_recov else 'not recovered'}")

        # plot markers
        for pt in args.annotate_points:
            if pt <= min_len:
                plt.scatter(pt, y[pt - 1], color=cmap(i), s=40, edgecolor='k', zorder=5)
                plt.text(pt, y[pt - 1], f" {pt}:{format_num(y[pt - 1])}", color=cmap(i), fontsize=9, verticalalignment='bottom')
        if key_recov:
            plt.axvline(key_recov, color=cmap(i), linestyle='--', alpha=0.6)
            plt.text(key_recov, plt.ylim()[1] * 0.9, f"{lab} recov@{key_recov}", color=cmap(i), fontsize=10, rotation=90, verticalalignment='top')

    if args.log:
        plt.yscale('log')
        plt.ylim(1e0, 256)
    else:
        plt.ylim(0, max(r[0][:min_len].max() for r in results) * 1.05)

    plt.xlabel('Number of Traces', fontsize=12)
    plt.ylabel('Key Rank (Guessing Entropy)', fontsize=12)
    plt.title('GE comparison (annotated)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')

    print('\n'.join(summary_lines))
    print(f'Annotated plot saved to: {args.output}')


if __name__ == '__main__':
    main()
