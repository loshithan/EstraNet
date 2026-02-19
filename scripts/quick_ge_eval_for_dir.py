"""
Run a QUICK GE (10 trials) for the latest checkpoint in a checkpoint directory.
Saves mean/std to results/quick_ge_{dirname}.txt
Usage:
  python scripts/quick_ge_eval_for_dir.py --checkpoint_dir checkpoints/mamba_gnn_finetune_from_ce
"""
import argparse
import importlib.util
import torch
from pathlib import Path
import numpy as np

# load train_mamba_gnn helpers
spec = importlib.util.spec_from_file_location('train_mamba_gnn', Path('mamba-gnn-scripts') / 'train_mamba_gnn.py')
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

p = argparse.ArgumentParser()
p.add_argument('--checkpoint_dir', type=str, required=True)
p.add_argument('--num_trials', type=int, default=10)
args = p.parse_args()

ckpt_dir = Path(args.checkpoint_dir)
if not ckpt_dir.exists():
    raise SystemExit(f'Checkpoint dir not found: {ckpt_dir}')

# find latest numeric checkpoint or fallback to checkpoint_latest.pth or best_model.pth
pths = sorted([p for p in ckpt_dir.glob('mamba_gnn-*.pth')])
if pths:
    ckpt_path = pths[-1]
else:
    if (ckpt_dir / 'best_model.pth').exists():
        ckpt_path = ckpt_dir / 'best_model.pth'
    else:
        ckpt_path = ckpt_dir / 'checkpoint_latest.pth'

print(f'Evaluating checkpoint: {ckpt_path}')
ck = torch.load(str(ckpt_path), map_location='cpu')
msd = ck.get('model_state_dict') or ck

# infer arch
try:
    d_model, mamba_layers, gnn_layers = train_mod.infer_arch_from_state(msd)
except Exception:
    d_model, mamba_layers, gnn_layers = 64, 2, 2

model = train_mod.OptimizedMambaGNN(
    trace_length=700,
    d_model=d_model,
    mamba_layers=mamba_layers,
    gnn_layers=gnn_layers,
    num_classes=256,
    k_neighbors=8,
    dropout=0.3
)
model.load_state_dict(msd)

# load attack data
_, _, X_attack, y_attack, m_attack = train_mod.load_ascad_data('data/ASCAD.h5', target_byte=2)
attack_dataset = train_mod.ASCADDataset(X_attack, y_attack)
attack_loader = torch.utils.data.DataLoader(attack_dataset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

mean_ranks, std_ranks = train_mod.evaluate_model_ge(model, attack_loader, m_attack, 2, device, num_trials=args.num_trials, max_traces=len(X_attack))

out_dir = Path('results')
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f'quick_ge_{ckpt_dir.name}_{ckpt_path.stem}.txt'
with open(out_path, 'w') as f:
    f.write('\t'.join(str(x) for x in mean_ranks) + '\n')
    f.write('\t'.join(str(x) for x in std_ranks) + '\n')

print(f'Quick GE saved to: {out_path}')
