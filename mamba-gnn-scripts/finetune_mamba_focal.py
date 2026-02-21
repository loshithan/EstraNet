"""
finetune_mamba_focal.py

Utility wrapper to fine‑tune an existing Cross‑Entropy checkpoint using FocalLoss.
Features:
 - copies a source checkpoint into a new target checkpoint folder as `checkpoint_latest.pth`
 - infers model architecture from the checkpoint state_dict (d_model, mamba/gnn layers)
 - computes an appropriate `train_steps` (source_global_step + extra_steps) unless overridden
 - launches `train_mamba_gnn.py --do_train --warm_start` with FocalLoss params

Usage (example):
python mamba-gnn-scripts/finetune_mamba_focal.py \
  --source_ckpt checkpoints/mamba_gnn_estranet/mamba_gnn-50000.pth \
  --target_dir checkpoints/mamba_gnn_finetune_from_ce \
  --extra_steps 25000 --focal_gamma 2.5 --learning_rate 1e-4

"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

# ----------------------------------------------------------------------------
def infer_arch_from_state(msd: dict):
    """Infer full architecture from checkpoint state_dict keys/shapes.

    Returns dict with: d_model, mamba_layers, gnn_layers,
                       use_ssm_mamba, use_patch_embed, d_state
    """
    # infer d_model from classifier or pos_encoding
    if 'classifier.0.weight' in msd:
        d_model = int(msd['classifier.0.weight'].shape[1])
    elif 'pos_encoding' in msd:
        d_model = int(msd['pos_encoding'].shape[2])
    else:
        raise RuntimeError('Cannot infer d_model from state_dict')

    # infer mamba_layers and gnn_layers by checking keys
    mamba_layers = len({k.split('.')[1] for k in msd.keys() if k.startswith('mamba_blocks.')})
    gnn_layers = len({k.split('.')[1] for k in msd.keys() if k.startswith('gnn_layers.')})
    # NOTE: gnn_layers=0 is valid (SSM-only model) — do NOT clamp to min 1

    # detect SSM vs legacy: SelectiveMambaBlock has A_log; OptimizedMambaBlock has conv_norm
    use_ssm_mamba = any('A_log' in k for k in msd.keys())

    # detect embedding mode
    use_patch_embed = any(k.startswith('patch_embed.') for k in msd.keys())
    # stride_embed detected by presence of stride_proj weight
    stride_embed    = any(k.startswith('stride_proj.') for k in msd.keys())

    # infer d_state from A_log shape: [d_inner, d_state]
    d_state = 16  # default fallback
    for k, v in msd.items():
        if k.endswith('.A_log') and hasattr(v, 'shape') and len(v.shape) == 2:
            d_state = int(v.shape[1])
            break

    return {
        'd_model': d_model,
        'mamba_layers': max(1, mamba_layers),
        'gnn_layers': gnn_layers,         # allow 0
        'use_ssm_mamba': use_ssm_mamba,
        'use_patch_embed': use_patch_embed,
        'stride_embed': stride_embed,
        'd_state': d_state,
    }


# ----------------------------------------------------------------------------
def copy_ckpt_to_target(src: Path, target_dir: Path) -> dict:
    if not src.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src}")
    target_dir.mkdir(parents=True, exist_ok=True)

    # copy original file for traceability
    dst_copy = target_dir / src.name
    shutil.copy2(src, dst_copy)

    # also create checkpoint_latest.pth so train_mamba_gnn.py --warm_start works
    dst_latest = target_dir / 'checkpoint_latest.pth'
    shutil.copy2(src, dst_latest)

    # load checkpoint metadata
    ck = torch.load(str(src), map_location='cpu')
    return ck


# ----------------------------------------------------------------------------
def build_train_command(cfg: dict) -> str:
    parts = [
        sys.executable, '-u', 'mamba-gnn-scripts/train_mamba_gnn.py',
        '--do_train', '--warm_start',
        f"--data_path={cfg['data_path']}",
        f"--checkpoint_dir={cfg['checkpoint_dir']}",
        f"--train_steps={cfg['train_steps']}",
        f"--save_steps={cfg['save_steps']}",
        f"--eval_steps={cfg['eval_steps']}",
        f"--learning_rate={cfg['learning_rate']}",
        f"--loss_type=focal",
        f"--focal_gamma={cfg['focal_gamma']}",
        f"--focal_alpha={cfg['focal_alpha']}",
        f"--label_smoothing=0.0",
        f"--train_batch_size={cfg['train_batch_size']}",
        f"--eval_batch_size={cfg['eval_batch_size']}",
        f"--d_model={cfg['d_model']}",
        f"--mamba_layers={cfg['mamba_layers']}",
        f"--gnn_layers={cfg['gnn_layers']}",
        f"--dropout={cfg['dropout']}",
    ]

    # SSM / embedding flags — MUST match the checkpoint architecture
    if cfg.get('use_ssm_mamba'):
        parts.append('--use_ssm_mamba')
    if not cfg.get('use_patch_embed', True):
        parts.append('--no_patch_embed')
    if cfg.get('stride_embed'):
        parts.append('--stride_embed')
    if cfg.get('d_state') is not None:
        parts.append(f"--d_state={cfg['d_state']}")
    if cfg.get('early_stopping') is not None:
        parts.append(f"--early_stopping={cfg['early_stopping']}")
    if cfg.get('warmup_steps') is not None:
        parts.append(f"--warmup_steps={cfg['warmup_steps']}")

    # Reset flags for fine-tuning
    if cfg.get('reset_lr_schedule'):
        parts.append('--reset_lr_schedule')
    if cfg.get('reset_optimizer'):
        parts.append('--reset_optimizer')

    if cfg.get('weight_decay') is not None:
        parts.append(f"--weight_decay={cfg['weight_decay']}")
    if cfg.get('augment_noise') is not None:
        parts.append(f"--augment_noise={cfg['augment_noise']}")
    if cfg.get('augment_shift') is not None:
        parts.append(f"--augment_shift={cfg['augment_shift']}")
    return ' '.join(map(str, parts))


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Fine‑tune a CE checkpoint with FocalLoss (warm‑start)')

    p.add_argument('--source_ckpt', type=str,
                   default='checkpoints/mamba_gnn_estranet/mamba_gnn-50000.pth',
                   help='Path to existing CE checkpoint (mamba_gnn-*.pth)')
    p.add_argument('--target_dir', type=str, default='checkpoints/mamba_gnn_finetune_from_ce',
                   help='Directory where fine‑tune checkpoints will be saved')
    p.add_argument('--extra_steps', type=int, default=25000,
                   help='Additional steps to run on top of source global_step')
    p.add_argument('--train_steps', type=int, default=None,
                   help='Optional total train_steps (overrides global_step+extra_steps calculation)')
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--focal_gamma', type=float, default=2.5)
    p.add_argument('--focal_alpha', type=float, default=1.0)
    p.add_argument('--train_batch_size', type=int, default=256)
    p.add_argument('--eval_batch_size', type=int, default=32)
    p.add_argument('--save_steps', type=int, default=5000)
    p.add_argument('--eval_steps', type=int, default=250)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--augment_noise', type=float, default=None)
    p.add_argument('--augment_shift', type=int, default=None)
    p.add_argument('--data_path', type=str, default='data/ASCAD.h5')
    p.add_argument('--dry_run', action='store_true')

    # Architecture override (auto-detected from checkpoint if not given)
    p.add_argument('--no_patch_embed', action='store_true', default=None,
                   help='Override: per-step linear (auto-detected from ckpt)')
    p.add_argument('--use_ssm_mamba', action='store_true', default=None,
                   help='Override: real S6 SSM (auto-detected from ckpt)')
    p.add_argument('--d_state', type=int, default=None,
                   help='Override: SSM state dim (auto-detected from ckpt)')

    # Fine-tune specific
    p.add_argument('--early_stopping', type=int, default=0,
                   help='Patience in eval periods. 0 = disabled.')
    p.add_argument('--warmup_steps', type=int, default=500,
                   help='LR warmup steps for fine-tune phase (default: 500)')
    p.add_argument('--reset_lr_schedule', action='store_true', default=True,
                   help='Reset cosine LR to start fresh from warm-start step (default: True)')
    p.add_argument('--no_reset_lr_schedule', dest='reset_lr_schedule', action='store_false',
                   help='Keep original cosine schedule position')
    p.add_argument('--reset_optimizer', action='store_true', default=True,
                   help='Discard old optimizer state (default: True — recommended when switching loss)')
    p.add_argument('--no_reset_optimizer', dest='reset_optimizer', action='store_false',
                   help='Keep old Adam momentum/variance buffers')

    args = p.parse_args()

    src = Path(args.source_ckpt)
    tgt = Path(args.target_dir)

    print(f"Source checkpoint: {src}")
    print(f"Target folder     : {tgt}")

    ck = copy_ckpt_to_target(src, tgt)
    print(f"✔ Copied checkpoint -> {tgt}/checkpoint_latest.pth")

    # determine global_step
    src_step = int(ck.get('global_step', 0) or 0)
    print(f"Source global_step: {src_step}")

    # determine architecture from state_dict
    msd = ck.get('model_state_dict') or ck
    try:
        arch = infer_arch_from_state(msd)
        print(f"Inferred arch -> d_model={arch['d_model']}, "
              f"mamba_layers={arch['mamba_layers']}, gnn_layers={arch['gnn_layers']}, "
              f"ssm={arch['use_ssm_mamba']}, patch_embed={arch['use_patch_embed']}, "
              f"d_state={arch['d_state']}")
    except Exception as e:
        print(f'⚠ Could not infer architecture from checkpoint ({e}) — using defaults')
        arch = {'d_model': 64, 'mamba_layers': 2, 'gnn_layers': 2,
                'use_ssm_mamba': False, 'use_patch_embed': True, 'd_state': 16}

    # CLI overrides take precedence over auto-detection
    d_model      = arch['d_model']
    mamba_layers = arch['mamba_layers']
    gnn_layers   = arch['gnn_layers']
    use_ssm      = args.use_ssm_mamba if args.use_ssm_mamba is not None else arch['use_ssm_mamba']
    use_patch    = not args.no_patch_embed if args.no_patch_embed is not None else arch['use_patch_embed']
    d_state      = args.d_state if args.d_state is not None else arch['d_state']

    # compute total train_steps
    if args.train_steps is None:
        total_steps = max(src_step + args.extra_steps, src_step + 1)
    else:
        total_steps = args.train_steps
        if total_steps <= src_step:
            print('⚠ requested train_steps <= source global_step; bumping to source_step + extra_steps')
            total_steps = src_step + args.extra_steps

    cfg = {
        'data_path': args.data_path,
        'checkpoint_dir': str(tgt),
        'train_steps': total_steps,
        'save_steps': args.save_steps,
        'eval_steps': args.eval_steps,
        'learning_rate': args.learning_rate,
        'focal_gamma': args.focal_gamma,
        'focal_alpha': args.focal_alpha,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'd_model': d_model,
        'mamba_layers': mamba_layers,
        'gnn_layers': gnn_layers,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'augment_noise': args.augment_noise,
        'augment_shift': args.augment_shift,
        # SSM / embed flags (auto-detected or overridden)
        'use_ssm_mamba': use_ssm,
        'use_patch_embed': use_patch,
        'd_state': d_state,
        'early_stopping': args.early_stopping,
        'warmup_steps': args.warmup_steps,
        'reset_lr_schedule': args.reset_lr_schedule,
        'reset_optimizer': args.reset_optimizer,
    }

    train_cmd = build_train_command(cfg)

    print('\n=== Fine-tune configuration ===')
    print(f"  total_train_steps : {total_steps}")
    print(f"  learning_rate     : {args.learning_rate}")
    print(f"  focal_gamma       : {args.focal_gamma}")
    print(f"  train batch size  : {args.train_batch_size}")
    print(f"  checkpoint target : {tgt}")
    print(f"  use_ssm_mamba     : {use_ssm}")
    print(f"  use_patch_embed   : {use_patch}")
    print(f"  d_state           : {d_state}")
    print(f"  reset_lr_schedule : {args.reset_lr_schedule}")
    print(f"  reset_optimizer   : {args.reset_optimizer}")
    print('  (label_smoothing forced to 0.0 for FocalLoss)')
    print('===============================\n')

    print('Training command:')
    print(train_cmd)

    if args.dry_run:
        print('\nDry run requested — not launching training')
        return

    # launch training (streams output)
    rc = subprocess.call(train_cmd, shell=True)
    if rc == 0:
        print('\n★ Fine‑tune job completed (exit 0)')
    else:
        print(f'\n✗ Fine‑tune job exited with code: {rc}')


if __name__ == '__main__':
    main()
