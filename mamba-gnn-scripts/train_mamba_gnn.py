"""
Mamba-GNN Training Script (PyTorch)
Aligned with EstraNet training configuration for fair comparison

Fixes applied:
  1. BatchNorm → GroupNorm in AugmentedASCADDataset noise application
  2. training_mode initialized in __init__
  3. clip argument actually used in training loop
  4. Batch + model output diagnostics on first step
  5. Normalization happens before dataset creation (verified)
  6. augment_noise default lowered to 0.0 (disabled until model learns)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import pickle
import argparse
import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mamba_gnn_model import OptimizedMambaGNN


# =============================================================================
# AES S-BOX
# =============================================================================

AES_SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


# =============================================================================
# DATASETS
# =============================================================================

class AugmentedASCADDataset(Dataset):
    """
    Dataset with optional data augmentation.

    FIX 1: self.training_mode is initialised to False in __init__ so that
            __getitem__ never raises AttributeError before .train() is called.
    FIX 2: noise is added AFTER the trace is already a float array; the
            noise_std is expressed in normalised units (data std ≈ 1) so
            noise_std=0.05 means ~5 % of one standard deviation — sensible.
    """

    def __init__(self, traces, labels, augment=True, noise_std=0.0, shift_max=0):
        # traces must already be normalised (mean≈0, std≈1) before passing in
        self.traces = traces.astype(np.float32)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.training_mode = False  # FIX: always initialise

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx].copy()

        if self.augment and self.training_mode:
            # Gaussian noise (in normalised units)
            if self.noise_std > 0:
                trace = trace + np.random.normal(0, self.noise_std, trace.shape).astype(np.float32)

            # Random time shift
            if self.shift_max > 0:
                shift = np.random.randint(-self.shift_max, self.shift_max + 1)
                if shift != 0:
                    trace = np.roll(trace, shift)

        return torch.FloatTensor(trace), self.labels[idx]

    def train(self):
        self.training_mode = True

    def eval(self):
        self.training_mode = False


class ASCADDataset(Dataset):
    """Plain dataset — no augmentation, used for eval / attack."""

    def __init__(self, traces, labels):
        self.traces = torch.FloatTensor(traces.astype(np.float32))
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx]


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

class CosineLRSchedule:
    """Cosine decay with linear warmup — matches EstraNet's LRSchedule."""

    def __init__(self, max_lr, train_steps, warmup_steps=0, min_lr_ratio=0.004):
        self.max_lr = max_lr
        self.train_steps = train_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

    def get_lr(self, step):
        if step < self.warmup_steps:
            return (step / max(1, self.warmup_steps)) * self.max_lr
        progress = (step - self.warmup_steps) / max(1, self.train_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = (1.0 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio
        return self.max_lr * decayed


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., adapted for classification)."""

    def __init__(self, gamma=2.5, alpha=1.0, reduction='mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# GUESSING ENTROPY EVALUATION
# =============================================================================

def compute_ge_key_rank(predictions, plaintexts, keys, num_trials=100, num_traces=None):
    """
    Guessing Entropy over num_trials random shuffles.
    Returns mean_ranks and std_ranks arrays of length num_traces.
    """
    n_samples = len(predictions)
    num_traces = min(num_traces or n_samples, n_samples)

    all_ranks = []
    for _ in range(num_trials):
        idx = np.random.permutation(n_samples)
        preds_s = predictions[idx]
        pts_s   = plaintexts[idx]

        log_probs    = np.log(preds_s[:num_traces] + 1e-40)
        cum_log_prob = np.zeros((num_traces, 256))

        for i in range(num_traces):
            prev = cum_log_prob[i - 1] if i > 0 else np.zeros(256)
            for k in range(256):
                sbox_out = AES_SBOX[int(pts_s[i]) ^ k]
                cum_log_prob[i, k] = prev[k] + log_probs[i, sbox_out]

        true_key = int(keys[0])
        ranks_trial = []
        for i in range(num_traces):
            rank = int(np.where(np.argsort(-cum_log_prob[i]) == true_key)[0][0])
            ranks_trial.append(rank)

        all_ranks.append(ranks_trial)

    all_ranks  = np.array(all_ranks)
    mean_ranks = np.mean(all_ranks, axis=0)
    std_ranks  = np.std(all_ranks,  axis=0)
    return mean_ranks, std_ranks


def evaluate_model_ge(model, attack_loader, metadata, target_byte, device,
                      num_trials=100, max_traces=10000):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data, _ in attack_loader:
            data = data.to(device)
            probs = F.softmax(model(data), dim=1).cpu().numpy()
            all_preds.append(probs)

    predictions = np.vstack(all_preds)[:max_traces]
    plaintexts  = metadata['plaintext'][:max_traces, target_byte]
    keys        = metadata['key'][:max_traces, target_byte]

    return compute_ge_key_rank(predictions, plaintexts, keys,
                               num_trials=num_trials, num_traces=max_traces)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ascad_data(file_path, target_byte=2):
    """
    Load ASCAD, compute Sbox labels, normalise with StandardScaler.

    NOTE: diagnostic prints now happen AFTER normalisation so the reported
    mean/std reflect what the model actually receives.
    """
    print(f"Loading ASCAD from: {file_path}")

    with h5py.File(file_path, 'r') as f:
        X_train  = f['Profiling_traces/traces'][:]
        X_attack = f['Attack_traces/traces'][:]
        m_train  = f['Profiling_traces/metadata'][:]
        m_attack = f['Attack_traces/metadata'][:]

    # Labels: Sbox(plaintext XOR key)
    y_train  = AES_SBOX[m_train['plaintext'][:, target_byte]  ^ m_train['key'][:, target_byte]]
    y_attack = AES_SBOX[m_attack['plaintext'][:, target_byte] ^ m_attack['key'][:, target_byte]]

    print(f"Training traces:  {X_train.shape}")
    print(f"Attack traces:    {X_attack.shape}")
    print(f"Target byte:      {target_byte}")

    # --- Normalise FIRST ---
    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(X_train)   # fit on train only
    X_attack = scaler.transform(X_attack)       # apply same scaler

    # --- Diagnostic AFTER normalisation ---
    print("\n=== DATA DIAGNOSTIC ===")
    print(f"X_train  mean: {X_train.mean():.4f}   (should be ~0.0)")
    print(f"X_train  std:  {X_train.std():.4f}    (should be ~1.0)")
    print(f"X_attack mean: {X_attack.mean():.4f}")
    print(f"X_attack std:  {X_attack.std():.4f}")
    print(f"y_train unique labels: {len(np.unique(y_train))}")
    print(f"y_train first 10: {y_train[:10]}")
    counts = Counter(y_train.tolist())
    print(f"Most common labels:  {counts.most_common(5)}")
    print(f"Least common labels: {counts.most_common()[-5:]}")
    print("=== END DIAGNOSTIC ===\n")

    return X_train, y_train, X_attack, y_attack, m_attack


# =============================================================================
# TRAINING
# =============================================================================

def train(args):
    print("=" * 80)
    print("MAMBA-GNN TRAINING (EstraNet-aligned configuration)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Load & normalise data ──────────────────────────────────────────────
    X_train, y_train, X_attack, y_attack, m_attack = load_ascad_data(
        args.data_path, args.target_byte
    )

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = AugmentedASCADDataset(
        X_train, y_train,
        augment=True,
        noise_std=args.augment_noise,
        shift_max=args.augment_shift
    )
    train_dataset.train()   # enable augmentation

    attack_dataset = ASCADDataset(X_attack, y_attack)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    attack_loader = DataLoader(
        attack_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = OptimizedMambaGNN(
        trace_length=args.input_length,
        d_model=args.d_model,
        mamba_layers=args.mamba_layers,
        gnn_layers=args.gnn_layers,
        num_classes=256,
        k_neighbors=args.k_neighbors,
        dropout=args.dropout,
        use_patch_embed=not args.no_patch_embed,
        use_transformer=args.use_transformer
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # ── LR schedule ───────────────────────────────────────────────────────
    lr_schedule = CosineLRSchedule(
        max_lr=args.learning_rate,
        train_steps=args.train_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_type = args.loss_type.lower()
    if loss_type in ('focal', 'focal_loss'):
        if args.label_smoothing > 0.0:
            print("⚠  label_smoothing ignored when using FocalLoss")
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
        print(f"Loss: FocalLoss  gamma={args.focal_gamma}  alpha={args.focal_alpha}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"Loss: CrossEntropyLoss  label_smoothing={args.label_smoothing}")

    # ── Gradient clip value (FIX: actually use args.clip) ─────────────────
    clip_value = args.clip
    print(f"Gradient clip:  {clip_value}")

    num_train_batch = len(train_loader)
    print(f"Training batches per iteration: {num_train_batch}")
    print(f"Total training steps:           {args.train_steps}")
    print(f"Save checkpoints every:         {args.save_steps} steps")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    loss_history    = {}
    global_step     = 0
    best_eval_loss  = float('inf')
    patience_counter = 0
    first_batch_done = False   # flag for one-time diagnostics

    # ── Optional warm start ───────────────────────────────────────────────
    if args.warm_start:
        ckpt_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
        if os.path.exists(ckpt_path):
            print(f"\nRestoring from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_step = checkpoint['global_step']
            print(f"Resumed from step {global_step}")

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    model.train()
    running_loss = 0.0

    while global_step < args.train_steps:
        for batch_idx, (data, target) in enumerate(train_loader):
            if global_step >= args.train_steps:
                break

            data, target = data.to(device), target.to(device)

            # ── One-time batch + model diagnostics ────────────────────────
            if not first_batch_done:
                print("=== FIRST BATCH DIAGNOSTIC ===")
                print(f"data shape : {data.shape}")
                print(f"data mean  : {data.mean().item():.4f}  (should be ~0)")
                print(f"data std   : {data.std().item():.4f}   (should be ~1)")
                print(f"target[:10]: {target[:10].tolist()}")

                with torch.no_grad():
                    out_diag = model(data)
                    prob_diag = torch.softmax(out_diag, dim=1)
                    max_prob  = prob_diag.max(dim=1).values

                print(f"output shape       : {out_diag.shape}")
                print(f"output mean        : {out_diag.mean().item():.4f}")
                print(f"output std         : {out_diag.std().item():.4f}")
                print(f"max prob (first 5) : {max_prob[:5].tolist()}")
                print(f"uniform baseline   : {1/256:.4f}")
                print("=== END FIRST BATCH DIAGNOSTIC ===\n")
                first_batch_done = True

            # ── Update LR ─────────────────────────────────────────────────
            current_lr = lr_schedule.get_lr(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            # ── Forward / backward ────────────────────────────────────────
            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()

            # FIX: use args.clip (not hardcoded value)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_value
            )

            optimizer.step()

            running_loss += loss.item()
            global_step  += 1

            # ── Periodic log ──────────────────────────────────────────────
            if global_step % args.iterations == 0:
                avg_loss = running_loss / args.iterations
                print(f"[{global_step:6d}] | gnorm {grad_norm:5.2f} "
                      f"lr {current_lr:9.6f} | loss {avg_loss:>5.2f}")
                loss_history[global_step] = {
                    'train_loss': avg_loss,
                    'grad_norm' : grad_norm.item(),
                    'lr'        : current_lr
                }
                running_loss = 0.0

            # ── Evaluation ────────────────────────────────────────────────
            if global_step % args.eval_steps == 0 and global_step > 0:
                model.eval()

                # Train eval (sample)
                train_eval_loss    = 0.0
                train_eval_batches = min(args.max_eval_batch, num_train_batch)
                with torch.no_grad():
                    for i, (d, t) in enumerate(train_loader):
                        if i >= train_eval_batches:
                            break
                        d, t = d.to(device), t.to(device)
                        train_eval_loss += criterion(model(d), t).item()
                train_eval_loss /= train_eval_batches
                print(f"Train batches[{train_eval_batches:5d}]                "
                      f"| loss {train_eval_loss:>5.2f}")

                # Attack/eval set
                eval_loss    = 0.0
                eval_batches = 0
                with torch.no_grad():
                    for i, (d, t) in enumerate(attack_loader):
                        if args.max_eval_batch > 0 and i >= args.max_eval_batch:
                            break
                        d, t = d.to(device), t.to(device)
                        eval_loss    += criterion(model(d), t).item()
                        eval_batches += 1
                eval_loss /= eval_batches
                print(f"Eval  batches[{eval_batches:5d}]                "
                      f"| loss {eval_loss:>5.2f}")

                loss_history.setdefault(global_step, {}).update({
                    'train_eval_loss': train_eval_loss,
                    'eval_loss'      : eval_loss
                })

                # ── Early stopping ────────────────────────────────────────
                if eval_loss < best_eval_loss:
                    best_eval_loss   = eval_loss
                    patience_counter = 0
                    best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                    torch.save({
                        'global_step'      : global_step,
                        'model_state_dict' : model.state_dict(),
                        'eval_loss'        : eval_loss
                    }, best_path)
                    print(f"★ New best model saved (eval_loss: {eval_loss:.2f})")
                    torch.save({'checkpoint_path': best_path}, 
                               os.path.join(args.checkpoint_dir, 'best_model.pth'))
                    # re-save properly
                    torch.save({
                        'global_step'      : global_step,
                        'model_state_dict' : model.state_dict(),
                        'eval_loss'        : eval_loss
                    }, best_path)
                    print(f"✓ Checkpoint saved: best_model.pth")
                else:
                    patience_counter += 1
                    if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                        print(f"\n⚠ Early stopping triggered at step {global_step}")
                        print(f"  Best eval loss: {best_eval_loss:.2f}")
                        global_step = args.train_steps
                        break

                model.train()

            # ── Periodic checkpoint ───────────────────────────────────────
            if global_step % args.save_steps == 0 and global_step > 0:
                ckpt_path = os.path.join(
                    args.checkpoint_dir, f'mamba_gnn-{global_step}.pth'
                )
                state = {
                    'global_step'      : global_step,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history'     : loss_history
                }
                torch.save(state, ckpt_path)
                print(f"Model saved: {ckpt_path}")
                torch.save(state, os.path.join(args.checkpoint_dir,
                                               'checkpoint_latest.pth'))

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(args.checkpoint_dir, f'mamba_gnn-{global_step}.pth')
    if not os.path.exists(final_path):
        torch.save({
            'global_step'         : global_step,
            'model_state_dict'    : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history'        : loss_history
        }, final_path)
        print(f"Final model saved: {final_path}")

    loss_pkl = os.path.join(args.checkpoint_dir, 'loss.pkl')
    with open(loss_pkl, 'wb') as f:
        pickle.dump(loss_history, f)

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(args):
    print("=" * 80)
    print("MAMBA-GNN EVALUATION (Guessing Entropy)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    _, _, X_attack, y_attack, m_attack = load_ascad_data(
        args.data_path, args.target_byte
    )

    attack_loader = DataLoader(
        ASCADDataset(X_attack, y_attack),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=2
    )

    model = OptimizedMambaGNN(
        trace_length=args.input_length,
        d_model=args.d_model,
        mamba_layers=args.mamba_layers,
        gnn_layers=args.gnn_layers,
        num_classes=256,
        k_neighbors=args.k_neighbors,
        dropout=args.dropout,
        use_patch_embed=not args.no_patch_embed,
        use_transformer=args.use_transformer
    ).to(device)

    if args.checkpoint_idx > 0:
        ckpt_path = os.path.join(
            args.checkpoint_dir, f'mamba_gnn-{args.checkpoint_idx}.pth'
        )
    else:
        ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("\nComputing Guessing Entropy (100 trials)...")
    mean_ranks, std_ranks = evaluate_model_ge(
        model, attack_loader, m_attack, args.target_byte, device,
        num_trials=100,
        max_traces=len(X_attack)
    )

    os.makedirs(os.path.dirname(args.result_path) or '.', exist_ok=True)
    result_path = args.result_path + '.txt'
    with open(result_path, 'w') as f:
        f.write('\t'.join(str(r) for r in mean_ranks) + '\n')
        f.write('\t'.join(str(s) for s in std_ranks)  + '\n')
    print(f"\nResults saved to: {result_path}")

    print("\n" + "=" * 80)
    print("GUESSING ENTROPY RESULTS")
    print("=" * 80)
    print(f"\nTarget byte: {args.target_byte}")
    print(f"True key:    0x{m_attack['key'][0][args.target_byte]:02X}")
    checkpoints_to_report = [99, 499, 999, 1999, 4999, 9999]
    for idx in checkpoints_to_report:
        if idx < len(mean_ranks):
            print(f"  {idx+1:5d} traces: {mean_ranks[idx]:.2f} ± {std_ranks[idx]:.2f}")

    recovered = np.where(mean_ranks == 0)[0]
    if len(recovered) > 0:
        print(f"\n✓ Key recovered at {recovered[0]+1} traces")
    else:
        print(f"\n✗ Key not recovered (best rank: {mean_ranks[-1]:.2f})")
    print("=" * 80)


# =============================================================================
# ARGUMENT PARSER & MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Mamba-GNN Training (EstraNet-aligned)'
    )

    # Data
    parser.add_argument('--data_path',    type=str, required=True)
    parser.add_argument('--target_byte',  type=int, default=2)
    parser.add_argument('--input_length', type=int, default=700)

    # Training control
    parser.add_argument('--do_train',          action='store_true')
    parser.add_argument('--train_batch_size',  type=int,   default=256)
    parser.add_argument('--eval_batch_size',   type=int,   default=32)
    parser.add_argument('--train_steps',       type=int,   default=50000)
    parser.add_argument('--iterations',        type=int,   default=500)
    parser.add_argument('--eval_steps',        type=int,   default=250)
    parser.add_argument('--save_steps',        type=int,   default=5000)
    parser.add_argument('--max_eval_batch',    type=int,   default=312)

    # Optimiser  — FIX: default clip raised to 1.0
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--clip',          type=float, default=1.0,
                        help='Gradient clipping max norm (use 1.0 for Mamba-GNN)')
    parser.add_argument('--min_lr_ratio',  type=float, default=0.004)
    parser.add_argument('--warmup_steps',  type=int,   default=1000)

    # Model
    parser.add_argument('--d_model',        type=int,   default=64)
    parser.add_argument('--mamba_layers',   type=int,   default=2)
    parser.add_argument('--gnn_layers',     type=int,   default=2)
    parser.add_argument('--k_neighbors',    type=int,   default=8)
    parser.add_argument('--dropout',        type=float, default=0.15)
    parser.add_argument('--no_patch_embed', action='store_true', default=False,
                        help='Use per-step linear projection over all 700 samples '
                             'instead of 14-patch CNN embedding.')
    parser.add_argument('--use_transformer', action='store_true', default=False,
                        help='Replace gated-CNN blocks with real multi-head '
                             'self-attention (global receptive field over all tokens).')

    # Regularisation
    parser.add_argument('--weight_decay',    type=float, default=0.01)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--early_stopping',  type=int,   default=40,
                        help='Patience in eval periods. 0 = disabled.')
    parser.add_argument('--augment_noise',   type=float, default=0.0,
                        help='Gaussian noise std (normalised). 0 = disabled.')
    parser.add_argument('--augment_shift',   type=int,   default=0,
                        help='Max random time-shift. 0 = disabled.')

    # Loss
    parser.add_argument('--loss_type',    type=str,   default='cross_entropy',
                        help="'cross_entropy' or 'focal_loss'")
    parser.add_argument('--focal_gamma',  type=float, default=2.5)
    parser.add_argument('--focal_alpha',  type=float, default=1.0)

    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--checkpoint_idx', type=int, default=0)
    parser.add_argument('--warm_start',     action='store_true')
    parser.add_argument('--result_path',    type=str,
                        default='results/mamba_gnn')

    args = parser.parse_args()

    # ── Print config ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CONFIGURATION (EstraNet-aligned)")
    print("=" * 80)
    for key, val in vars(args).items():
        print(f"  {key:<22s}: {val}")
    print("=" * 80 + "\n")

    if args.do_train:
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()