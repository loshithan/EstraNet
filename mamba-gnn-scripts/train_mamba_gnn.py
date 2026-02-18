"""
Mamba-GNN Training Script (PyTorch)
Aligned with EstraNet training configuration for fair comparison
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mamba_gnn_model import OptimizedMambaGNN


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class AugmentedASCADDataset(Dataset):
    """Dataset with data augmentation to prevent overfitting"""
    def __init__(self, traces, labels, augment=True, noise_std=0.1, shift_max=5):
        self.traces = traces
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.noise_std = noise_std
        self.shift_max = shift_max

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx].copy()
        
        if self.augment and self.training_mode:
            # Add Gaussian noise
            trace = trace + np.random.normal(0, self.noise_std, trace.shape)
            
            # Random time shift
            shift = np.random.randint(-self.shift_max, self.shift_max + 1)
            if shift != 0:
                trace = np.roll(trace, shift)
        
        return torch.FloatTensor(trace), self.labels[idx]
    
    def train(self):
        self.training_mode = True
    
    def eval(self):
        self.training_mode = False

# AES S-box for label computation
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
# DATASET
# =============================================================================

class ASCADDataset(Dataset):
    def __init__(self, traces, labels):
        self.traces = torch.FloatTensor(traces)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx]


# =============================================================================
# LEARNING RATE SCHEDULER (Matching EstraNet)
# =============================================================================

class CosineLRSchedule:
    """Cosine decay with warmup - matches EstraNet's LRSchedule"""
    def __init__(self, max_lr, train_steps, warmup_steps=0, min_lr_ratio=0.004):
        self.max_lr = max_lr
        self.train_steps = train_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
    
    def get_lr(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return (step / self.warmup_steps) * self.max_lr
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.train_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed = (1.0 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio
            return self.max_lr * decayed


# =============================================================================
# GUESSING ENTROPY EVALUATION (Matching EstraNet - 100 trials)
# =============================================================================

def compute_ge_key_rank(predictions, plaintexts, keys, num_trials=100, num_traces=None):
    """
    Compute Guessing Entropy (average key rank over multiple shuffled trials)
    Matches EstraNet's evaluation methodology
    
    Args:
        predictions: Model output probabilities [n_samples, 256]
        plaintexts: Plaintext bytes [n_samples]
        keys: Key bytes [n_samples] (all same value)
        num_trials: Number of independent trials (default: 100 like EstraNet)
        num_traces: Max traces to use per trial (default: all)
    
    Returns:
        key_ranks: Array of ranks at each trace count [num_traces]
        std_ranks: Standard deviation of ranks [num_traces]
    """
    n_samples = len(predictions)
    if num_traces is None:
        num_traces = n_samples
    
    num_traces = min(num_traces, n_samples)
    
    # Run multiple trials with random shuffling
    all_ranks = []
    for trial in range(num_trials):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        preds_shuffled = predictions[indices]
        pts_shuffled = plaintexts[indices]
        
        # Compute cumulative log probabilities for each key hypothesis
        log_probs = np.log(preds_shuffled[:num_traces] + 1e-40)
        cum_log_prob = np.zeros((num_traces, 256))
        
        for i in range(num_traces):
            # For each key hypothesis k
            for k in range(256):
                # S-box output: SBOX[plaintext XOR k]
                sbox_out = AES_SBOX[pts_shuffled[i] ^ k]
                cum_log_prob[i, k] = cum_log_prob[i-1, k] + log_probs[i, sbox_out] if i > 0 else log_probs[i, sbox_out]
        
        # Rank the true key at each trace count
        true_key = keys[0]  # All keys are same
        ranks_trial = []
        for i in range(num_traces):
            sorted_keys = np.argsort(-cum_log_prob[i])  # Descending order
            rank = np.where(sorted_keys == true_key)[0][0]
            ranks_trial.append(rank)
        
        all_ranks.append(ranks_trial)
    
    # Compute mean and std across trials
    all_ranks = np.array(all_ranks)  # [num_trials, num_traces]
    mean_ranks = np.mean(all_ranks, axis=0)  # [num_traces]
    std_ranks = np.std(all_ranks, axis=0)
    
    return mean_ranks, std_ranks


def evaluate_model_ge(model, attack_loader, metadata, target_byte, device, 
                      num_trials=100, max_traces=10000):
    """
    Evaluate model using Guessing Entropy (GE) methodology
    """
    model.eval()
    all_preds = []
    
    # Get predictions
    with torch.no_grad():
        for data, _ in attack_loader:
            data = data.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1).cpu().numpy()
            all_preds.append(probs)
    
    predictions = np.vstack(all_preds)[:max_traces]
    plaintexts = metadata['plaintext'][:max_traces, target_byte]
    keys = metadata['key'][:max_traces, target_byte]
    
    # Compute GE (100 trials)
    mean_ranks, std_ranks = compute_ge_key_rank(
        predictions, plaintexts, keys, 
        num_trials=num_trials,
        num_traces=max_traces
    )
    
    return mean_ranks, std_ranks


# =============================================================================
# DATA LOADING (Matching EstraNet)
# =============================================================================

def load_ascad_data(file_path, target_byte=2):
    """Load ASCAD dataset with same preprocessing as EstraNet"""
    print(f"Loading ASCAD from: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        X_train = f['Profiling_traces/traces'][:]
        X_attack = f['Attack_traces/traces'][:]
        
        m_train = f['Profiling_traces/metadata'][:]
        m_attack = f['Attack_traces/metadata'][:]
    
    # Create labels (S-box output)
    y_train = AES_SBOX[m_train['plaintext'][:, target_byte] ^ m_train['key'][:, target_byte]]
    y_attack = AES_SBOX[m_attack['plaintext'][:, target_byte] ^ m_attack['key'][:, target_byte]]
    
    print(f"Training traces:  {X_train.shape}")
    print(f"Attack traces:    {X_attack.shape}")
    print(f"Target byte:      {target_byte}")
    
    # Normalize using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_attack = scaler.transform(X_attack)
    
    return X_train, y_train, X_attack, y_attack, m_attack


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(args):
    """
    Main training loop - configuration matched to EstraNet
    """
    print("="*80)
    print("MAMBA-GNN TRAINING (EstraNet-aligned configuration)")
    print("="*80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    X_train, y_train, X_attack, y_attack, m_attack = load_ascad_data(
        args.data_path, args.target_byte
    )
    
    # Create dataloaders with augmentation for training
    train_dataset = AugmentedASCADDataset(
        X_train, y_train, 
        augment=True,
        noise_std=args.augment_noise,
        shift_max=args.augment_shift
    )
    train_dataset.train()  # Enable augmentation
    
    attack_dataset = ASCADDataset(X_attack, y_attack)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,  # 256 like EstraNet
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    attack_loader = DataLoader(
        attack_dataset,
        batch_size=args.eval_batch_size,  # 32 like EstraNet
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model - MATCHED TO ESTRANET CONFIG
    model = OptimizedMambaGNN(
        trace_length=args.input_length,
        d_model=args.d_model,
        mamba_layers=args.mamba_layers,
        gnn_layers=args.gnn_layers,
        num_classes=256,
        k_neighbors=args.k_neighbors,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    
    # Optimizer - Adam with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay  # L2 regularization
    )
    
    # Learning rate scheduler - COSINE DECAY like EstraNet
    lr_schedule = CosineLRSchedule(
        max_lr=args.learning_rate,
        train_steps=args.train_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    
    # Loss function with label smoothing for regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Training metrics
    num_train_batch = len(train_loader)
    print(f"Training batches per iteration: {num_train_batch}")
    print(f"Total training steps: {args.train_steps}")
    print(f"Save checkpoints every: {args.save_steps} steps")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training statistics
    loss_history = {}
    global_step = 0
    best_eval_loss = float('inf')
    patience_counter = 0
    
    # Restore checkpoint if warm_start
    if args.warm_start:
        ckpt_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
        if os.path.exists(ckpt_path):
            print(f"\nRestoring from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_step = checkpoint['global_step']
            print(f"Resumed from step {global_step}")
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    # Training loop
    model.train()
    running_loss = 0.0
    
    while global_step < args.train_steps:
        for batch_idx, (data, target) in enumerate(train_loader):
            if global_step >= args.train_steps:
                break
            
            data, target = data.to(device), target.to(device)
            
            # Update learning rate
            current_lr = lr_schedule.get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (matching EstraNet)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            optimizer.step()
            
            # Track loss
            running_loss += loss.item()
            global_step += 1
            
            # Log every args.iterations steps (like EstraNet)
            if global_step % args.iterations == 0:
                avg_loss = running_loss / args.iterations
                print(f"[{global_step:6d}] | gnorm {grad_norm:5.2f} lr {current_lr:9.6f} | loss {avg_loss:>5.2f}")
                
                loss_history[global_step] = {
                    'train_loss': avg_loss,
                    'grad_norm': grad_norm.item(),
                    'lr': current_lr
                }
                
                running_loss = 0.0
            
            # Evaluate every eval_steps
            if global_step % args.eval_steps == 0 and global_step > 0:
                model.eval()
                
                # Evaluate training loss
                train_eval_loss = 0.0
                train_eval_batches = min(args.max_eval_batch, num_train_batch)
                with torch.no_grad():
                    for i, (data, target) in enumerate(train_loader):
                        if i >= train_eval_batches:
                            break
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output, target)
                        train_eval_loss += loss.item()
                
                train_eval_loss /= train_eval_batches
                print(f"Train batches[{train_eval_batches:5d}]                | loss {train_eval_loss:>5.2f}")
                
                # Evaluate on attack set
                eval_loss = 0.0
                eval_batches = 0
                with torch.no_grad():
                    for i, (data, target) in enumerate(attack_loader):
                        if args.max_eval_batch > 0 and i >= args.max_eval_batch:
                            break
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output, target)
                        eval_loss += loss.item()
                        eval_batches += 1
                
                eval_loss /= eval_batches
                print(f"Eval  batches[{eval_batches:5d}]                | loss {eval_loss:>5.2f}")
                
                # Ensure history entry exists when eval runs before the regular log interval
                loss_history.setdefault(global_step, {})
                loss_history[global_step].update({
                    'train_eval_loss': train_eval_loss,
                    'eval_loss': eval_loss
                })
                
                # Early stopping check
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                    # Save best model
                    best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'eval_loss': eval_loss
                    }, best_path)
                    print(f"★ New best model saved (eval_loss: {eval_loss:.2f})")
                else:
                    patience_counter += 1
                    if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                        print(f"\n⚠ Early stopping triggered at step {global_step}")
                        print(f"  Best eval loss: {best_eval_loss:.2f}")
                        # Set flag to exit training
                        global_step = args.train_steps
                        break
                
                model.train()
            
            # Save checkpoint every save_steps
            if global_step % args.save_steps == 0 and global_step > 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f'mamba_gnn-{global_step}.pth')
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history
                }, ckpt_path)
                print(f"Model saved: {ckpt_path}")
                
                # Save latest checkpoint
                latest_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history
                }, latest_path)
    
    # Final save
    if global_step % args.save_steps != 0:
        ckpt_path = os.path.join(args.checkpoint_dir, f'mamba_gnn-{global_step}.pth')
        torch.save({
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history
        }, ckpt_path)
        print(f"Final model saved: {ckpt_path}")
    
    # Save loss history
    loss_path = os.path.join(args.checkpoint_dir, 'loss.pkl')
    with open(loss_path, 'wb') as f:
        pickle.dump(loss_history, f)
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate(args):
    """
    Evaluation using Guessing Entropy (100 trials like EstraNet)
    """
    print("="*80)
    print("MAMBA-GNN EVALUATION (Guessing Entropy)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    _, _, X_attack, y_attack, m_attack = load_ascad_data(
        args.data_path, args.target_byte
    )
    
    attack_loader = DataLoader(
        ASCADDataset(X_attack, y_attack),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Load model
    model = OptimizedMambaGNN(
        trace_length=args.input_length,
        d_model=args.d_model,
        mamba_layers=args.mamba_layers,
        gnn_layers=args.gnn_layers,
        num_classes=256,
        k_neighbors=args.k_neighbors,
        dropout=args.dropout
    ).to(device)
    
    # Load checkpoint
    if args.checkpoint_idx > 0:
        ckpt_path = os.path.join(args.checkpoint_dir, f'mamba_gnn-{args.checkpoint_idx}.pth')
    else:
        ckpt_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate using GE (100 trials)
    print("\nComputing Guessing Entropy (100 trials)...")
    mean_ranks, std_ranks = evaluate_model_ge(
        model, attack_loader, m_attack, args.target_byte, device,
        num_trials=100,  # Match EstraNet
        max_traces=len(X_attack)
    )
    
    # Save results (matching EstraNet format)
    result_path = args.result_path + '.txt'
    with open(result_path, 'w') as f:
        # Write mean ranks
        for rank in mean_ranks:
            f.write(f"{rank}\t")
        f.write('\n')
        
        # Write std ranks
        for std in std_ranks:
            f.write(f"{std}\t")
        f.write('\n')
    
    print(f"\nResults saved to: {result_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("GUESSING ENTROPY RESULTS")
    print("="*80)
    print(f"\nTarget byte: {args.target_byte}")
    print(f"True key: 0x{m_attack['key'][0][args.target_byte]:02X}")
    print(f"\nKey Rank (Guessing Entropy):")
    print(f"  100 traces:   {mean_ranks[99]:.2f} ± {std_ranks[99]:.2f}")
    print(f"  500 traces:   {mean_ranks[499]:.2f} ± {std_ranks[499]:.2f}")
    print(f"  1000 traces:  {mean_ranks[999]:.2f} ± {std_ranks[999]:.2f}")
    if len(mean_ranks) >= 5000:
        print(f"  5000 traces:  {mean_ranks[4999]:.2f} ± {std_ranks[4999]:.2f}")
    if len(mean_ranks) >= 10000:
        print(f"  10000 traces: {mean_ranks[9999]:.2f} ± {std_ranks[9999]:.2f}")
    
    # Find first rank=0 (key recovered)
    recovered_idx = np.where(mean_ranks == 0)[0]
    if len(recovered_idx) > 0:
        print(f"\n✓ Key recovered at {recovered_idx[0]+1} traces")
    else:
        print(f"\n✗ Key not recovered (best rank: {mean_ranks[-1]:.2f})")
    
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Mamba-GNN Training (EstraNet-aligned)')
    
    # Data config
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to ASCAD dataset')
    parser.add_argument('--target_byte', type=int, default=2,
                       help='Target byte for key recovery')
    parser.add_argument('--input_length', type=int, default=700,
                       help='Input trace length')
    
    # Training config (MATCHED TO ESTRANET)
    parser.add_argument('--do_train', action='store_true',
                       help='Perform training')
    parser.add_argument('--train_batch_size', type=int, default=256,
                       help='Training batch size (EstraNet: 256)')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                       help='Evaluation batch size (EstraNet: 32)')
    parser.add_argument('--train_steps', type=int, default=100000,
                       help='Total training steps (EstraNet: 100000)')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Log every N steps (EstraNet: 500)')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--save_steps', type=int, default=10000,
                       help='Save checkpoint every N steps (EstraNet: 10000)')
    parser.add_argument('--max_eval_batch', type=int, default=312,
                       help='Max batches for evaluation')
    
    # Optimizer config (MATCHED TO ESTRANET)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                       help='Learning rate (EstraNet: 2.5e-4)')
    parser.add_argument('--clip', type=float, default=0.25,
                       help='Gradient clipping (EstraNet: 0.25)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.004,
                       help='Minimum LR ratio (EstraNet: 0.004)')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Warmup steps (EstraNet: 0-1000)')
    
    # Model config
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension (EstraNet: 128)')
    parser.add_argument('--mamba_layers', type=int, default=4,
                       help='Number of Mamba layers')
    parser.add_argument('--gnn_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--k_neighbors', type=int, default=8,
                       help='K-neighbors for graph')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (EstraNet: 0.1)')
    
    # Regularization config (NEW - to prevent overfitting)
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for L2 regularization')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--early_stopping', type=int, default=10,
                       help='Early stopping patience (eval periods). 0=disabled')
    parser.add_argument('--augment_noise', type=float, default=0.1,
                       help='Gaussian noise std for augmentation')
    parser.add_argument('--augment_shift', type=int, default=5,
                       help='Max random time shift for augmentation')
    
    # Checkpoint config
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Checkpoint directory')
    parser.add_argument('--checkpoint_idx', type=int, default=0,
                       help='Checkpoint index for evaluation (0=latest)')
    parser.add_argument('--warm_start', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--result_path', type=str, default='results/mamba_gnn',
                       help='Path for evaluation results')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION (EstraNet-aligned)")
    print("="*80)
    print(f"data_path:         {args.data_path}")
    print(f"target_byte:       {args.target_byte}")
    print(f"input_length:      {args.input_length}")
    print(f"train_batch_size:  {args.train_batch_size}")
    print(f"eval_batch_size:   {args.eval_batch_size}")
    print(f"train_steps:       {args.train_steps}")
    print(f"learning_rate:     {args.learning_rate}")
    print(f"d_model:           {args.d_model}")
    print(f"mamba_layers:      {args.mamba_layers}")
    print(f"gnn_layers:        {args.gnn_layers}")
    print(f"checkpoint_dir:    {args.checkpoint_dir}")
    print("="*80 + "\n")
    
    if args.do_train:
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
