"""
TensorFlow Training Script for Mamba-GNN (EstraNet-aligned)

This script trains the Mamba-GNN model using TensorFlow/Keras, enabling:
- Fair comparison with EstraNet models
- Easy conversion to TFLite for deployment
- Compatible with EstraNet evaluation pipeline

Configuration matches train_trans.py for consistency.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import pickle
import numpy as np
import h5py
from tqdm import tqdm

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mamba_gnn_model_tf import OptimizedMambaGNNTF

# Disable GPU memory growth issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# AES S-box
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
# LEARNING RATE SCHEDULE (Cosine Decay)
# =============================================================================

class CosineLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay with warmup - matches EstraNet's LRSchedule"""
    
    def __init__(self, max_lr, train_steps, warmup_steps=0, min_lr_ratio=0.004):
        super().__init__()
        self.max_lr = max_lr
        self.train_steps = train_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        train_steps = tf.cast(self.train_steps, tf.float32)
        
        # Warmup
        warmup_lr = (step / warmup_steps) * self.max_lr
        
        # Cosine decay
        progress = (step - warmup_steps) / (train_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * progress))
        decayed = (1.0 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio
        decay_lr = self.max_lr * decayed
        
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            'max_lr': self.max_lr,
            'train_steps': self.train_steps,
            'warmup_steps': self.warmup_steps,
            'min_lr_ratio': self.min_lr_ratio,
        }


# =============================================================================
# GUESSING ENTROPY EVALUATION
# =============================================================================

def compute_ge_key_rank(predictions, plaintexts, keys, num_trials=100, num_traces=None):
    """
    Compute Guessing Entropy (average key rank over multiple shuffled trials)
    Matches EstraNet's evaluation methodology
    """
    n_samples = len(predictions)
    if num_traces is None:
        num_traces = n_samples
    
    num_traces = min(num_traces, n_samples)
    
    all_ranks = []
    for trial in range(num_trials):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        preds_shuffled = predictions[indices]
        pts_shuffled = plaintexts[indices]
        
        # Compute cumulative log probabilities
        log_probs = np.log(preds_shuffled[:num_traces] + 1e-40)
        cum_log_prob = np.zeros((num_traces, 256))
        
        for i in range(num_traces):
            for k in range(256):
                sbox_out = AES_SBOX[pts_shuffled[i] ^ k]
                cum_log_prob[i, k] = cum_log_prob[i-1, k] + log_probs[i, sbox_out] if i > 0 else log_probs[i, sbox_out]
        
        # Rank the true key
        true_key = keys[0]
        ranks_trial = []
        for i in range(num_traces):
            sorted_keys = np.argsort(-cum_log_prob[i])
            rank = np.where(sorted_keys == true_key)[0][0]
            ranks_trial.append(rank)
        
        all_ranks.append(ranks_trial)
    
    all_ranks = np.array(all_ranks)
    mean_ranks = np.mean(all_ranks, axis=0)
    std_ranks = np.std(all_ranks, axis=0)
    
    return mean_ranks, std_ranks


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ascad_data(file_path, target_byte=2):
    """Load ASCAD dataset with same preprocessing as EstraNet"""
    print(f"Loading ASCAD from: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        X_train = f['Profiling_traces/traces'][:]
        X_attack = f['Attack_traces/traces'][:]
        
        m_train = f['Profiling_traces/metadata'][:]
        m_attack = f['Attack_traces/metadata'][:]
    
    # Create labels
    y_train = AES_SBOX[m_train['plaintext'][:, target_byte] ^ m_train['key'][:, target_byte]]
    y_attack = AES_SBOX[m_attack['plaintext'][:, target_byte] ^ m_attack['key'][:, target_byte]]
    
    print(f"Training traces:  {X_train.shape}")
    print(f"Attack traces:    {X_attack.shape}")
    print(f"Target byte:      {target_byte}")
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_attack = scaler.transform(X_attack)
    
    return X_train, y_train, X_attack, y_attack, m_attack


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(data_path, checkpoint_dir, config):
    """
    Main training loop - TensorFlow version
    """
    print("="*80)
    print("MAMBA-GNN TRAINING (TensorFlow / EstraNet-aligned)")
    print("="*80)
    
    # Load data
    X_train, y_train, X_attack, y_attack, m_attack = load_ascad_data(
        data_path, config['target_byte']
    )
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(config['train_batch_size']).prefetch(tf.data.AUTOTUNE)
    
    attack_dataset = tf.data.Dataset.from_tensor_slices((X_attack, y_attack))
    attack_dataset = attack_dataset.batch(config['eval_batch_size']).prefetch(tf.data.AUTOTUNE)
    
    # Create model
    model = OptimizedMambaGNNTF(
        trace_length=config['input_length'],
        d_model=config['d_model'],
        mamba_layers=config['mamba_layers'],
        gnn_layers=config['gnn_layers'],
        num_classes=256,
        k_neighbors=config['k_neighbors'],
        dropout=config['dropout']
    )
    
    # Build model
    model.build(input_shape=(None, config['input_length']))
    total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nModel Parameters: {total_params:,}")
    
    # Learning rate schedule
    lr_schedule = CosineLRSchedule(
        max_lr=config['learning_rate'],
        train_steps=config['train_steps'],
        warmup_steps=config['warmup_steps'],
        min_lr_ratio=config['min_lr_ratio']
    )
    
    # Optimizer - Adam (not AdamW) to match EstraNet
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Loss function - Cross Entropy
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    # Checkpoint manager
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=10)
    
    # Restore if exists
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"Restored from {manager.latest_checkpoint}")
    
    # Training metrics
    loss_history = {}
    global_step = 0
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    # Training loop
    num_epochs = config['train_steps'] // (len(X_train) // config['train_batch_size']) + 1
    
    for epoch in range(num_epochs):
        if global_step >= config['train_steps']:
            break
        
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            if global_step >= config['train_steps']:
                break
            
            # Training step
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)
            
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_weights)
            
            # Clip gradients
            gradients, grad_norm = tf.clip_by_global_norm(gradients, config['clip'])
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            
            train_loss(loss)
            global_step += 1
            
            # Log every iterations steps
            if global_step % config['iterations'] == 0:
                current_lr = lr_schedule(global_step).numpy()
                avg_loss = train_loss.result().numpy()
                print(f"[{global_step:6d}] | gnorm {grad_norm.numpy():5.2f} lr {current_lr:9.6f} | loss {avg_loss:>5.2f}")
                
                loss_history[global_step] = {
                    'train_loss': float(avg_loss),
                    'grad_norm': float(grad_norm.numpy()),
                    'lr': float(current_lr)
                }
                
                train_loss.reset_states()
            
            # Save checkpoint
            if global_step % config['save_steps'] == 0 and global_step > 0:
                save_path = manager.save()
                print(f"Model saved: {save_path}")
    
    # Final save
    save_path = manager.save()
    print(f"\nFinal model saved: {save_path}")
    
    # Save loss history
    loss_path = os.path.join(checkpoint_dir, 'loss.pkl')
    with open(loss_path, 'wb') as f:
        pickle.dump(loss_history, f)
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    
    return model


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate(model, data_path, target_byte, max_traces=10000):
    """
    Evaluation using Guessing Entropy (100 trials)
    """
    print("="*80)
    print("MAMBA-GNN EVALUATION (Guessing Entropy)")
    print("="*80)
    
    # Load data
    _, _, X_attack, y_attack, m_attack = load_ascad_data(data_path, target_byte)
    
    # Get predictions
    attack_dataset = tf.data.Dataset.from_tensor_slices(X_attack[:max_traces])
    attack_dataset = attack_dataset.batch(32)
    
    all_preds = []
    for x_batch in attack_dataset:
        logits = model(x_batch, training=False)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        all_preds.append(probs)
    
    predictions = np.vstack(all_preds)
    plaintexts = m_attack['plaintext'][:max_traces, target_byte]
    keys = m_attack['key'][:max_traces, target_byte]
    
    # Compute GE
    print("\nComputing Guessing Entropy (100 trials)...")
    mean_ranks, std_ranks = compute_ge_key_rank(
        predictions, plaintexts, keys,
        num_trials=100,
        num_traces=max_traces
    )
    
    # Print summary
    print("\n" + "="*80)
    print("GUESSING ENTROPY RESULTS")
    print("="*80)
    print(f"\nTarget byte: {target_byte}")
    print(f"True key: 0x{keys[0]:02X}")
    print(f"\nKey Rank (Guessing Entropy):")
    print(f"  100 traces:   {mean_ranks[99]:.2f} ± {std_ranks[99]:.2f}")
    print(f"  500 traces:   {mean_ranks[499]:.2f} ± {std_ranks[499]:.2f}")
    print(f"  1000 traces:  {mean_ranks[999]:.2f} ± {std_ranks[999]:.2f}")
    
    recovered_idx = np.where(mean_ranks == 0)[0]
    if len(recovered_idx) > 0:
        print(f"\n✓ Key recovered at {recovered_idx[0]+1} traces")
    else:
        print(f"\n✗ Key not recovered (best rank: {mean_ranks[-1]:.2f})")
    
    print("="*80)
    
    return mean_ranks, std_ranks


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Mamba-GNN Training (TensorFlow/EstraNet-aligned)')
    
    # Data config
    parser.add_argument('--data_path', type=str, default='data/ASCAD.h5',
                       help='Path to ASCAD dataset')
    parser.add_argument('--target_byte', type=int, default=2,
                       help='Target byte for key recovery')
    parser.add_argument('--input_length', type=int, default=700,
                       help='Input trace length')
    
    # Training config
    parser.add_argument('--do_train', action='store_true',
                       help='Perform training')
    parser.add_argument('--train_batch_size', type=int, default=256,
                       help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                       help='Evaluation batch size')
    parser.add_argument('--train_steps', type=int, default=100000,
                       help='Total training steps')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Log every N steps')
    parser.add_argument('--save_steps', type=int, default=10000,
                       help='Save checkpoint every N steps')
    
    # Optimizer config
    parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                       help='Learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                       help='Gradient clipping')
    parser.add_argument('--min_lr_ratio', type=float, default=0.004,
                       help='Minimum LR ratio')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Warmup steps')
    
    # Model config
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--mamba_layers', type=int, default=4,
                       help='Number of Mamba layers')
    parser.add_argument('--gnn_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--k_neighbors', type=int, default=8,
                       help='K-neighbors for graph')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Checkpoint config
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/mamba_gnn_tf',
                       help='Checkpoint directory')
    parser.add_argument('--result_path', type=str, default='results/mamba_gnn_tf',
                       help='Path for evaluation results')
    
    args = parser.parse_args()
    
    config = vars(args)
    
    print("\n" + "="*80)
    print("CONFIGURATION (TensorFlow / EstraNet-aligned)")
    print("="*80)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("="*80 + "\n")
    
    if args.do_train:
        model = train(args.data_path, args.checkpoint_dir, config)
        
        # Evaluate after training
        mean_ranks, std_ranks = evaluate(
            model, args.data_path, args.target_byte
        )
        
        # Save results
        result_file = args.result_path + '.txt'
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w') as f:
            for rank in mean_ranks:
                f.write(f"{rank}\t")
            f.write('\n')
            for std in std_ranks:
                f.write(f"{std}\t")
            f.write('\n')
        
        print(f"\nResults saved to: {result_file}")
        
        # Export to TFLite
        print("\nExporting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(args.checkpoint_dir, 'mamba_gnn.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ TFLite model saved: {tflite_path}")
        print(f"  Size: {len(tflite_model) / 1024:.2f} KB")
