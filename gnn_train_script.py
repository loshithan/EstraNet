# ============================================================================
# TRAIN UPGRADED "ATTENTION GNN" MODEL IN COLAB (FIXED)
# ============================================================================
# Paste this into a new Colab cell

print("🔷 Training Upgraded GNN Model (v2: Attention + Deeper + 50k Steps)")
print("="*70)

# Configuration
CONFIG = {
    'checkpoint_dir': '/content/drive/MyDrive/EstraNet/checkpoints_gnn_attention_v2',
    'result_path': 'results/gnn_attention',
    'train_steps': 50000,   # 50k steps
    'save_steps': 2000,
    'train_batch_size': 256,
    'eval_batch_size': 32,
    'learning_rate': 0.0002,
    'model_type': 'gnn',
    'n_gcn_layers': 4,
    'k_neighbors': 15,
    'graph_pooling': 'attention'
}

import os
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
os.makedirs(CONFIG['result_path'], exist_ok=True)

# Build training command (NO COMMENTS to avoid Bash errors)
# Explicitly use warm_start=False to force fresh training
train_cmd = f"""
python train_trans.py \\
    --data_path=data/ASCAD.h5 \\
    --checkpoint_dir={CONFIG['checkpoint_dir']} \\
    --model_type={CONFIG['model_type']} \\
    --dataset=ASCAD \\
    --input_length=700 \\
    --d_model=128 \\
    --n_gcn_layers={CONFIG['n_gcn_layers']} \\
    --k_neighbors={CONFIG['k_neighbors']} \\
    --graph_pooling={CONFIG['graph_pooling']} \\
    --conv_kernel_size=3 \\
    --n_conv_layer=2 \\
    --pool_size=2 \\
    --dropout=0.1 \\
    --do_train=True \\
    --warm_start=False \\
    --learning_rate={CONFIG['learning_rate']} \\
    --clip=0.25 \\
    --min_lr_ratio=0.004 \\
    --warmup_steps=1000 \\
    --train_batch_size={CONFIG['train_batch_size']} \\
    --eval_batch_size={CONFIG['eval_batch_size']} \\
    --train_steps={CONFIG['train_steps']} \\
    --iterations=500 \\
    --save_steps={CONFIG['save_steps']} \\
    --result_path={CONFIG['result_path']}
"""

print("Starting GNN training...")
print(f"Model: GNN-Attention (Est. 350k parameters)")
print(f"Features: {CONFIG['n_gcn_layers']} Layers, {CONFIG['k_neighbors']} Neighbors, Attention Pooling")
print(f"Checkpoints: {CONFIG['checkpoint_dir']}")
print(f"Training steps: {CONFIG['train_steps']:,}\n")

os.system(train_cmd)
