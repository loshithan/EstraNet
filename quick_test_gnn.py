"""
Quick test for GNN model - checks it builds and runs
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

import tensorflow as tf
from gnn_estranet import GNNEstraNet

print("Testing GNN Model...")

model = GNNEstraNet(n_classes=256)
x = tf.zeros((2, 10000))
y = model(x)

params = model.count_params()
print(f"Model built successfully!")
print(f"Parameters: {params:,}")
print(f"Transformer: 431,233")
print(f"Mamba: 425,569")
print(f"GNN: {params:,}")

if params < 431233:
    reduction = ((431233 - params) / 431233) * 100
    print(f"GNN has {reduction:.1f}% fewer params")
