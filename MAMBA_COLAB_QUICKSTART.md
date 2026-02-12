# Mamba Training Quick Start for Colab

## ✅ What's Done

1. **Modified `train_trans.py`** - Added Mamba support via `--model_type=mamba` flag
2. **Created `train_mamba_colab.sh`** - Shell script with Mamba training config
3. **Created `MAMBA_TRAINING_CELL.txt`** - Ready-to-paste Colab cell

## 🚀 How to Train Mamba in Colab

### Step 1: Add the Training Cell

1. Open `train_estranet_colab.ipynb` in Colab
2. Add a **new code cell** after the Transformer training section
3. **Copy the entire contents** of `MAMBA_TRAINING_CELL.txt`
4. **Paste** into the new cell

### Step 2: Run Training

Simply run the cell! It will:

- Create checkpoint directory in Google Drive
- Train Mamba model for 50,000 steps
- Save checkpoints every 5,000 steps
- Use same hyperparameters as Transformer training

### Step 3: Evaluate

After training completes, modify the evaluation cell:

```python
checkpoint_path = './checkpoints_mamba/mamba_model-11'  # Point to Mamba checkpoint
```

## 📊 Comparison: Transformer vs Mamba

Both models are trained with identical parameters:

- Input length: 10,000
- Layers: 2
- Model dimension: 128
- Training steps: 50,000
- Batch size: 256

**Key Difference:** Mamba uses selective state-space mechanism instead of attention, which may offer better efficiency on long sequences.

## 🔍 Files Modified

- `train_trans.py` - Added `--model_type` flag and MambaNet support
- `train_mamba_colab.sh` - New training script for Mamba
- `MAMBA_TRAINING_CELL.txt` - Colab cell code

## ⚡ Next Steps

1. Train the Mamba model in Colab
2. Compare key ranks between Transformer and Mamba
3. Analyze training time and memory usage differences
