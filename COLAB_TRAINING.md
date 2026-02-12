# EstraNet Training on Google Colab - Quick Start

## 🚀 One-Line Setup

Open this notebook in Colab:

```
train_estranet_colab.ipynb
```

Or use this link (after pushing to GitHub):

```
https://colab.research.google.com/github/YOUR_USERNAME/EstraNet/blob/main/train_estranet_colab.ipynb
```

## 📋 What You Need

1. **ASCAD.h5 dataset** (44MB) - Download from [ANSSI](https://www.data.gouv.fr/fr/datasets/ascad/)
2. **Google Colab account** (free tier works!)
3. **This repository**

## 🎯 Quick Training

### Method 1: Using the Colab Notebook (Recommended)

1. Open `train_estranet_colab.ipynb` in Google Colab
2. Run all cells in order
3. Wait ~2-4 hours for training to complete

### Method 2: Using Shell Script in Colab

```python
# In a Colab cell:
!git clone https://github.com/YOUR_USERNAME/EstraNet.git
%cd EstraNet

# Upload ASCAD.h5 to data/ folder
!mkdir -p data
# (Upload file via Colab UI)

# Run training
!bash run_trans_ascadf.sh train
```

### Method 3: Direct Python Command

```python
!python train_trans.py \
    --data_path=data/ASCAD.h5 \
    --checkpoint_dir=./checkpoints \
    --dataset=ASCAD \
    --input_length=10000 \
    --train_batch_size=32 \
    --train_steps=50000 \
    --do_train=True
```

## ⚙️ Configuration

**Default settings in `run_trans_ascadf.sh`:**

- Input length: 10,000 points
- Batch size: 16
- Training steps: 4,000,000 (⚠️ VERY LONG - consider reducing to 50,000 for Colab)
- Model: 2 layers, 128 dimensions

**Recommended for Colab:**

- Training steps: 50,000 (2-4 hours)
- Batch size: 32
- Save to Google Drive for persistence

## 📊 Expected Results

- **Training time**: 2-4 hours on Colab GPU (T4)
- **Key rank**: Should achieve rank 0 with <1000 traces
- **Model size**: ~425K parameters

## 💾 Files Created

After training:

```
checkpoints/
├── trans_long-1      # Checkpoint at step 5000
├── trans_long-2      # Checkpoint at step 10000
└── loss.pkl          # Training history

results/
└── eval_results.txt  # Key rank results
```

## 🔧 Troubleshooting

**Out of Memory?**

- Reduce `train_batch_size` to 16 or 8
- Reduce `input_length` to 700

**Training too slow?**

- Enable GPU: Runtime → Change runtime type → GPU
- Reduce `train_steps` to 50000

**Session timeout?**

- Save checkpoints to Google Drive
- Resume with `--warm_start=True`

## 📚 Next Steps

1. ✅ Train Transformer model (this guide)
2. 🔄 Train Mamba model (use `mamba_transformer.py`)
3. 📈 Compare results
4. 🎯 Deploy best model

## 🆘 Need Help?

See the full notebook `train_estranet_colab.ipynb` for:

- Detailed explanations
- Monitoring tools
- Visualization code
- Advanced configurations
