# Quick Start Guide - EstraNet Training

## ✅ What's Ready

I've created a **complete Jupyter notebook** that you can run in Google Colab or locally to train the EstraNet model on the ASCADf dataset.

**Notebook**: [`EstraNet_ASCADf_Training.ipynb`](file:///c:/Users/User/Desktop/estranet%20transformer/EstraNet/EstraNet_ASCADf_Training.ipynb)

---

## 🚀 How to Get Started

### Option 1: Google Colab (Recommended - Free GPU!)

1. **Go to Google Colab**: https://colab.research.google.com/

2. **Upload the notebook**:
   - Click `File` → `Upload notebook`
   - Select `EstraNet_ASCADf_Training.ipynb` from your computer

3. **Enable GPU** (IMPORTANT for fast training):
   - Click `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Click `Save`

4. **Run the notebook**:
   - Click the play button (▶) on each cell from top to bottom
   - Or use `Shift + Enter` to run each cell
   - The notebook will automatically:
     - Clone the repository
     - Install dependencies (including `gdown`)
     - Download the ASCADf dataset from Google Drive (~1.5 GB)
     - Configure training
     - Train the model
     - Evaluate results

### Option 2: Local Jupyter

1. **Open terminal in the EstraNet directory**:

   ```bash
   cd "c:\Users\User\Desktop\estranet transformer\EstraNet"
   ```

2. **Install Jupyter** (if not already installed):

   ```bash
   pip install jupyter
   ```

3. **Launch Jupyter**:

   ```bash
   jupyter notebook
   ```

4. **Open and run the notebook**:
   - Click on `EstraNet_ASCADf_Training.ipynb`
   - Run cells sequentially

---

## 📋 What the Notebook Does

### Automatic Setup

- ✅ Detects if running on Colab or locally
- ✅ Checks for GPU availability
- ✅ Clones repository (Colab only)
- ✅ Installs all dependencies

### Dataset Download (Using gdown)

- ✅ Downloads ASCADf dataset from Google Drive
- ✅ File ID: `1WNajWT0qFbpqPJiuePS_HeXxsCvUHI5M`
- ✅ Shows download progress
- ✅ Verifies dataset integrity

### Training

- ✅ Configurable hyperparameters
- ✅ Default: 4M training steps
- ✅ Saves checkpoints every 40K steps
- ✅ Full model architecture setup

### Evaluation

- ✅ Computes guessing entropy
- ✅ Saves results to `results/` directory

---

## ⚙️ Configuration

The notebook includes a configuration cell where you can adjust parameters:

```python
config = {
    'input_length': 10000,      # Trace length
    'train_batch_size': 16,     # Batch size
    'train_steps': 4000000,     # Total steps
    'learning_rate': 2.5e-4,    # Learning rate
    # ... and more
}
```

### For Quick Testing

If you want to test the setup without waiting hours, modify these values:

```python
config['train_steps'] = 100000   # ~2.5% of full training
config['warmup_steps'] = 10000   # Reduced warmup
config['save_steps'] = 10000     # Save more frequently
```

---

## 💡 Important Tips

### GPU is Essential

> **Without GPU**: Training 4M steps could take 100+ hours  
> **With GPU**: Training takes 8-12 hours

Always enable GPU in Colab!

### Dataset Download

The dataset is downloaded automatically using `gdown` from Google Drive:

- Size: ~1.5 GB
- Time: 2-10 minutes depending on connection
- Stored in: `data/ASCAD.h5`

### Memory Issues?

If you get out-of-memory errors:

- Reduce `train_batch_size` to 8 or 4
- Reduce `input_length` to 5000
- Reduce `d_model` to 64

---

## 📚 Next Steps

1. Upload the notebook to Google Colab
2. Enable GPU runtime
3. Run all cells in order
4. Monitor training progress
5. Check results after training completes

That's it! The notebook handles everything else automatically.
