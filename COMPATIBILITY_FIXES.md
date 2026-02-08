# TensorFlow Compatibility Fixes

## Issues Fixed

### 1. SyncBatchNormalization Import Error

**Error**: `ModuleNotFoundError: No module named 'tensorflow.keras.layers.experimental'`

**Root Cause**: In TensorFlow 2.13.0, `SyncBatchNormalization` was moved from `tensorflow.keras.layers.experimental` to `tensorflow.keras.layers`.

**Fix Applied** in [`transformer.py`](file:///c:/Users/User/Desktop/estranet%20transformer/EstraNet/transformer.py#L4-L9):

```python
# SyncBatchNormalization import - compatible with TensorFlow 2.13
try:
    from tensorflow.keras.layers import SyncBatchNormalization
except ImportError:
    # Fallback for older TensorFlow versions
    from tensorflow.keras.layers.experimental import SyncBatchNormalization
```

### 2. Integer Comparison Syntax Warning

**Warning**: `SyntaxWarning: "is" with 'int' literal. Did you mean "=="`

**Root Cause**: Line 244 used `is` operator for integer comparison, which is incorrect for value comparison.

**Fix Applied** in [`transformer.py`](file:///c:/Users/User/Desktop/estranet%20transformer/EstraNet/transformer.py#L244):

```python
# Before:
ks = 11 if l is 0 else self.conv_kernel_size

# After:
ks = 11 if l == 0 else self.conv_kernel_size
```

---

## Verification

You can now retry training. The errors should be resolved. Run the training cell again in your Jupyter notebook.

If you encounter any other errors, please share them and I'll help fix them!
