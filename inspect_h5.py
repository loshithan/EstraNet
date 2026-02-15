import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = "data/ATMega8515_raw_traces.h5"

print(f"🔍 Analyzing HDF5 file: {file_path}")
print("="*60)

try:
    with h5py.File(file_path, 'r') as f:
        traces = f['traces']
        metadata = f['metadata']
        
        # 1. Check for Masking (Correlation Check)
        # If masked, Sbox(pt ^ key) should NOT correlate with power consumption directly.
        # We'll check the 'masks' field existence first.
        masks = metadata['masks']
        print(f"🎭 Masks field found: {masks.shape}")
        
        # Check if masks are all zeros (unmasked) or random
        mask_samples = masks[:1000]
        if np.all(mask_samples == 0):
            print("   ⚠️ Masks appear to be all ZEROS. Likely UNMASKED.")
        else:
            print("   ✅ Masks contain non-zero values. Likely MASKED.")
            print(f"      Mask Sample: {mask_samples[0]}")

        # 2. Check for Desynchronization (Visual Inspection)
        # Plot mean vs variance
        # High variance peaks with sharp mean peaks usually implies alignment.
        # Blurred mean peaks implies desync.
        
        n_traces = 100
        trace_samples = traces[:n_traces]
        
        plt.figure(figsize=(12, 6))
        plt.plot(trace_samples.T, color='grey', alpha=0.1)
        plt.plot(np.mean(trace_samples, axis=0), color='red', label='Mean Trace', linewidth=2)
        plt.title(f"Visual Inspection of {n_traces} Traces (Check for Desync)")
        plt.legend()
        plt.savefig("trace_inspection.png")
        print("\n📈 Saved trace plot to 'trace_inspection.png'. Check for alignment.")

        # Metric: distinct peaks in variance?
        var_trace = np.var(trace_samples, axis=0)
        plt.figure(figsize=(12, 6))
        plt.plot(var_trace, color='blue', label='Variance Trace')
        plt.title(f"Variance of {n_traces} Traces")
        plt.legend()
        plt.savefig("trace_variance.png")
        print("📈 Saved variance plot to 'trace_variance.png'.")

except Exception as e:
    print(f"❌ Error analysis: {e}")
