
# ============================================================================
# PLOT EFFICIENCY (Traces to Recover Key)
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import numpy as np

def plot_efficiency():
    # Find all result files (Support both 'attention' and 'mean' pooling patterns)
    files = glob.glob("evaluation_results_*_*.csv")
    if not files:
        print("❌ No evaluation results found!")
        return

    print(f"found {len(files)} files: {files}")
    
    all_data = []

    for f in files:
        try:
            # Parse start_index from filename (e.g., evaluation_results_attention_2000.csv)
            parts = f.replace(".csv", "").split("_")
            start_index = parts[-1] 
            
            df = pd.read_csv(f)
            
            # Filter for successful attacks (BrokenAt is a number, not >2000)
            df['BrokenAtNumeric'] = pd.to_numeric(df['BrokenAt'], errors='coerce')
            
            # Add metadata
            df['StartIndex'] = start_index
            
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

    if not all_data:
        print("❌ No valid data found!")
        return

    full_df = pd.concat(all_data)
    
    # FILTER INVALID MODELS
    # Drop rows where SizeMB is effectively 0 (corrupted/empty)
    if 'SizeMB' in full_df.columns:
        full_df['SizeMB'] = pd.to_numeric(full_df['SizeMB'], errors='coerce')
        full_df = full_df[full_df['SizeMB'] > 0.1]
    
    # AGGREGATE DUPLICATES (Crucial Step: Ensure Unique (Name, StartIndex) pairs)
    # Group by Name and StartIndex, then take the MEAN of BrokenAtNumeric
    agg_df = full_df.groupby(['Name', 'StartIndex'], as_index=False)['BrokenAtNumeric'].mean()
    
    # Filter only models that broke at least once (i.e. BrokenAtNumeric is not NaN)
    plot_df = agg_df[agg_df['BrokenAtNumeric'].notna()].copy()
    
    if plot_df.empty:
        print("❌ No models successfully broke the key in any test!")
        return

    # Sort logic: Best model has lowest MEAN traces to break across all experiments
    mean_performance = plot_df.groupby('Name')['BrokenAtNumeric'].mean().sort_values()
    sorted_models = mean_performance.index.tolist()
    
    print(f"Plots will include {len(sorted_models)} models.")

    # ---------------------------------------------------------
    # PLOT 1: Heatmap (Models vs Dataset) - Traces to Break
    # ---------------------------------------------------------
    try:
        pivot_table = plot_df.pivot(index='Name', columns='StartIndex', values='BrokenAtNumeric')
        
        # Sort Columns Numerically
        cols = sorted(pivot_table.columns, key=lambda x: int(x) if x.isdigit() else 99999)
        pivot_table = pivot_table[cols]
        
        # Sort Rows by Performance
        pivot_table = pivot_table.reindex(sorted_models)
        
        plt.figure(figsize=(12, len(sorted_models) * 0.5 + 2))
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="RdYlGn_r", linewidths=.5)
        plt.title("Traces to Break Key (Lower is Better)")
        plt.tight_layout()
        plt.savefig("efficiency_heatmap.png")
        print("✅ Saved efficiency_heatmap.png")
    except Exception as e:
        print(f"❌ Heatmap Error: {e}")

    # ---------------------------------------------------------
    # PLOT 2: Bar Chart (Average Efficiency)
    # ---------------------------------------------------------
    try:
        plt.figure(figsize=(12, 6))
        # Use aggregation to avoid duplicate label error in older seaborn/pandas versions
        sns.barplot(x='Name', y='BrokenAtNumeric', data=plot_df, order=sorted_models, errorbar=None)
        plt.xticks(rotation=45, ha='right')
        plt.title("Average Traces to Recovry (Cross-Validation)")
        plt.ylabel("Traces")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("efficiency_barplot.png")
        print("✅ Saved efficiency_barplot.png")
    except Exception as e:
        print(f"❌ Barplot Error: {e}")
    
    # ---------------------------------------------------------
    # TEXT SUMMARY
    # ---------------------------------------------------------
    print("\n---------------------------------------------------")
    print("🏅 TOP PERFORMING MODELS (Average Traces)")
    print("---------------------------------------------------")
    print(mean_performance.head(5))

if __name__ == "__main__":
    plot_efficiency()
