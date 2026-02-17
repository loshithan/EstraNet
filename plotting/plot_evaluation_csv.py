
import pandas as pd
import matplotlib.pyplot as plt
import os

csv_file = "transformer_evaluation_results.csv"

if not os.path.exists(csv_file):
    print("CSV file not found!")
    exit()

try:
    df = pd.read_csv(csv_file)
    
    # Filter out weird rows (like trans_long-98 if it's an outlier or sort properly)
    # Extract number from checkpoint name
    df['CheckpointNum'] = df['Checkpoint'].apply(lambda x: int(x.split('-')[-1]) if '-' in x else -1)
    df = df.sort_values('CheckpointNum')
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['CheckpointNum'], df['Final_Rank'], marker='o', linestyle='-')
    plt.title('Transformer Checkpoint Evaluation (Progress)')
    plt.xlabel('Checkpoint Number (Epoch)')
    plt.ylabel('Final Mean Rank (Lower is Better)')
    plt.grid(True)
    
    # Annotate best point
    if not df.empty:
        best_row = df.loc[df['Final_Rank'].idxmin()]
        plt.annotate(f"Best: {best_row['Checkpoint']} (Rank {best_row['Final_Rank']})",
                     xy=(best_row['CheckpointNum'], best_row['Final_Rank']),
                     xytext=(best_row['CheckpointNum'], best_row['Final_Rank'] + 20),
                     arrowprops=dict(facecolor='green', shrink=0.05))

    plot_file = "current_progress.png"
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    
except Exception as e:
    print(f"Error plotting: {e}")
