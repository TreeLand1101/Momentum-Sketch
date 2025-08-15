import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) != 3:
    print("Usage: python PlotHeatmap.py <csv_file> <output_dir>")
    sys.exit(1)

csv_file = sys.argv[1]
output_dir = sys.argv[2]

LABEL_SIZE = 14
TICK_SIZE = 12
Y_ROTATION = 0   
LABELPAD = 28  

df = pd.read_csv(csv_file)
df["Memory Ratio"] = df["Filter Ratio"].astype(str) + ":" + df["Sketch Ratio"].astype(str)

metrics = [
    ("F1 Score", "YlOrRd", "heatmap_f1.png", "F1 Score Heatmap"),
    ("Precision", "YlOrRd", "heatmap_precision.png", "Precision Heatmap"),
    ("Recall", "YlOrRd", "heatmap_recall.png", "Recall Heatmap"),
    ("Insert Throughput (Mops)", "YlGnBu", "heatmap_insert.png", "Insert Throughput Heatmap"),
    ("Query Throughput (Mops)", "YlGnBu", "heatmap_query.png", "Query Throughput Heatmap"),
]

for metric, cmap, filename, title in metrics:
    pivot_table = df.pivot(index="Memory Ratio", columns="Admission Ratio", values=metric)
    
    plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        annot_kws={"size": TICK_SIZE}
    )
    
    ax.set_xlabel("α", fontsize=LABEL_SIZE)
    ax.set_ylabel("γ:1-γ", fontsize=LABEL_SIZE, rotation=Y_ROTATION, va='center') 
    ax.yaxis.labelpad = LABELPAD
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    
    plt.setp(ax.get_yticklabels(), rotation=Y_ROTATION, ha='right')

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")
