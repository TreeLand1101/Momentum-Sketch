#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import sys
import os
from matplotlib.ticker import MaxNLocator

LABEL_SIZE  = 34
TICK_SIZE   = 34
LEGEND_SIZE = 16
MARKER_SIZE = 16

# Define a vivid colormap
VIVID_CM = colormaps['tab10']  # You can switch to 'tab20' or another for more colors

# Preferred legend order (put your important methods here so they appear first)
PREFERRED_ORDER = ["MomentumSketch", "TwoStage"]

def _reorder_handles_labels(handles, labels, preferred=PREFERRED_ORDER):
    """Reorder handles/labels so that any label containing a preferred substring
    (case-insensitive) appears first, preserving the original relative order for the rest."""
    ordered_handles = []
    ordered_labels = []
    used = set()
    for p in preferred:
        for i, lab in enumerate(labels):
            if i in used:
                continue
            if p.lower() in lab.lower():
                ordered_handles.append(handles[i])
                ordered_labels.append(labels[i])
                used.add(i)
    # append remaining (preserve original relative order)
    for i, lab in enumerate(labels):
        if i in used:
            continue
        ordered_handles.append(handles[i])
        ordered_labels.append(labels[i])
    return ordered_handles, ordered_labels



def load_data(dataset, memory_values, alpha):
    data = {}
    for memory in memory_values:
        file_path = f"Performance/{dataset}/memory_{memory}KB_alpha_{alpha}.csv"
        try:
            df = pd.read_csv(file_path)
            data[f"memory_{memory}KB_alpha_{alpha}"] = [
                (row["Sketch Name"], {
                    "Insert Throughput (Mops)": row["Insert Throughput (Mops)"],
                    "Query Throughput (Mops)": row["Query Throughput (Mops)"],
                    "Recall": row["Recall"],
                    "Precision": row["Precision"],
                    "F1 Score": row["F1 Score"],
                    "AAE": row["AAE"],
                    "ARE": row["ARE"]
                }) for _, row in df.iterrows()
            ]
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
    return data


def write_metric_csv(data, metric, dataset, alpha, memory_values):
    rows = []
    first_key = f"memory_{memory_values[0]}KB_alpha_{alpha}"
    methods = [entry[0] for entry in data.get(first_key, [])]
    for memory in memory_values:
        key = f"memory_{memory}KB_alpha_{alpha}"
        entries = data.get(key, [])
        row = {"memory": f"{memory}KB"}
        for method in methods:
            row[method] = next((e[1][metric] for e in entries if e[0] == method), None)
        rows.append(row)
    df_out = pd.DataFrame(rows)
    out_file = f"Performance/{dataset}/{metric.replace(' ', '_')}_alpha_{alpha}.csv"
    df_out.to_csv(out_file, index=False)
    print(f"Saved CSV for {metric} to {out_file}")
    return out_file


def plot_line_chart(data, metric, short_title, alpha, dataset, markers, line_styles,
                    figsize=(12, 8), y_min=None, y_max=None, markersize=None):
    """Plot line chart and save three files:
    1) main figure WITH legend (original)
    2) main figure WITHOUT legend
    3) separate legend figure
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    filtered = {k: v for k, v in data.items() if k.endswith(f"alpha_{alpha}")}
    memories = sorted(int(k.split('_')[1].rstrip('KB')) for k in filtered)
    methods = [e[0] for e in filtered[f"memory_{memories[0]}KB_alpha_{alpha}"]]
    x_pos = list(range(len(memories)))
    cm_N = getattr(VIVID_CM, 'N', 10)
    colors = [VIVID_CM(i % cm_N) for i in range(len(methods))]

    for i, method in enumerate(methods):
        values = [
            next((e[1][metric] for e in filtered[f"memory_{memory}KB_alpha_{alpha}"] if e[0] == method), None)
            for memory in memories
        ]
        ms = markersize if markersize is not None else MARKER_SIZE
        ax.plot(x_pos, values,
                marker=markers[i % len(markers)],
                linestyle=line_styles[0],
                linewidth=3.5,
                color=colors[i],
                label=method,
                alpha=0.9,
                markersize=ms)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(m) for m in memories], fontsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.set_xlabel("Memory (KB)", fontsize=LABEL_SIZE)
    ax.set_ylabel(short_title, fontsize=LABEL_SIZE)

    if y_min is not None or y_max is not None:
        low = y_min if y_min is not None else 0
        high = y_max if y_max is not None else ax.get_ylim()[1]
        ax.set_ylim(low, high)
    elif metric in ["Recall", "Precision", "F1 Score"]:
        ax.set_ylim(0, 1)

    ax.grid(True, linestyle='--', alpha=0.7)
    # Collect handles/labels from actual plotted artists (so markers/lines appear in legend)
    handles, labels = ax.get_legend_handles_labels()
    # Reorder legend entries so preferred methods appear first (use global PREFERRED_ORDER)
    handles, labels = _reorder_handles_labels(handles, labels, PREFERRED_ORDER)
    ncols = min(6, max(1, len(labels)))

    # 1) Save main figure WITH legend (styled like original)
    legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=5, fontsize=LEGEND_SIZE
)
    out_with = f"Performance/{dataset}/{short_title.replace(' ', '_')}_alpha_{alpha}_line_with_legend.png"
    fig.savefig(out_with, bbox_inches='tight')
    print(f"Saved main line chart WITH legend to {out_with}")

    # 2) Remove legend and save main figure WITHOUT legend
    if legend is not None:
        legend.remove()
    out = f"Performance/{dataset}/{short_title.replace(' ', '_')}_alpha_{alpha}_line.png"
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved main line chart WITHOUT legend to {out}")

    plt.close(fig)

    # 3) Prepare and save separate legend figure
    LEGEND_TAG = "line"
    if handles:
        fig_legend = plt.figure(figsize=(10, 1.2))
        # draw legend with same style (rounded box, black edge)
        lg = fig_legend.legend(handles, labels, loc='center', ncol=5, fontsize=LEGEND_SIZE
)
        # ensure legend has visible frame
        for text in lg.get_texts():
            text.set_fontsize(LEGEND_SIZE)
        out_legend = f"Performance/{dataset}/{short_title.replace(' ', '_')}_alpha_{alpha}_legend.png"
        out_legend = out_legend.replace("_legend.png", f"_{LEGEND_TAG}_legend.png")
        fig_legend.savefig(out_legend, bbox_inches='tight')
        plt.close(fig_legend)
        print(f"Saved legend to {out_legend}")
    else:
        print("No legend handles found; no legend saved.")
    if handles:
        fig_legend = plt.figure(figsize=(10, 1.2))
        lg = fig_legend.legend(handles, labels, loc='center', ncol=5, fontsize=LEGEND_SIZE
)
        for text in lg.get_texts():
            text.set_fontsize(LEGEND_SIZE)
        out_legend = f"Performance/{dataset}/{short_title.replace(' ', '_')}_alpha_{alpha}_legend.png"
        out_legend = out_legend.replace("_legend.png", f"_{LEGEND_TAG}_legend.png")
        fig_legend.savefig(out_legend, bbox_inches='tight')
        plt.close(fig_legend)
        print(f"Saved legend to {out_legend}")
    else:
        print("No legend handles found; no legend saved.")


def plot_bar_chart(data, metric, short_title, alpha, dataset,
                   figsize=(12, 8), y_min=None, y_max=None):
    """Plot bar chart and save three files:
    1) main figure WITH legend (original)
    2) main figure WITHOUT legend
    3) separate legend figure
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    filtered = {k: v for k, v in data.items() if k.endswith(f"alpha_{alpha}")}
    memories = sorted(int(k.split('_')[1].rstrip('KB')) for k in filtered)
    methods = [e[0] for e in filtered[f"memory_{memories[0]}KB_alpha_{alpha}"]]
    num = len(methods)
    x = np.arange(len(memories))
    width = 0.8/num
    cm_N = getattr(VIVID_CM, 'N', 10)

    bars = []
    for i, method in enumerate(methods):
        values = [
            next((e[1][metric] for e in filtered[f"memory_{memory}KB_alpha_{alpha}"] if e[0]==method), None)
            for memory in memories
        ]
        bar = ax.bar(x+i*width, values, width, color=VIVID_CM(i%cm_N), label=method)
        bars.append(bar)

    ax.set_xticks(x+width*(num-1)/2)
    ax.set_xticklabels([str(m) for m in memories], fontsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.set_xlabel("Memory (KB)", fontsize=LABEL_SIZE)
    ax.set_ylabel(short_title, fontsize=LABEL_SIZE)

    if y_min is not None or y_max is not None:
        low = y_min if y_min is not None else (0 if metric in ["Recall","Precision","F1 Score"] else ax.get_ylim()[0])
        high = y_max if y_max is not None else (1 if metric in ["Recall","Precision","F1 Score"] else ax.get_ylim()[1])
        ax.set_ylim(low, high)
    elif metric in ["Recall","Precision","F1 Score"]:
        ax.set_ylim(0,1)

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Collect handles/labels
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=VIVID_CM(i%cm_N)) for i in range(len(methods))]
    labels = methods
    # Reorder legend entries so preferred methods appear first (use global PREFERRED_ORDER)
    handles, labels = _reorder_handles_labels(handles, labels, PREFERRED_ORDER)
    ncols = min(6, max(1, len(labels)))

    # 1) Save main figure WITH legend
    legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=5, fontsize=LEGEND_SIZE
)
    out_with = f"Performance/{dataset}/{short_title.replace(' ', '_')}_alpha_{alpha}_bar_with_legend.png"
    fig.savefig(out_with, bbox_inches='tight')
    print(f"Saved main bar chart WITH legend to {out_with}")

    # 2) Remove legend and save main figure WITHOUT legend
    if legend is not None:
        legend.remove()
    out = f"Performance/{dataset}/{short_title.replace(' ', '_')}_alpha_{alpha}_bar.png"
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved main bar chart WITHOUT legend to {out}")

    plt.close(fig)

    # 3) Save separate legend figure
    LEGEND_TAG = "bar"
    if handles:
        fig_legend = plt.figure(figsize=(10, 1.2))
        fig_legend.legend(handles, labels, loc='center', ncol=5, fontsize=LEGEND_SIZE)
        out_legend = f"Performance/{dataset}/{short_title.replace(' ', '_')}_alpha_{alpha}_legend.png"
        out_legend = out_legend.replace("_legend.png", f"_{LEGEND_TAG}_legend.png")
        fig_legend.savefig(out_legend, bbox_inches='tight')
        plt.close(fig_legend)
        print(f"Saved legend to {out_legend}")
    else:
        print("No legend handles found; no legend saved.")


def plot_param_vs_metric_per_memory(csv_file, metric_title, x_label='Parameter', markersize=None):
    df = pd.read_csv(csv_file)
    if df.columns[0].lower() != 'memory':
        print("Error: first column must be 'memory'")
        return
    param_cols = df.columns[1:]
    x_vals = list(map(float, param_cols))
    ms = markersize if markersize is not None else MARKER_SIZE
    for _, row in df.iterrows():
        mem = row['memory']
        y_vals = row[param_cols].tolist()
        fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
        ax.plot(x_vals, y_vals, marker='o', linewidth=3, markersize=ms, alpha=0.8)
        ax.set_xlabel(x_label, fontsize=LABEL_SIZE)
        ax.set_ylabel(metric_title, fontsize=LABEL_SIZE)
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
        ax.grid(True, linestyle='--', alpha=0.7)
        out_fig = f"{os.path.splitext(csv_file)[0]}_{mem}.png"
        fig.savefig(out_fig, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved chart for {mem} to {out_fig}")


if __name__ == '__main__':
    if len(sys.argv)<4:
        print("Usage: python metric.py <dataset> <alpha> <memory1> <memory2> ...")
        sys.exit(1)
    dataset=sys.argv[1]
    alpha=float(sys.argv[2])
    memory_values=list(map(int, sys.argv[3:]))
    markers=['o','s','^','D','v','p','*','h','H','x','+']
    line_styles=['-']
    data=load_data(dataset, memory_values, alpha)
    if not data:
        print("No valid data loaded.")
        sys.exit(1)
        
    # Line charts
    plot_line_chart(data, "Insert Throughput (Mops)", "Insert Throughput (Mops)", alpha, dataset, markers, line_styles, markersize=MARKER_SIZE)
    plot_line_chart(data, "Query Throughput (Mops)", "Query Throughput (Mops)", alpha, dataset, markers, line_styles, markersize=MARKER_SIZE)
    plot_line_chart(data, "AAE", "AAE", alpha, dataset, markers, line_styles, y_max=800, markersize=MARKER_SIZE)
    plot_line_chart(data, "ARE", "ARE", alpha, dataset, markers, line_styles, y_max=0.2, markersize=MARKER_SIZE)
    plot_line_chart(data, "Recall", "Recall", alpha, dataset, markers, line_styles, y_min=0, y_max=1.05, markersize=MARKER_SIZE)
    plot_line_chart(data, "Precision", "Precision", alpha, dataset, markers, line_styles, y_min=0, y_max=1.05, markersize=MARKER_SIZE)
    plot_line_chart(data, "F1 Score", "F1 Score", alpha, dataset, markers, line_styles, y_min=0, y_max=1.05, markersize=MARKER_SIZE)

    # Bar charts
    plot_bar_chart(data, "Insert Throughput (Mops)", "Insert Throughput (Mops)", alpha, dataset)
    plot_bar_chart(data, "Query Throughput (Mops)", "Query Throughput (Mops)", alpha, dataset)

    # CSV outputs
    metrics=["Insert Throughput (Mops)","Query Throughput (Mops)","Recall","Precision","F1 Score","AAE","ARE"]
    csv_files={}
    for m in metrics:
        csv_files[m]=write_metric_csv(data, m, dataset, alpha, memory_values)

    # plot_param_vs_metric_per_memory(csv_files["F1 Score"], "F1 Score", x_label="δ", markersize=MARKER_SIZE)
    # plot_param_vs_metric_per_memory(csv_files["ARE"], "ARE", x_label="δ", markersize=MARKER_SIZE)
