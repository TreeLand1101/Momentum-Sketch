#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
import sys

LABEL_SIZE = 20
TICK_SIZE = 20
LEGEND_SIZE = 16
MARKER_SIZE = 16

VIVID_CM = colormaps['tab10']

def load_data(memory_values):
    data = {}
    for memory in memory_values:
        file_path = f"memory_{memory}KB.csv"
        try:
            df = pd.read_csv(file_path)
            data[f"memory_{memory}KB"] = [
                (row["Filter Name"], {
                    "Insert Throughput (Mops)": row["Insert Throughput (Mops)"],
                    "Query Throughput (Mops)": row["Query Throughput (Mops)"],
                    "AAE": row["AAE"],
                    "ARE": row["ARE"]
                }) for _, row in df.iterrows()
            ]
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
    return data


def plot_line_chart(data, metric, ylabel, markers, line_styles, figsize=(10, 6), marker_size=None):
    plt.figure(figsize=figsize)
    memory_keys = sorted(data.keys(), key=lambda k: int(k.split('_')[1].rstrip('KB')))
    memories = [int(k.split('_')[1].rstrip('KB')) for k in memory_keys]
    x_pos = list(range(len(memories)))
    methods = [entry[0] for entry in data[memory_keys[0]]]

    ms = marker_size if marker_size is not None else MARKER_SIZE

    for i, method in enumerate(methods):
        values = []
        for key in memory_keys:
            val = next((entry[1][metric] for entry in data[key] if entry[0] == method), None)
            values.append(val)
        plt.plot(
            x_pos,
            values,
            marker=markers[i % len(markers)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2.5,
            color=VIVID_CM(i % VIVID_CM.N),
            label=method,
            alpha=0.9,
            markersize=ms
        )

    plt.xticks(x_pos, [str(m) for m in memories], fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.xlabel("Memory (KB)", fontsize=LABEL_SIZE)
    plt.ylabel(ylabel, fontsize=LABEL_SIZE)
    plt.legend(
        fontsize=LEGEND_SIZE,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=len(methods),
    )
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    fname = metric.replace(' ', '_').replace('(', '').replace(')', '')
    output_file = f"{fname}_line.png"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved line chart to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metric.py <memory1> <memory2> ...")
        sys.exit(1)

    memory_values = sys.argv[1:]
    data = load_data(memory_values)

    if not data:
        print("No data found. Please check if CSV files are generated correctly.")
        sys.exit(1)

    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', 'x', '+']
    line_styles = ['-', '--']

    plot_line_chart(data, "Insert Throughput (Mops)", "Insert Throughput (Mops)", markers, line_styles, marker_size=14)
    plot_line_chart(data, "Query Throughput (Mops)", "Query Throughput (Mops)", markers, line_styles, marker_size=14)
    plot_line_chart(data, "AAE", "AAE", markers, line_styles, marker_size=14)
    plot_line_chart(data, "ARE", "ARE", markers, line_styles, marker_size=14)
