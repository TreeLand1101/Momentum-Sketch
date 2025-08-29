import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter

LABEL_SIZE  = 30
TICK_SIZE   = 30

def plot_rank_frequency(ranks, freqs, out_dir: Path, prefix: str, label_flow: str, log_scale: bool):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(ranks, freqs, marker='o', linestyle='-', alpha=0.7)
    
    ax.set_xlabel("Rank", fontsize=LABEL_SIZE)
    ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)
    
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    
    if log_scale:
        suffix = f"{prefix}_rank-frequency_log.png"
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
    else:
        suffix = f"{prefix}_rank-frequency.png"
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        ax.xaxis.set_major_formatter(ScalarFormatter())  # ensure no sci notation style
        ax.ticklabel_format(style='plain', axis='x')
    
    # Adjust offset text size for both axes
    ax.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    out = out_dir / suffix
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

def plot_cdf(freqs, out_dir: Path, prefix: str, label_flow: str, log_x: bool):
    sorted_freq = np.sort(freqs)
    cdf = np.linspace(1 / freqs.size, 1, freqs.size)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(sorted_freq, cdf, marker='o', linestyle='-', alpha=0.7)
    
    ax.set_xlabel("Frequency", fontsize=LABEL_SIZE)
    ax.set_ylabel("CDF", fontsize=LABEL_SIZE)
    
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    
    if log_x:
        suffix = f"{prefix}_cdf_log.png"
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
    else:
        suffix = f"{prefix}_cdf.png"
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')
    
    # Adjust offset text size for x-axis
    ax.xaxis.get_offset_text().set_fontsize(TICK_SIZE)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    out = out_dir / suffix
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

def process_file(out_dir: Path, prefix: str, tag: str, label_flow: str):
    csv_file = out_dir / f"{prefix}_{tag}.csv"
    try:
        freqs = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Error opening file: {csv_file}")
        return
    ranks = np.arange(1, freqs.size + 1)
    
    plot_rank_frequency(ranks, freqs, out_dir, f"{prefix}_{tag}", label_flow, log_scale=False)
    plot_rank_frequency(ranks, freqs, out_dir, f"{prefix}_{tag}", label_flow, log_scale=True)
    plot_cdf(freqs, out_dir, f"{prefix}_{tag}", label_flow, log_x=False)
    plot_cdf(freqs, out_dir, f"{prefix}_{tag}", label_flow, log_x=True)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Rank-Frequency Distribution and CDF of Flow Frequencies with Memory"
    )
    parser.add_argument("--dataset", nargs='+', help="Dataset filenames (e.g., Campus.dat)")
    parser.add_argument("--memory", type=int, required=True, help="Memory parameter")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha threshold")
    args = parser.parse_args()

    prefix = f"memory_{args.memory}KB_threshold_{args.alpha:.6f}"
    for dataset in args.dataset:
        out_dir = Path(dataset)
        if not out_dir.is_dir():
            print(f"Warning: not a directory: {out_dir}")
            continue

        for tag, label_flow in [("total", "total"), ("retained", "retained")]:
            print(f"Processing {out_dir / f'{prefix}_{tag}.csv'}...")
            process_file(out_dir, prefix, tag, label_flow)

if __name__ == "__main__":
    main()
