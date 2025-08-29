import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import matplotlib.ticker as ticker

LABEL_SIZE  = 30
TICK_SIZE   = 30
LEGEND_SIZE = 24

# Default square figure size (width, height)
FIG_SIZE = (9, 6)

CAP_VALUE = 10**9


def read_csv(file_name):
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 PlotCDF.py <dataset1> [dataset2 ...]")
        sys.exit(1)

    datasets = sys.argv[1:]

    for ds in datasets:
        folder = ds

        nonheavy_file    = os.path.join(folder, "nonheavy_momentum_cdf.csv")
        heavyhitter_file = os.path.join(folder, "heavyhitter_momentum_cdf.csv")

        if not os.path.exists(nonheavy_file) or not os.path.exists(heavyhitter_file):
            print(f"Warning: CSV not found in folder '{folder}'. Skipping.")
            continue

        nonheavy_x, nonheavy_y = read_csv(nonheavy_file)
        heavy_x,    heavy_y    = read_csv(heavyhitter_file)

        # Cap values and optionally rescale if desired (not shown here)
        nonheavy_x = np.minimum(nonheavy_x, CAP_VALUE)
        heavy_x    = np.minimum(heavy_x,    CAP_VALUE)

        # Create a square plot
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        ax.plot(nonheavy_x, nonheavy_y,
                marker='o', linestyle='-',
                label='Non-Heavy Flows')
        ax.plot(heavy_x,    heavy_y,
                marker='o', linestyle='-',
                label='Heavy Hitters')

        ax.set_xlabel("Momentum", fontsize=LABEL_SIZE)
        ax.set_ylabel("CDF",       fontsize=LABEL_SIZE)

        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # Enable scientific notation for x-axis
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        # Adjust the offset text size (e.g., '1e9')
        ax.xaxis.get_offset_text().set_fontsize(TICK_SIZE)

        ax.legend(fontsize=LEGEND_SIZE)
        ax.grid(True, which="both", linestyle="--", alpha=0.7)

        out_png = os.path.join(folder, "momentum_cdf_plot.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved square plot: {out_png}")
