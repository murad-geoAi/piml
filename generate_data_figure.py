from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = Path(__file__).resolve().parent / "synthetic_sensor_data.csv"

SEED = 42
H = 2.0
TV_MAX = 1.0
N_INTERIOR = 2000
NOISE_STD = 0.05
BETA_A = 1.2
BETA_B = 3.0

FIGURE_BG = "#FFFFFF"
AXES_BG = "#FFFFFF"
DARK_TEXT = "#000000"
GRID_WARM = "#E0E0E0"
RED = "#D50000"
SALMON = "#DD5D61"
ORANGE = "#F0A65D"
YELLOW = "#F0C55B"
GREEN = "#8FB85C"
DARK_GREEN = "#5F9B61"
REFERENCE_BLUE = "#3366CC"  # Darkened for higher contrast
COLLOCATION_GRAY = "#008080" # Changed to Teal for distinction

COLOR_COLLOCATION = COLLOCATION_GRAY
COLOR_TIME = SALMON
COLOR_DEPTH = GREEN
COLOR_TARGET = REFERENCE_BLUE
COLOR_REFERENCE = DARK_TEXT
COLOR_GRID = GRID_WARM
CMAP_PRESSURE = LinearSegmentedColormap.from_list(
    "warm_pressure",
    [DARK_GREEN, GREEN, YELLOW, ORANGE, SALMON, RED],
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.facecolor": FIGURE_BG,
        "axes.facecolor": AXES_BG,
        "savefig.facecolor": FIGURE_BG,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": DARK_TEXT,
        "axes.labelcolor": DARK_TEXT,
        "text.color": DARK_TEXT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT,
        "legend.fontsize": 8.5,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "legend.edgecolor": "#CCCCCC",
        "legend.facecolor": "#FFFFFF",
    }
)


def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x_safe = np.clip(x, 1e-6, 1.0 - 1e-6)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return np.exp((a - 1.0) * np.log(x_safe) + (b - 1.0) * np.log1p(-x_safe) - log_beta)


def apply_light_grid(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(CSV_PATH)

    sensor_z = df["Depth_z"].to_numpy()
    sensor_t = df["Time_t"].to_numpy()
    sensor_u_noisy = df["Noisy_u"].to_numpy()

    collocation_z = rng.uniform(0.0, H, N_INTERIOR)
    collocation_t = rng.uniform(0.0, TV_MAX, N_INTERIOR)

    fig = plt.figure(figsize=(11.4, 5.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(2.25, 1.0), height_ratios=(1.0, 1.0))

    ax_domain = fig.add_subplot(gs[:, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_depth = fig.add_subplot(gs[1, 1])

    ax_domain.scatter(
        collocation_t,
        collocation_z,
        s=7,
        color=COLOR_COLLOCATION,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
        label="Representative PDE collocation sample",
    )

    sensor_sizes = 36 + 190 * np.abs(sensor_u_noisy) / (np.max(np.abs(sensor_u_noisy)) + 1e-8)
    sensor_scatter = ax_domain.scatter(
        sensor_t,
        sensor_z,
        c=sensor_u_noisy,
        s=sensor_sizes,
        cmap=CMAP_PRESSURE,
        edgecolors="white",
        linewidths=0.75,
        alpha=0.94,
        zorder=4,
        label="Noisy sensor observations",
    )

    ax_domain.axhline(0.0, color=COLOR_REFERENCE, linestyle="--", linewidth=1.2, zorder=2)
    ax_domain.axhline(H, color=COLOR_REFERENCE, linestyle="--", linewidth=1.2, zorder=2)
    ax_domain.axvline(0.0, color=COLOR_REFERENCE, linestyle=":", linewidth=1.1, zorder=2)

    ax_domain.text(
        0.985,
        0.035,
        "$u=0$ drainage boundaries: $z=0,H$",
        transform=ax_domain.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        color=COLOR_REFERENCE,
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.05",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CCCCCC",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
    )

    ax_domain.set_title("(a) Training data and physics points in the consolidation domain")
    ax_domain.set_xlabel("Time factor, $T_v$")
    ax_domain.set_ylabel("Depth, $z$ (m)")
    ax_domain.set_xlim(0.0, TV_MAX)
    ax_domain.set_ylim(H, 0.0)
    ax_domain.set_box_aspect(0.72)
    apply_light_grid(ax_domain)

    collocation_handle = mlines.Line2D(
        [],
        [],
        color=COLOR_COLLOCATION,
        marker="o",
        linestyle="None",
        markersize=5,
        alpha=0.45,
        label="PDE collocation, $n=2000$ per epoch",
    )
    sensor_handle = mlines.Line2D(
        [],
        [],
        color="#555555",
        marker="o",
        markerfacecolor=RED,
        markeredgecolor="white",
        linestyle="None",
        markersize=8,
        label="Sensor data, $n=50$",
    )
    boundary_handle = mlines.Line2D(
        [],
        [],
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=1.2,
        label="Zero-pressure drainage boundaries",
    )
    ax_domain.legend(
        handles=[sensor_handle, collocation_handle, boundary_handle],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        borderpad=0.8,
    )

    cbar = fig.colorbar(sensor_scatter, ax=ax_domain, shrink=0.82, pad=0.02)
    cbar.set_label("Noisy excess pore pressure, $u_i^{\\mathrm{noisy}}$")

    ax_time.hist(
        sensor_t,
        bins=np.linspace(0.0, TV_MAX, 11),
        density=True,
        color=COLOR_TIME,
        alpha=0.62,
        edgecolor="white",
        linewidth=0.8,
        label="Realized sensors",
    )
    x_beta = np.linspace(0.001, 0.999, 300)
    ax_time.plot(
        x_beta,
        beta_pdf(x_beta, BETA_A, BETA_B),
        color=COLOR_TARGET,
        linewidth=2.0,
        linestyle="--",
        label=r"Target $T_v \sim \mathrm{Beta}(1.2,3.0)$",
    )
    ax_time.set_title("(b) Sensor time sampling")
    ax_time.set_xlabel("Time factor, $T_v$")
    ax_time.set_ylabel("Density")
    ax_time.set_xlim(0.0, TV_MAX)
    ax_time.legend(loc="upper right")
    apply_light_grid(ax_time)

    ax_depth.hist(
        sensor_z,
        bins=np.linspace(0.0, H, 11),
        density=True,
        color=COLOR_DEPTH,
        alpha=0.62,
        edgecolor="white",
        linewidth=0.8,
        label="Realized sensors",
    )
    ax_depth.hlines(
        1.0 / (0.96 * H),
        xmin=0.02 * H,
        xmax=0.98 * H,
        color=COLOR_TARGET,
        linewidth=2.0,
        linestyles="--",
        label=r"Target $z \sim U(0.02H,0.98H)$",
    )
    ax_depth.set_title("(c) Sensor depth sampling")
    ax_depth.set_xlabel("Depth, $z$ (m)")
    ax_depth.set_ylabel("Density")
    ax_depth.set_xlim(0.0, H)
    ax_depth.legend(loc="upper right")
    apply_light_grid(ax_depth)

    recipe = (
        "Synthetic observations: "
        r"$u_i^{noisy}=u(z_i,T_{v,i};c_v=1.2)+\epsilon_i$, "
        r"$\epsilon_i \sim N(0,0.05^2)$"
        "\nPhysics loss: uniformly sampled interior collocation points over "
        r"$0<z<H$, $0<T_v\leq1$."
    )
    fig.text(
        0.52,
        0.005,
        recipe,
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#000000",
    )

    output_path_png = FIG_DIR / "figure_data_distribution.png"
    output_path_pdf = FIG_DIR / "figure_data_distribution.pdf"
    fig.savefig(output_path_png, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Successfully generated corrected data distribution figure at {output_path_png}")
    print(f"Successfully generated corrected data distribution PDF at {output_path_pdf}")


if __name__ == "__main__":
    main()
