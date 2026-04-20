from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = FIG_DIR / "methodology_workflow.png"

BACKGROUND = "#F3EDE4"
PANEL = "#FBF6EF"
TEXT_DARK = "#2F2F2F"
TEXT_MUTED = "#5E5B55"
EDGE = "#2F2F2F"
RED = "#D50000"
SALMON = "#DD5D61"
ORANGE = "#F0A65D"
YELLOW = "#F0C55B"
GREEN = "#8FB85C"
DARK_GREEN = "#5F9B61"
REFERENCE_BLUE = "#6EA0D6"
LIGHT_RED = "#F8D6D1"
LIGHT_ORANGE = "#FBE5C5"
LIGHT_GREEN = "#E7EFCF"
LIGHT_YELLOW = "#FAEFC7"
LIGHT_BLUE = "#DCEAF7"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 10,
        "axes.linewidth": 0.0,
        "figure.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
    }
)


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    accent: str,
    title_size: int = 13,
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.14",
        linewidth=1.4,
        edgecolor=EDGE,
        facecolor=face,
    )
    ax.add_patch(box)

    ax.add_patch(
        patches.Rectangle(
            (x, y + h - 0.16),
            w,
            0.16,
            linewidth=0,
            facecolor=accent,
            alpha=0.95,
        )
    )

    ax.text(
        x + 0.28,
        y + h - 0.46,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=TEXT_DARK,
    )
    ax.text(
        x + 0.28,
        y + h - 0.98,
        body,
        ha="left",
        va="top",
        fontsize=9.4,
        color=TEXT_MUTED,
        linespacing=1.22,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
    color: str = TEXT_DARK,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "mutation_scale": 18,
            "lw": 1.9,
            "color": color,
            "shrinkA": 5,
            "shrinkB": 5,
            "connectionstyle": f"arc3,rad={rad}",
        },
    )


def add_stage_label(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.08",
            "facecolor": PANEL,
            "edgecolor": "#C9BFAF",
            "linewidth": 0.8,
        },
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=300)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    add_box(
        ax,
        0.65,
        5.32,
        3.15,
        1.52,
        "Training Data",
        "50 noisy measurements\n$(z, T_v, u_i^{noisy})$",
        LIGHT_RED,
        RED,
    )
    add_box(
        ax,
        0.65,
        3.22,
        3.15,
        1.52,
        "Physics Points",
        "2,000 collocation points\nTerzaghi PDE domain",
        LIGHT_ORANGE,
        ORANGE,
    )
    add_box(
        ax,
        4.7,
        4.34,
        4.25,
        2.42,
        "PINN Architecture",
        "Multilayer perceptron\n5 hidden layers x 50 neurons\nDropout, $p = 0.1$\nLearnable parameter: $c_v$",
        LIGHT_ORANGE,
        SALMON,
    )
    add_box(
        ax,
        4.7,
        1.15,
        4.25,
        2.15,
        "Optimization",
        "$L = L_{data} + L_{physics}$\nAutomatic differentiation\nAdam optimizer, 5,000 epochs",
        LIGHT_GREEN,
        DARK_GREEN,
    )
    add_box(
        ax,
        10.25,
        4.62,
        3.15,
        1.94,
        "MC Dropout UQ",
        "1,000 stochastic passes\nPredictive distribution\nPointwise uncertainty",
        LIGHT_BLUE,
        REFERENCE_BLUE,
    )
    add_box(
        ax,
        10.25,
        1.62,
        3.15,
        1.94,
        "Final Outputs",
        "Recovered $c_v$\nPredictive mean $\\hat{u}$\n95% confidence bounds",
        LIGHT_YELLOW,
        GREEN,
    )

    add_stage_label(ax, 2.22, 7.16, "Inputs", RED)
    add_stage_label(ax, 6.82, 7.16, "Inverse PINN Training", SALMON)
    add_stage_label(ax, 11.82, 7.16, "Prediction + UQ", GREEN)

    add_arrow(ax, (3.82, 6.08), (4.68, 5.78), color=RED)
    add_arrow(ax, (3.82, 3.98), (4.68, 5.04), color=ORANGE, rad=-0.12)
    add_arrow(ax, (6.82, 4.32), (6.82, 3.34), color=GREEN)
    add_arrow(ax, (8.96, 5.55), (10.22, 5.55), color=REFERENCE_BLUE)
    add_arrow(ax, (11.82, 4.6), (11.82, 3.58), color=GREEN)
    add_arrow(ax, (6.0, 3.32), (6.0, 4.32), color=GREEN, rad=-0.32)

    ax.text(
        5.05,
        3.82,
        "update weights and $c_v$",
        ha="left",
        va="center",
        fontsize=8.8,
        color=TEXT_MUTED,
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.06",
            "facecolor": BACKGROUND,
            "edgecolor": "none",
            "alpha": 0.96,
        },
    )
    ax.text(
        9.55,
        5.9,
        "inference",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color=TEXT_MUTED,
    )
    ax.text(
        12.16,
        4.08,
        "aggregate",
        ha="left",
        va="center",
        fontsize=8.8,
        color=TEXT_MUTED,
    )


    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Methodological workflow figure successfully generated at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
