"""Shared plotting helpers: consistent styling and figure saving."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file without needing a display
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
DPI = 300

# Okabe-Ito palette: distinguishable for colour-blind readers.
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

# Fixed colours for severity so it looks the same in every figure.
SEVERITY_COLOURS = {
    "Fatal accident": "#D55E00",
    "Serious injury accident": "#E69F00",
    "Other injury accident": "#0072B2",
    "Non injury accident": "#999999",
}


def set_style():
    """Apply the project-wide matplotlib style. Call once before plotting."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": plt.cycler(color=COLOURS),
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def label_axes(ax, title, xlabel, ylabel):
    """Set title and axis labels. Include units in the labels,
    e.g. 'Speed limit (km/h)', 'Crashes (count)'."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def save_fig(fig, name):
    """Save fig to figures/<name>.png at 300 dpi, close it, return the path."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def save_table(table, name):
    """Save a DataFrame to results/<name>.csv (without the index) and return
    the path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    table.to_csv(path, index=False)
    return path
