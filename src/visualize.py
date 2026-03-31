import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

from src.parse import MAJOR_GROUP_LABELS

logger = logging.getLogger(__name__)

# Curated set of specific occupation title substrings to label.
# Kept small and diverse to avoid crowding.
LABEL_OCCUPATIONS = [
    # Legal
    "Lawyer",
    "Judge",
    # Health — clinical
    "General practitioner",
    "Dentist",
    # Information technology
    "Software developer",
    "Systems analyst",
    # Engineering
    "Civil engineer",
    "Mechanical engineer",
    "Electrical engineer",
    "Electronics engineer",
    # Education
    "Primary school teacher",
    "Secondary school teacher",
    # Transport
    "Truck driver",
    "Bus driver",
    # Finance
    "Accountant",
    "Financial analyst",
    # Food
    "Chef",
    "Cook",
    # Social sciences
    "Economist",
    "Statistician",
    # Design & planning
    "Architect",
    "Urban planner",
    # Protective services
    "Police officer",
    "Firefighter",
]

FIELD_TITLES = {
    "title": "Title",
    "definition": "Definition",
    "tasks": "Tasks",
    "included": "Included Occupations",
}


def plot_umap(
    reduced: dict[str, dict[str, pd.DataFrame]],
    output_dir: str = "outputs/figures",
    filename: str = "umap_2x2.png",
) -> str:
    """Generate a 2×2 grid of UMAP projections, one panel per text field.

    Each panel is colored by ISCO major group with selective occupation labels.

    Args:
        reduced: Nested dict from reduce_embeddings: {field: {"umap": DataFrame, ...}}.
        output_dir: Directory to save the figure.
        filename: Output filename.

    Returns:
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fields = [f for f in ["title", "definition", "tasks", "included"] if f in reduced]
    if not fields:
        raise ValueError("No UMAP data found in reduced dict.")

    cmap = plt.get_cmap("tab10")
    major_groups = sorted(MAJOR_GROUP_LABELS.keys())
    color_map = {mg: cmap(i / 10) for i, mg in enumerate(major_groups)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for ax, field in zip(axes_flat, fields):
        df = reduced[field]["umap"]
        _draw_panel(ax, df, color_map, FIELD_TITLES.get(field, field))

    # Hide any unused panels
    for ax in axes_flat[len(fields):]:
        ax.set_visible(False)

    # Shared legend
    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=color_map.get(mg, "grey"),
            markersize=8,
            label=f"{mg} – {MAJOR_GROUP_LABELS.get(mg, mg)}",
        )
        for mg in major_groups
    ]
    fig.legend(
        handles=legend_handles,
        title="ISCO Major Group",
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8,
        title_fontsize=9,
    )

    fig.suptitle("ISCO-08 Occupations — UMAP Projections by Text Field", fontsize=13, y=1.01)
    plt.tight_layout()
    # Reserve space at the bottom so the legend sits below all subplots without overlapping.
    fig.subplots_adjust(bottom=0.12)

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)
    return out_path


def _remove_outliers(df: pd.DataFrame, iqr_factor: float = 1.5) -> pd.DataFrame:
    # Outliers are removed from the plot only — all other outputs (embeddings,
    # similarity matrices, reduced coordinate CSVs) retain the full dataset.
    # UMAP can project a small number of occupations far from the main cloud
    # (isolated points with no close neighbours), which distort the axis scale
    # and make the bulk of the data hard to read. IQR filtering on both axes
    # trims these extreme points without touching the underlying data files.
    result = df.copy()
    for axis in ("x", "y"):
        q1, q3 = result[axis].quantile(0.25), result[axis].quantile(0.75)
        iqr = q3 - q1
        result = result[result[axis].between(q1 - iqr_factor * iqr, q3 + iqr_factor * iqr)]
    removed = df.loc[~df.index.isin(result.index), "title"]
    if not removed.empty:
        logger.debug("Removed %d outlier(s) from plot (IQR × %.1f)", len(removed), iqr_factor)
        print(f"  Outliers removed from plot ({len(removed)}):")
        for name in removed.values:
            print(f"    - {name}")
    return result


def _draw_panel(ax: plt.Axes, df: pd.DataFrame, color_map: dict, title: str) -> None:
    """Draw a single UMAP scatter panel with coloring and selective labels."""
    df = _remove_outliers(df)

    for mg, group in df.groupby("major_group"):
        color = color_map.get(str(mg), "grey")
        ax.scatter(
            group["x"],
            group["y"],
            c=[color],
            s=12,
            alpha=0.7,
            linewidths=0,
        )

    # Selective labels: for each keyword find the single shortest matching title
    # (shortest = most generic unit group, avoids specialised sub-variants).
    seen: set[int] = set()
    rows_to_label = []
    for keyword in LABEL_OCCUPATIONS:
        matches = df[df["title"].str.contains(keyword, case=False, na=False)]
        if matches.empty:
            continue
        best = matches.loc[matches["title"].str.len().idxmin()]
        if best.name not in seen:
            seen.add(best.name)
            rows_to_label.append(best)
    labeled = pd.DataFrame(rows_to_label) if rows_to_label else pd.DataFrame()

    texts = []
    for _, row in labeled.iterrows():
        t = ax.text(row["x"], row["y"], row["title"], fontsize=8, alpha=0.9)
        texts.append(t)

    if texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
            expand=(1.2, 1.4),
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
