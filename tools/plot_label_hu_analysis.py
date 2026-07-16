#!/usr/bin/env python3
"""Create a publication-style summary of the Label 3 HU analysis."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = {"charite": "#C44E52", "pengwin": "#168AAD"}
DISPLAY_NAMES = {"charite": "Charite cohort", "pengwin": "PENGWIN cohort"}
SOURCES = ("charite", "pengwin")


def load_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def values_by_source(cases: list[dict[str, str]], field: str) -> dict[str, np.ndarray]:
    values = {}
    for source in SOURCES:
        values[source] = np.asarray(
            [float(case[field]) for case in cases if case["source"] == source and case[field]],
            dtype=float,
        )
    return values


def format_p(value: float) -> str:
    if value < 0.001:
        exponent = int(np.floor(np.log10(value)))
        coefficient = value / (10**exponent)
        return rf"$p={coefficient:.1f}\times10^{{{exponent}}}$"
    return rf"$p={value:.3f}$"


def mann_whitney_p(values: dict[str, np.ndarray]) -> float:
    """Two-sided asymptotic Mann-Whitney p-value with tie correction."""
    first, second = (values[source] for source in SOURCES)
    combined = np.concatenate((first, second))
    order = np.argsort(combined, kind="stable")
    ranks = np.empty(combined.size, dtype=float)
    start = 0
    while start < combined.size:
        stop = start + 1
        while stop < combined.size and combined[order[stop]] == combined[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop

    n_first, n_second = first.size, second.size
    u = ranks[:n_first].sum() - n_first * (n_first + 1) / 2
    expected = n_first * n_second / 2
    tie_term = sum(count**3 - count for count in Counter(combined).values())
    n_total = combined.size
    variance = n_first * n_second / 12 * (
        n_total + 1 - tie_term / (n_total * (n_total - 1))
    )
    z = (abs(u - expected) - 0.5) / np.sqrt(variance)
    return math.erfc(z / np.sqrt(2))


def distribution_panel(
    ax: plt.Axes,
    values: dict[str, np.ndarray],
    ylabel: str,
    title: str,
    panel: str,
    p_value: float,
    baseline: float | None = None,
) -> None:
    positions = np.arange(len(SOURCES), dtype=float)
    arrays = [values[source] for source in SOURCES]

    violins = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, source in zip(violins["bodies"], SOURCES):
        body.set_facecolor(COLORS[source])
        body.set_edgecolor(COLORS[source])
        body.set_alpha(0.22)

    box = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1B1B1B", "linewidth": 2.0},
        whiskerprops={"color": "#444444", "linewidth": 1.1},
        capprops={"color": "#444444", "linewidth": 1.1},
    )
    for patch, source in zip(box["boxes"], SOURCES):
        patch.set_facecolor(COLORS[source])
        patch.set_edgecolor(COLORS[source])
        patch.set_alpha(0.72)

    rng = np.random.default_rng(20260713)
    for position, source, array in zip(positions, SOURCES, arrays):
        jitter = rng.normal(0.0, 0.055, size=array.size)
        ax.scatter(
            position + jitter,
            array,
            s=16,
            color=COLORS[source],
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )

    if baseline is not None:
        ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.1, zorder=0)

    ax.set_xticks(positions, [f"n = {len(array)}" for array in arrays])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.text(-0.14, 1.04, panel, transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.text(0.98, 0.98, format_p(p_value), transform=ax.transAxes, ha="right", va="top")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases_csv", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()

    cases = load_cases(args.cases_csv)

    panels = (
        (
            "label3_median_hu",
            "Median HU within Label 3",
            "Label 3 intensity",
            "A",
            None,
        ),
        (
            "local_median_delta_hu",
            "Label 3 - neighboring bone (HU)",
            "Local intensity contrast",
            "B",
            0.0,
        ),
        (
            "local_auc_separability",
            "Intensity separability (AUC)",
            "Local HU separability",
            "C",
            0.5,
        ),
        (
            "local_overlap_coefficient",
            "Overlap coefficient",
            "Local distribution overlap",
            "D",
            None,
        ),
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#111111",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    for ax, panel in zip(axes.flat, panels):
        field, ylabel, title, letter, baseline = panel
        values = values_by_source(cases, field)
        distribution_panel(
            ax,
            values,
            ylabel,
            title,
            letter,
            mann_whitney_p(values),
            baseline,
        )

    axes[1, 0].set_ylim(0.48, 0.79)
    axes[1, 1].set_ylim(0.45, 0.98)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS[source],
            markeredgecolor="none",
            markersize=8,
            label=DISPLAY_NAMES[source],
        )
        for source in SOURCES
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        frameon=False,
    )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
