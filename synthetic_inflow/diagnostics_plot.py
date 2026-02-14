"""Utility functions for diagnostic visualization of historical vs synthetic inflow series."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# 项目根目录 -> markdown/image/论文一
PROJECT_ROOT = Path(__file__).resolve().parents[4]
FIGURE7_DIR = PROJECT_ROOT / "markdown" / "image" / "论文一"

# 图例名称映射，用于将原来名称替换为 D1/D2/D3
LEGEND_NAME_MAP = {
    "Historical": "HD",
    "baseline": "GD",
    "enhanced_flow": "ED",
}


fontsize = 20
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = fontsize
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


def _format_limits(limit: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if limit is None:
        return None
    left, right = limit
    if left is None and right is None:
        return None
    if left is None:
        left = 0.0
    if right is None:
        return (left, None)
    if left > right:
        left, right = right, left
    return left, right


def save_diagnostics(
    hist_summary,
    synth_summary,
    synthetic_series: pd.Series,
    observed_series: pd.Series,
    output_dir: Path,
    figsize: Tuple[int, int] = (16, 10),
    bins: int = 60,
) -> Path:
    """Generate diagnostic comparison plots and save them to ``diagnostics.png``.

    Parameters
    ----------
    hist_summary, synth_summary
        Statistical summaries produced by ``StatisticalValidator``.
    synthetic_series : pd.Series
        Generated synthetic inflow series.
    observed_series : pd.Series
        Historical inflow observations.
    output_dir : pathlib.Path
        Destination directory for the exported figure.
    histogram_xlim : tuple, optional
        (min, max) bounds for the histogram subplot. Defaults to (0, 5000).
    qq_xlim : tuple, optional
        (min, max) bounds for the QQ subplot; the default adapts to the data.
    figsize : tuple, optional
        Figure size passed to matplotlib.
    bins : int, optional
        Number of histogram bins.

    Returns
    -------
    Path
        The path of the saved diagnostic figure.
    """

    months = hist_summary.monthly_mean.index
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    axes[0].plot(months, hist_summary.monthly_mean.values, marker="o", label="Historical")
    axes[0].plot(months, synth_summary.monthly_mean.values, marker="s", label="Synthetic")
    axes[0].set_title("Monthly mean inflow")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Discharge (m$^3$/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    obs_clean = observed_series.dropna()
    syn_clean = synthetic_series.dropna()
    axes[1].hist(obs_clean, bins=bins, density=True, alpha=0.6, label="Historical")
    axes[1].hist(syn_clean, bins=bins, density=True, alpha=0.6, label="Synthetic")
    axes[1].set_title("Daily discharge density")
    axes[1].set_xlabel("Discharge (m$^3$/s)")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)
    axes[1].set_xlim(left=0, right=40000)

    persistence_hist = hist_summary.persistence_metrics
    persistence_synth = synth_summary.persistence_metrics
    labels = ["Low-flow mean", "Low-flow max", "High-flow mean", "High-flow max"]
    hist_vals = [
        persistence_hist.low_mean,
        persistence_hist.low_max,
        persistence_hist.high_mean,
        persistence_hist.high_max,
    ]
    synth_vals = [
        persistence_synth.low_mean,
        persistence_synth.low_max,
        persistence_synth.high_mean,
        persistence_synth.high_max,
    ]
    x = np.arange(len(labels))
    width = 0.35
    axes[2].bar(x - width / 2, hist_vals, width, label="Historical")
    axes[2].bar(x + width / 2, synth_vals, width, label="Synthetic")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=-12, ha="center")
    axes[2].set_ylabel("Duration (days)")
    axes[2].set_title("High/low-flow duration")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend(frameon=False, loc='upper left')

    if obs_clean.size > 0 and syn_clean.size > 0:
        quantile_count = min(200, obs_clean.size, syn_clean.size)
        probs = np.linspace(0.01, 0.99, quantile_count)
        hist_quant = np.quantile(obs_clean, probs)
        synth_quant = np.quantile(syn_clean, probs)
        axes[3].scatter(hist_quant, synth_quant, alpha=0.7, label="Quantile pairs")

        x_limits = axes[3].get_xlim()
        y_limits = axes[3].get_ylim()
        line_min = min(x_limits[0], y_limits[0])
        line_max = max(x_limits[1], y_limits[1])
        axes[3].plot([0, 40000], [0, 40000], "k--", label="45° reference line")
        axes[3].legend(frameon=False)
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(
            0.5,
            0.5,
            "Insufficient data for QQ plot",
            ha="center",
            va="center",
            transform=axes[3].transAxes,
        )
        axes[3].set_axis_off()

    axes[3].set_title("Q-Q plot: quantile comparison")
    axes[3].set_xlabel("Historical quantile (m$^3$/s)")
    axes[3].set_ylabel("Synthetic quantile (m$^3$/s)")
    axes[3].set_xlim(left=0, right=40000)
    axes[3].set_ylim(bottom=0, top=40000)

    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    for label, ax in zip(subplot_labels, axes):
        ax.text(
            -0.12,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "diagnostics.png"
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_multi_diagnostics(
    hist_summary,
    observed_series: pd.Series,
    synthetic_entries: List[dict],
    output_dir: Path,
    figsize: Tuple[int, int] = (16, 10),
    bins: int = 60,
) -> Path:
    """Generate diagnostics comparing historical data with multiple synthetic series.

    Parameters
    ----------
    hist_summary
        Statistical summary of the historical series.
    observed_series : pd.Series
        Historical inflow observations.
    synthetic_entries : List[dict]
        List of experiment dicts with keys ``name`` (str), ``series`` (pd.Series),
        and ``summary`` (StatisticalSummary).
    output_dir : pathlib.Path
        Destination directory for the exported figure.
    figsize : tuple, optional
        Figure size passed to matplotlib.
    bins : int, optional
        Number of histogram bins.

    Returns
    -------
    Path
        Path of the saved diagnostic figure.
    """

    if not synthetic_entries:
        raise ValueError("synthetic_entries 不能为空")

    # Figure7 固定输出目录
    figure_dir = FIGURE7_DIR
    figure_dir.mkdir(parents=True, exist_ok=True)

    obs_clean = observed_series.dropna()
    colors = [plt.get_cmap("tab10")(idx % 10) for idx in range(len(synthetic_entries))]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # (a) Monthly mean inflow
    months = hist_summary.monthly_mean.index
    mapped_historical = LEGEND_NAME_MAP.get("Historical", "Historical")
    axes[0].plot(
        months,
        hist_summary.monthly_mean.values,
        marker="o",
        color="black",
        linewidth=2.5,
        label=mapped_historical,
    )
    for color, entry in zip(colors, synthetic_entries):
        synth_month = entry["summary"].monthly_mean.reindex(months)
        mapped_name = LEGEND_NAME_MAP.get(entry["name"], entry["name"])
        axes[0].plot(
            months,
            synth_month.values,
            marker="s",
            linewidth=2,
            color=color,
            alpha=0.85,
            label=mapped_name,
        )
    axes[0].set_title("Monthly mean inflow")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Discharge (m$^3$/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False, loc="best")

    # (b) Daily discharge density
    axes[1].hist(
        obs_clean,
        bins=bins,
        density=True,
        alpha=0.5,
        color="black",
        label=mapped_historical,
    )
    for color, entry in zip(colors, synthetic_entries):
        series = entry["series"].dropna()
        axes[1].hist(
            series,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            alpha=0.9,
            label=LEGEND_NAME_MAP.get(entry["name"], entry["name"]),
        )
    axes[1].set_title("Daily discharge density")
    axes[1].set_xlabel("Discharge (m$^3$/s)")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False, loc="best")
    axes[1].set_xlim(left=0, right=40000)

    # (c) Basic statistics (mean, std dev, skewness, kurtosis)
    stat_labels = ["Mean", "Std Dev", "Skewness", "Kurtosis"]
    metric_keys = ["mean", "std", "skew", "kurtosis"]
    hist_basic = hist_summary.basic_stats
    hist_vals = [hist_basic.get(key, np.nan) for key in metric_keys]

    primary_idx = np.array([0, 1])
    secondary_idx = np.array([2, 3])
    width = 0.8 / (len(synthetic_entries) + 1)
    x_positions = np.arange(len(stat_labels))

    def _collect(metric_index: int) -> list:
        values = [hist_vals[metric_index]]
        for entry in synthetic_entries:
            synth_basic = entry["summary"].basic_stats
            values.append(synth_basic.get(metric_keys[metric_index], np.nan))
        return [v for v in values if np.isfinite(v)]

    left_axis_values = _collect(0) + _collect(1)
    right_axis_values = _collect(2) + _collect(3)

    def _limits(values: list, padding: float = 0.08, top_extra: float = 0.14) -> Tuple[float, float]:
        if not values:
            return (0.0, 1.0)
        v_min, v_max = min(values), max(values)
        if v_min == v_max:
            delta = abs(v_min) * 0.1 + 0.1
            return (v_min - delta, v_max + delta * (1.0 + top_extra))
        span = v_max - v_min
        lower = v_min - span * padding
        upper = v_max + span * (padding + top_extra)
        return (lower, upper)

    def _format_value(metric_index: int, value: float) -> str:
        if not np.isfinite(value):
            return "--"
        if metric_index in primary_idx:
            return f"{value:.0f}"
        return f"{value:.2f}"

    def _diff_ratio(hist_value: float, synth_value: float) -> Optional[str]:
        if not (np.isfinite(hist_value) and np.isfinite(synth_value)):
            return None
        if hist_value == 0:
            return None
        ratio = (synth_value - hist_value) / hist_value * 100
        sign = "+" if ratio >= 0 else ""
        return f"({sign}{ratio:.1f}%)"

    ax_stats = axes[2]
    ax_stats_right = ax_stats.twinx()

    left_limits = _limits(left_axis_values)
    right_limits = _limits(right_axis_values)

    # Historical bars
    left_hist_bars = ax_stats.bar(
        primary_idx - 0.5 * width,
        [hist_vals[idx] for idx in primary_idx],
        width,
        color="black",
        alpha=0.7,
        label=mapped_historical,
        zorder=3,
    )
    right_hist_bars = ax_stats_right.bar(
        secondary_idx - 0.5 * width,
        [hist_vals[idx] for idx in secondary_idx],
        width,
        color="black",
        alpha=0.7,
        label=mapped_historical,
        zorder=3,
    )

    # Synthetic bars and annotations
    left_synth_containers = []
    right_synth_containers = []
    for idx, (color, entry) in enumerate(zip(colors, synthetic_entries)):
        synth_basic = entry["summary"].basic_stats
        synth_vals = [synth_basic.get(key, np.nan) for key in metric_keys]
        offset = -0.5 * width + (idx + 1) * width

        mapped_name = LEGEND_NAME_MAP.get(entry["name"], entry["name"])
        left_container = ax_stats.bar(
            primary_idx + offset,
            [synth_vals[i] for i in primary_idx],
            width,
            color=color,
            alpha=0.85,
            label=mapped_name,
            zorder=2,
        )
        right_container = ax_stats_right.bar(
            secondary_idx + offset,
            [synth_vals[i] for i in secondary_idx],
            width,
            color=color,
            alpha=0.85,
            label=mapped_name,
            zorder=2,
        )
        left_synth_containers.append(left_container)
        right_synth_containers.append(right_container)

    ax_stats.set_xticks(x_positions)
    ax_stats.set_xticklabels(stat_labels)
    ax_stats.set_ylabel("Mean / Std Dev (m$^3$/s)")
    ax_stats_right.set_ylabel("Skewness / Kurtosis")
    ax_stats.set_title("Basic statistics")

    ax_stats.set_ylim(left_limits)
    ax_stats_right.set_ylim(right_limits)

    # Historical reference lines within each metric cluster
    cluster_span = len(synthetic_entries) * width

    def _cluster_bounds(metric_position: float) -> Tuple[float, float]:
        left = metric_position - width
        right = metric_position + cluster_span
        # Ensure a minimal span in case there are no synthetic entries
        if right <= left:
            right = left + width
        return left, right

    for idx_val in primary_idx:
        if np.isfinite(hist_vals[idx_val]):
            xmin, xmax = _cluster_bounds(idx_val)
            ax_stats.hlines(
                hist_vals[idx_val],
                xmin,
                xmax,
                colors="red",
                linestyles="--",
                linewidth=1.2,
                alpha=0.8,
                zorder=1,
            )
    for idx_val in secondary_idx:
        if np.isfinite(hist_vals[idx_val]):
            xmin, xmax = _cluster_bounds(idx_val)
            ax_stats_right.hlines(
                hist_vals[idx_val],
                xmin,
                xmax,
                colors="red",
                linestyles="--",
                linewidth=1.2,
                alpha=0.8,
                zorder=1,
            )

    def _annotate(container, axis, metric_indices, is_synthetic: bool = False):
        y_limits = axis.get_ylim()
        span = y_limits[1] - y_limits[0]
        value_offset = span * 0.015
        ratio_offset = span * 0.085
        for bar, metric_index in zip(container.patches, metric_indices):
            value = bar.get_height()
            label = _format_value(metric_index, value)
            if label == "--":
                continue
            x = bar.get_x() + bar.get_width() / 2
            axis.text(
                x,
                value + value_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=fontsize - 6.5,
                color="black",
            )
            if is_synthetic:
                hist_value = hist_vals[metric_index]
                ratio = _diff_ratio(hist_value, value)
                if ratio:
                    axis.text(
                        x,
                        value + ratio_offset,
                        ratio,
                        ha="center",
                        va="bottom",
                        fontsize=fontsize - 7.5,
                        color="darkgreen",
                    )

    # Annotate historical bars
    _annotate(left_hist_bars, ax_stats, primary_idx)
    _annotate(right_hist_bars, ax_stats_right, secondary_idx)

    # Annotate synthetic bars
    for container in left_synth_containers:
        _annotate(container, ax_stats, primary_idx, is_synthetic=True)
    for container in right_synth_containers:
        _annotate(container, ax_stats_right, secondary_idx, is_synthetic=True)

    ax_stats.grid(True, axis="y", alpha=0.3)

    legend_handles = [Patch(color="black", alpha=0.7, label=mapped_historical)]
    legend_labels = [mapped_historical]
    for color, entry in zip(colors, synthetic_entries):
        legend_handles.append(Patch(color=color, alpha=0.85))
        legend_labels.append(LEGEND_NAME_MAP.get(entry["name"], entry["name"]))
    ax_stats.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=1,
    )

    # (d) QQ plot
    if obs_clean.size > 0:
        probs = np.linspace(0.01, 0.99, min(200, obs_clean.size))
        hist_quant = np.quantile(obs_clean, probs)
        axes[3].plot(
            [0, 40000],
            [0, 40000],
            "k--",
            linewidth=1.5,
            label="45° reference",
        )
        for color, entry in zip(colors, synthetic_entries):
            syn_clean = entry["series"].dropna()
            if syn_clean.empty:
                continue
            local_probs = np.linspace(0.01, 0.99, min(200, syn_clean.size, obs_clean.size))
            hist_q = np.quantile(obs_clean, local_probs)
            synth_q = np.quantile(syn_clean, local_probs)
            axes[3].scatter(
                hist_q,
                synth_q,
                color=color,
                s=22,
                alpha=0.85,
                label=LEGEND_NAME_MAP.get(entry["name"], entry["name"]),
                edgecolors="none",
            )
    else:
        axes[3].text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=axes[3].transAxes,
        )
    axes[3].set_title("Q-Q plot: quantile comparison")
    axes[3].set_xlabel("Historical quantile (m$^3$/s)")
    axes[3].set_ylabel("Synthetic quantile (m$^3$/s)")
    axes[3].set_xlim(left=0, right=40000)
    axes[3].set_ylim(bottom=0, top=40000)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(frameon=False, loc="best")

    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    for label, ax in zip(subplot_labels, axes):
        ax.text(
            -0.12,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
        )

    png_path = figure_dir / "Figure7.png"
    eps_path = figure_dir / "Figure7.eps"
    plt.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(eps_path, format="eps", dpi=300)
    plt.close(fig)
    return png_path
