"""
Plotting suite for the Monte Carlo evaluation.

1. Sampling distributions: histograms of τ̂ across simulations.
2. Bias / Variance / RMSE bar charts.
3. Coverage & SE-calibration diagnostics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SimulationConfig


# ═══════════════════════════════════════════════════════════════════════════
# 1. Histograms of τ̂ across simulations
# ═══════════════════════════════════════════════════════════════════════════

def plot_estimator_distributions(
    df: pd.DataFrame,
    cfg: SimulationConfig,
) -> plt.Figure:
    """Histograms of all tracked estimators with true τ reference line."""
    cols_labels = [
        ("naive_tau", "Naive diff-in-means"),
        ("cuped_population_tau", "CUPED (pop-weighted)"),
        ("cuped_precision_tau", "CUPED (precision-weighted)"),
        ("bayes_tau", "Bayesian posterior mean"),
    ]
    if cfg.p_new_users > 0:
        cols_labels.insert(2, ("cuped_returning_tau", "CUPED (returning only)"))
        cols_labels.insert(3, ("cuped_new_tau", "Diff-in-means (new only)"))

    n_panels = len(cols_labels)
    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharey=True, squeeze=False)
    axes_flat = axes.ravel()

    for ax, (col, title) in zip(axes_flat, cols_labels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].dropna()
        ax.hist(vals, bins=50, alpha=0.7, color="steelblue", edgecolor="white",
                linewidth=0.5)
        ax.axvline(cfg.tau_true, color="red", linestyle="--", linewidth=1.4,
                   label=f"τ = {cfg.tau_true}")
        ax.axvline(vals.mean(), color="orange", linestyle=":", linewidth=1.2,
                   label=f"mean = {vals.mean():.4f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("τ̂")
        ax.legend(fontsize=7)
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    axes_flat[0].set_ylabel("Frequency")
    fig.suptitle("Sampling Distributions of Estimators", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. Bias / RMSE / Coverage summary bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_summary_bars(summary: pd.DataFrame) -> plt.Figure:
    """Bar charts for key metrics: bias, RMSE, coverage, SE/SD ratio."""
    metrics = ["bias", "rmse", "coverage", "se_sd_ratio"]
    titles = ["Bias", "RMSE", "Coverage (95 % CI)", "SE / SD ratio"]
    refs = [0.0, None, 0.95, 1.0]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    fig, axes = plt.subplots(1, 4, figsize=(max(16, 4 * len(summary)), 4.5))
    for ax, metric, title, ref, col in zip(axes, metrics, titles, refs, colors):
        vals = summary[metric] if metric in summary.columns else pd.Series(dtype=float)
        bars = ax.bar(range(len(vals)), vals.values, color=col, edgecolor="black",
                      linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        if ref is not None:
            ax.axhline(ref, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. SE calibration: scatter of analytic SE vs. |τ̂ − τ_true|
# ═══════════════════════════════════════════════════════════════════════════

def plot_se_calibration(df: pd.DataFrame, cfg: SimulationConfig) -> plt.Figure:
    """
    For each main estimator: scatter of analytic SE vs. absolute error.
    Well-calibrated SE should have ~68 % of points below the diagonal.
    """
    estimators = ["naive", "cuped_population", "cuped_precision", "bayes"]
    labels = ["Naive", "CUPED pop", "CUPED prec", "Bayesian"]

    fig, axes = plt.subplots(1, len(estimators), figsize=(4.5 * len(estimators), 4),
                             squeeze=False)
    axes_flat = axes.ravel()

    for ax, est, lab in zip(axes_flat, estimators, labels):
        tc = f"{est}_tau"
        sc = f"{est}_se"
        if tc not in df.columns or sc not in df.columns:
            ax.set_visible(False)
            continue
        abs_err = np.abs(df[tc].values - cfg.tau_true)
        se = df[sc].values
        valid = np.isfinite(abs_err) & np.isfinite(se)
        ax.scatter(se[valid], abs_err[valid], s=4, alpha=0.25, color="steelblue")
        lim = max(se[valid].max(), abs_err[valid].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="SE = |error|")
        frac_below = float((abs_err[valid] <= se[valid]).mean())
        ax.set_title(f"{lab}  ({frac_below:.0%} below diag)", fontsize=9)
        ax.set_xlabel("Analytic SE")
        ax.set_ylabel("|τ̂ − τ|")
        ax.legend(fontsize=7)
    fig.suptitle("SE Calibration: analytic SE vs. absolute error", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig
