"""
Plotting utilities for the Bayesian CUPED simulation study.

- Stratum-level shrinkage plot
- Estimator distribution histograms
- Summary bar charts (Bias, RMSE, Coverage, RE)
- CI forest plot (single-experiment)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "Naive DIM": "#8c8c8c",
    "CUPED OLS": "#2196F3",
    "Stratified OLS": "#4CAF50",
    "EB CUPED": "#FF9800",
    "Bayesian MCMC": "#9C27B0",
}


def _get_color(label: str) -> str:
    for key, c in COLORS.items():
        if key in label:
            return c
    return "#333333"


# ===================================================================
# Stratum-level shrinkage plot
# ===================================================================
def plot_shrinkage(
    stratum_df: pd.DataFrame,
    true_effects: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Show raw vs. EB-shrunk stratum estimates, with optional true effects."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    K = len(stratum_df)
    x = np.arange(K)

    ax.scatter(x, stratum_df["τ̂_raw"], marker="o", s=80, color="#2196F3",
               label="Raw τ̂_k", zorder=3)
    ax.scatter(x, stratum_df["τ̂_EB"], marker="s", s=80, color="#FF9800",
               label="EB-shrunk τ̂_k", zorder=3)

    for i in range(K):
        ax.annotate("", xy=(x[i], stratum_df["τ̂_EB"].iloc[i]),
                     xytext=(x[i], stratum_df["τ̂_raw"].iloc[i]),
                     arrowprops=dict(arrowstyle="->", color="#999", lw=1.2))

    if true_effects is not None:
        ax.scatter(x, true_effects, marker="D", s=60, color="#E53935",
                   label="True τ_k", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(stratum_df["Group"].values)
    ax.set_ylabel("Treatment effect")
    ax.set_title("James–Stein Shrinkage: Raw → EB Estimates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ===================================================================
# CI forest plot (single experiment)
# ===================================================================
def plot_forest(
    summary_df: pd.DataFrame,
    tau_true: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Forest plot of estimator CIs from the experiment report."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    labels = summary_df.index.tolist()
    n = len(labels)
    y_pos = np.arange(n)

    for i, label in enumerate(labels):
        row = summary_df.loc[label]
        color = _get_color(label)
        ax.errorbar(row["τ̂"], i,
                     xerr=[[row["τ̂"] - row["95% CI lower"]],
                            [row["95% CI upper"] - row["τ̂"]]],
                     fmt="o", color=color, capsize=4, markersize=7)

    if tau_true is not None:
        ax.axvline(tau_true, color="#E53935", linestyle="--", lw=1.5, label=f"True PATE = {tau_true:.3f}")
        ax.legend()

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Treatment effect (τ̂)")
    ax.set_title("95% Confidence Intervals — Experiment Report")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    return ax


# ===================================================================
# Estimator distribution histograms (Monte Carlo)
# ===================================================================
def plot_distributions(
    mc_results: Dict[str, List[float]],
    tau_true: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlapping histograms of τ̂ across B replications."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    for label, values in mc_results.items():
        color = _get_color(label)
        ax.hist(values, bins=40, alpha=0.45, label=label, color=color, density=True)

    ax.axvline(tau_true, color="#E53935", linestyle="--", lw=2, label=f"True PATE = {tau_true:.3f}")
    ax.set_xlabel("τ̂")
    ax.set_ylabel("Density")
    ax.set_title("Sampling Distributions (Monte Carlo)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ===================================================================
# Summary bar charts
# ===================================================================
def plot_summary_bars(
    summary: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Bar charts for key summary metrics from Monte Carlo evaluation.
    summary: DataFrame indexed by Estimator with columns like Bias, RMSE, Coverage, RE.
    """
    if metrics is None:
        metrics = ["RMSE", "Bias", "Coverage", "RE vs Naive"]

    available = [m for m in metrics if m in summary.columns]
    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4.5))
    if n_metrics == 1:
        axes = [axes]

    labels = summary.index.tolist()
    x = np.arange(len(labels))
    bar_colors = [_get_color(l) for l in labels]

    for ax, metric in zip(axes, available):
        vals = summary[metric].values
        ax.bar(x, vals, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(metric, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        if metric == "Coverage":
            ax.axhline(0.95, color="#E53935", linestyle="--", lw=1, label="Nominal 0.95")
            ax.legend(fontsize=7)
        elif metric == "RE vs Naive":
            ax.axhline(1.0, color="#999", linestyle="--", lw=1)

    fig.suptitle("Monte Carlo Summary", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


# ===================================================================
# SE calibration scatter
# ===================================================================
def plot_se_calibration(
    mc_records: Dict[str, List[dict]],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter of analytic SE vs. |τ̂ - τ*| across MC replications."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for label, recs in mc_records.items():
        ses = [r["se"] for r in recs]
        abs_err = [abs(r["tau_hat"] - r.get("tau_true", 0)) for r in recs]
        color = _get_color(label)
        ax.scatter(ses, abs_err, alpha=0.3, s=12, color=color, label=label)

    lim = ax.get_xlim()
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="45° line")
    ax.set_xlabel("Analytic SE")
    ax.set_ylabel("|τ̂ − τ*|")
    ax.set_title("SE Calibration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax
