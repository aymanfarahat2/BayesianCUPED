"""
Monte Carlo evaluation harness.

For each simulation we record (τ̂, SE, covers_true) for every estimator.
Across S simulations the summary table reports:

  Bias         = mean(τ̂) − τ_true
  Empirical SD = sd(τ̂)                  (Monte Carlo variability)
  Mean SE      = mean(SE)                (average analytic SE)
  SE/SD ratio  = Mean SE / Empirical SD  (should be ≈ 1 if SE is well-calibrated)
  RMSE         = √(mean((τ̂ − τ_true)²))
  Coverage     = fraction of 95 % CIs that contain τ_true  (should be ≈ 0.95)
  Power        = fraction of 95 % CIs that exclude 0
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import SimulationConfig
from simulation import simulate_user_level_data
from estimators import (
    EstimatorResult,
    diff_in_means,
    cuped_estimator,
    cuped_stratified,
    bayesian_estimator,
)


# ═══════════════════════════════════════════════════════════════════════════
# Single simulation
# ═══════════════════════════════════════════════════════════════════════════

def run_single_simulation(
    cfg: SimulationConfig, seed: int,
) -> Dict[str, EstimatorResult]:
    """
    Generate one dataset and compute every estimator.

    Returns a dict of EstimatorResult keyed by estimator name.
    When p_new_users == 0, the stratified results collapse to plain CUPED.
    """
    rng = np.random.default_rng(seed)
    data = simulate_user_level_data(cfg, rng)

    results: Dict[str, EstimatorResult] = {}

    # 1. Naive diff-in-means (all users)
    results["naive"] = diff_in_means(data, label="naive")

    # 2. Bayesian (all users, on raw Y)
    results["bayes"] = bayesian_estimator(data, cfg, label="bayes")

    # 3. CUPED / Stratified CUPED
    has_new = data["Y_pre"].isna().any()

    if has_new:
        strat = cuped_stratified(data)
        results["cuped_returning"] = strat.returning
        results["cuped_new"] = strat.new
        results["cuped_population"] = strat.population
        results["cuped_precision"] = strat.precision
    else:
        res_cuped = cuped_estimator(data)
        results["cuped_returning"] = res_cuped
        results["cuped_population"] = EstimatorResult(
            label="cuped_population", tau_hat=res_cuped.tau_hat,
            se=res_cuped.se, n=res_cuped.n)
        results["cuped_precision"] = EstimatorResult(
            label="cuped_precision", tau_hat=res_cuped.tau_hat,
            se=res_cuped.se, n=res_cuped.n)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Flatten one simulation into a row dict
# ═══════════════════════════════════════════════════════════════════════════

def _flatten(results: Dict[str, EstimatorResult],
             cfg: SimulationConfig) -> Dict[str, float]:
    """Flatten estimator results into a flat dict for DataFrame storage."""
    row: Dict[str, float] = {"tau_true": cfg.tau_true}
    for name, r in results.items():
        row[f"{name}_tau"] = r.tau_hat
        row[f"{name}_se"] = r.se
        row[f"{name}_covers"] = float(r.covers(cfg.tau_true))
        row[f"{name}_rejects"] = float(r.rejects_null)
    return row


# ═══════════════════════════════════════════════════════════════════════════
# Summary across simulations
# ═══════════════════════════════════════════════════════════════════════════

def _summarize_one(tau_col: np.ndarray, se_col: np.ndarray,
                   covers_col: np.ndarray, rejects_col: np.ndarray,
                   tau_true: float) -> Dict[str, float]:
    valid = np.isfinite(tau_col)
    tau = tau_col[valid]
    se = se_col[valid]
    cov = covers_col[valid]
    rej = rejects_col[valid]
    if len(tau) == 0:
        return {k: float("nan") for k in
                ["bias", "empirical_sd", "mean_se", "se_sd_ratio",
                 "rmse", "coverage", "power"]}
    bias = float(tau.mean() - tau_true)
    emp_sd = float(tau.std(ddof=1))
    mean_se = float(se.mean())
    ratio = mean_se / emp_sd if emp_sd > 0 else float("inf")
    rmse = float(np.sqrt(((tau - tau_true) ** 2).mean()))
    coverage = float(cov.mean())
    power = float(rej.mean())
    return {
        "bias": bias,
        "empirical_sd": emp_sd,
        "mean_se": mean_se,
        "se_sd_ratio": ratio,
        "rmse": rmse,
        "coverage": coverage,
        "power": power,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════════════

def run_simulations(
    cfg: SimulationConfig,
    n_sim: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run *n_sim* Monte Carlo replications.

    Returns
    -------
    df : pd.DataFrame
        One row per simulation; columns  <est>_tau, <est>_se, <est>_covers, <est>_rejects.
    summary : pd.DataFrame
        One row per estimator; columns bias, empirical_sd, mean_se, se_sd_ratio, rmse, coverage, power.
    """
    base_rng = np.random.default_rng(cfg.random_seed)
    seeds = base_rng.integers(0, 2**32 - 1, size=n_sim)

    rows: List[Dict[str, float]] = []
    for i, s in enumerate(seeds, start=1):
        res = run_single_simulation(cfg, int(s))
        row = _flatten(res, cfg)
        row["sim"] = float(i)
        rows.append(row)

    df = pd.DataFrame(rows)

    estimator_names = ["naive", "cuped_returning", "cuped_population",
                       "cuped_precision", "bayes"]
    if cfg.p_new_users > 0:
        estimator_names.insert(2, "cuped_new")

    summary_rows = {}
    for name in estimator_names:
        tc = f"{name}_tau"
        sc = f"{name}_se"
        cc = f"{name}_covers"
        rc = f"{name}_rejects"
        if tc in df.columns:
            summary_rows[name] = _summarize_one(
                df[tc].values, df[sc].values,
                df[cc].values, df[rc].values,
                cfg.tau_true)

    summary = pd.DataFrame.from_dict(summary_rows, orient="index")
    return df, summary
