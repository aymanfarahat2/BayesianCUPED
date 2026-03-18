"""
Monte Carlo evaluation harness for the Bayesian CUPED paper.

Runs B replications of all fast estimators (E1–E4), collects
bias, variance, RMSE, MAE, CI width, coverage, and relative efficiency.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import SimulationConfig
from simulation import simulate_user_level_data
from estimators import (
    EstimatorResult,
    naive_dim, cuped_ols, stratified_ols, eb_cuped, cuped_matching_vwatt,
)


def run_single_sim(
    cfg: SimulationConfig, rng: np.random.Generator,
) -> Dict[str, EstimatorResult]:
    """Simulate one dataset and run E1–E4."""
    data = simulate_user_level_data(cfg, rng)
    res_naive = naive_dim(data)
    res_cuped = cuped_ols(data)
    res_strat = stratified_ols(data)
    res_eb, _ = eb_cuped(data)
    return {
        res_naive.label: res_naive,
        res_cuped.label: res_cuped,
        res_strat.label: res_strat,
        res_eb.label: res_eb,
    }


def run_monte_carlo(
    cfg: SimulationConfig,
    B: int = 300,
    seed: Optional[int] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Run B replications and return a summary DataFrame.

    Columns: Estimator, Bias, Variance, RMSE, MAE, Mean_SE, Coverage, RE, Mean_CI_Width
    """
    base_seed = seed if seed is not None else cfg.random_seed
    tau_star = cfg.true_pate()

    records: Dict[str, List] = {}

    for b in range(B):
        if progress and (b + 1) % 50 == 0:
            print(f"  replication {b + 1}/{B}")
        rng = np.random.default_rng(base_seed + b)
        results = run_single_sim(cfg, rng)
        for label, r in results.items():
            if label not in records:
                records[label] = []
            records[label].append({
                "tau_hat": r.tau_hat,
                "se": r.se,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "covers": r.covers(tau_star),
            })

    # Summarize
    rows = []
    naive_var = None

    for label in ["Naive DIM", "CUPED OLS", "Stratified OLS", "EB CUPED"]:
        if label not in records:
            continue
        recs = records[label]
        tau_hats = np.array([r["tau_hat"] for r in recs])
        ses = np.array([r["se"] for r in recs])
        covers = np.array([r["covers"] for r in recs])
        ci_widths = np.array([r["ci_upper"] - r["ci_lower"] for r in recs])

        bias = float(np.mean(tau_hats - tau_star))
        variance = float(np.var(tau_hats, ddof=1))
        rmse = float(np.sqrt(bias**2 + variance))
        mae = float(np.mean(np.abs(tau_hats - tau_star)))
        mean_se = float(np.mean(ses))
        se_sd_ratio = mean_se / np.std(tau_hats, ddof=1) if np.std(tau_hats, ddof=1) > 0 else float("nan")
        coverage = float(np.mean(covers))
        ci_width = float(np.mean(ci_widths))

        if label == "Naive DIM":
            naive_var = variance
        re = naive_var / variance if (naive_var is not None and variance > 0) else 1.0

        rows.append({
            "Estimator": label,
            "Bias": bias,
            "Variance": variance,
            "RMSE": rmse,
            "MAE": mae,
            "Mean_SE": mean_se,
            "SE/SD": se_sd_ratio,
            "Coverage": coverage,
            "RE vs Naive": re,
            "Mean_CI_Width": ci_width,
        })

    return pd.DataFrame(rows).set_index("Estimator")
