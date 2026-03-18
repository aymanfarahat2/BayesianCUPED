"""
Single-experiment analysis — the primary use case.

Given one dataset, produce a full experiment report with:
  - Stratum-level estimates (raw, shrunk, shrinkage factor)
  - Overall ATE (VWATT via CUPED, PATE via EB CUPED)
  - Point estimate, SE, CI, t-stat, p-value for every row
  - Naive DIM as benchmark

Reference: Farahat (2026), Sections 2–4.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import SimulationConfig
from simulation import simulate_user_level_data
from estimators import (
    EstimatorResult, StratumResult,
    naive_dim, cuped_ols, stratified_ols,
    cuped_matching_vwatt, eb_cuped, bayesian_mcmc,
    _compute_stratum_stats,
)


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


def _result_row(label: str, method: str, r: EstimatorResult) -> dict:
    return {
        "Group": label,
        "Method": method,
        "τ̂": r.tau_hat,
        "SE": r.se,
        "95% CI lower": r.ci_lower,
        "95% CI upper": r.ci_upper,
        "t-stat": r.t_stat,
        "p-value": r.p_value,
        "Sig.": _stars(r.p_value),
        "n": r.n,
    }


# ===================================================================
# Experiment report — the main entry point
# ===================================================================
def experiment_report(
    data: pd.DataFrame,
    cfg: Optional[SimulationConfig] = None,
    run_mcmc: bool = False,
    mcmc_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Produce a comprehensive experiment report.

    Rows:
      - Stratum-level raw and EB-shrunk estimates
      - VWATT (CUPED matching, Theorem 1)
      - PATE (EB CUPED, Algorithm 1)
      - Naive DIM (benchmark)
      - Optionally: Bayesian MCMC

    Returns a styled-ready DataFrame.
    """
    rows = []

    # --- E1: Naive ---
    res_naive = naive_dim(data)

    # --- E2: CUPED OLS ---
    res_cuped = cuped_ols(data)

    # --- E3: Stratified OLS ---
    res_strat = stratified_ols(data)

    # --- CUPED matching (Theorem 1, VWATT) ---
    res_match, sr_match = cuped_matching_vwatt(data)

    # --- E4: EB CUPED (Algorithm 1, PATE) ---
    res_eb, sr_eb = eb_cuped(data)

    # Stratum-level rows
    K = len(sr_eb.tau_k_raw)
    for k in range(K):
        rows.append({
            "Group": f"Stratum {k}",
            "Method": "Within-stratum DIM",
            "τ̂_raw": sr_eb.tau_k_raw[k],
            "SE_raw": sr_eb.se_k[k],
            "τ̂_EB": sr_eb.tau_k_shrunk[k],
            "B_k": sr_eb.B_k[k],
            "λ_k (VWATT wt)": sr_match.lambda_k[k],
            "w_k (pop wt)": sr_eb.w_k[k],
            "n_k": int(sr_eb.n_k[k]),
        })

    stratum_df = pd.DataFrame(rows)

    # Summary rows
    summary_rows = []
    summary_rows.append(_result_row("VWATT (Theorem 1)", "CUPED matching", res_match))
    summary_rows.append(_result_row("CUPED OLS (E2)", "OLS Y ~ D + X", res_cuped))
    summary_rows.append(_result_row("Stratified OLS (E3)", "OLS Y ~ D + strata", res_strat))
    summary_rows.append(_result_row("PATE — EB CUPED (E4)", "EB shrinkage", res_eb))
    summary_rows.append(_result_row("Naive DIM (E1)", "ȳ_T - ȳ_C", res_naive))

    if run_mcmc:
        kw = mcmc_kwargs or {}
        res_mcmc, mcmc_detail = bayesian_mcmc(data, **kw)
        summary_rows.append(_result_row("PATE — Bayesian MCMC (E5)", "Hierarchical", res_mcmc))

    summary_df = pd.DataFrame(summary_rows).set_index("Group")

    return stratum_df, summary_df


# ===================================================================
# Console utility
# ===================================================================
def analyze(
    cfg: Optional[SimulationConfig] = None,
    seed: Optional[int] = None,
    data: Optional[pd.DataFrame] = None,
    run_mcmc: bool = False,
):
    """Simulate or accept data, then print the experiment report."""
    if cfg is None:
        from config import paper_default
        cfg = paper_default()

    if data is None:
        rng = np.random.default_rng(seed or cfg.random_seed)
        data = simulate_user_level_data(cfg, rng)

    stratum_df, summary_df = experiment_report(data, cfg, run_mcmc=run_mcmc)

    print("\n" + "=" * 80)
    print("EXPERIMENT REPORT — Bayesian CUPED Pipeline")
    print("=" * 80)

    print("\n--- Stratum-level estimates ---")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(stratum_df.to_string(index=False))

    print(f"\n  μ̂_τ (precision-weighted global mean) = "
          f"{stratum_df['τ̂_raw'].values @ (1/stratum_df['SE_raw'].values**2) / (1/stratum_df['SE_raw'].values**2).sum():.4f}")

    print("\n--- Summary estimators ---")
    fmt_dict = {"τ̂": "{:.6f}", "SE": "{:.6f}",
                "95% CI lower": "{:.6f}", "95% CI upper": "{:.6f}",
                "t-stat": "{:.3f}", "p-value": "{:.4f}", "n": "{:.0f}"}
    print(summary_df.to_string(float_format="{:.4f}".format))

    if cfg is not None:
        print(f"\n  True PATE = {cfg.true_pate():.4f}")
        print(f"  ρ(X, Y) ≈ {cfg.effective_autocorrelation():.3f}")

    print("\n  Significance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
    print("=" * 80)

    return stratum_df, summary_df


# ===================================================================
# CLI entry
# ===================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Single-experiment analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--mcmc", action="store_true")
    args = parser.parse_args()

    cfg = SimulationConfig(
        n_users=args.n, rho=args.rho, n_strata=args.K, sigma=args.sigma,
        stratum_effects=[0.05, 0.10, 0.20, 0.30, 0.45][:args.K],
    )
    analyze(cfg, seed=args.seed, run_mcmc=args.mcmc)
