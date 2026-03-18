"""
Single-experiment analysis — the primary use case.

Given one dataset (simulated or real), produce:

  1. Treatment effect on EXISTING (returning) users  — CUPED-adjusted
  2. Treatment effect on NEW users                    — diff-in-means
  3. Overall average treatment effect (ATE)           — population-weighted

Each with: τ̂, SE, 95% CI, t-stat, p-value.

Usage from command line:
    python inference.py                          # simulate with defaults
    python inference.py --p_new_users 0.3        # override a parameter

Usage from Python:
    from inference import experiment_report
    report = experiment_report(data)             # your own DataFrame
    print(report)

    from inference import analyze                # simulate + report
    table = analyze(cfg, seed=42)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import SimulationConfig
from simulation import simulate_user_level_data
from estimators import (
    EstimatorResult,
    StratifiedResult,
    diff_in_means,
    cuped_estimator,
    cuped_stratified,
    bayesian_estimator,
)


# ═══════════════════════════════════════════════════════════════════════════
# Row builder
# ═══════════════════════════════════════════════════════════════════════════

def _result_to_row(name: str, r: EstimatorResult, method: str) -> Dict:
    return {
        "Group": name,
        "Method": method,
        "τ̂": r.tau_hat,
        "SE": r.se,
        "95% CI lower": r.ci_lower,
        "95% CI upper": r.ci_upper,
        "t-stat": r.t_stat,
        "p-value": r.p_value,
        "Signif": _stars(r.p_value),
        "n": r.n,
    }


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


# ═══════════════════════════════════════════════════════════════════════════
# Primary entry point: experiment report
# ═══════════════════════════════════════════════════════════════════════════

def experiment_report(
    data: pd.DataFrame,
    cfg: Optional[SimulationConfig] = None,
) -> pd.DataFrame:
    """
    Analyze a single experiment.

    Returns a DataFrame with one row per quantity of interest:
      - Existing (returning) users: CUPED if Y_pre available, else diff-in-means
      - New users:                  diff-in-means (only if new users exist)
      - Overall ATE:                population-weighted combination (or CUPED if no new users)

    Each row carries: τ̂, SE, 95% CI, t-stat, p-value, significance stars.

    Parameters
    ----------
    data : pd.DataFrame
        Must have columns 'D' (treatment 0/1), 'Y' (outcome), 'Y_pre' (pre-period, NaN for new).
    cfg : SimulationConfig, optional
        Only needed if you want the Bayesian estimator row.  If None, Bayesian row is skipped.
    """
    rows = []
    has_new = data["Y_pre"].isna().any()
    has_returning = data["Y_pre"].notna().any()

    if has_new and has_returning:
        strat = cuped_stratified(data)

        rows.append(_result_to_row(
            "Existing users", strat.returning, "CUPED (OLS)"))
        rows.append(_result_to_row(
            "New users", strat.new, "Diff-in-means"))
        rows.append(_result_to_row(
            "Overall ATE (pop-weighted)", strat.population, "Stratified CUPED"))
        rows.append(_result_to_row(
            "Overall ATE (precision-weighted)", strat.precision, "Stratified CUPED"))

    elif has_returning:
        res_cuped = cuped_estimator(data)
        rows.append(_result_to_row(
            "All users (no new users)", res_cuped, "CUPED (OLS)"))

    else:
        res_naive = diff_in_means(data, label="naive")
        rows.append(_result_to_row(
            "All users (no Y_pre)", res_naive, "Diff-in-means"))

    # Naive on full population for reference
    rows.append(_result_to_row(
        "Naive (all users, no CUPED)", diff_in_means(data, label="naive"), "Diff-in-means"))

    # Optional Bayesian
    if cfg is not None:
        rows.append(_result_to_row(
            "Bayesian (all users)", bayesian_estimator(data, cfg), "Normal-normal"))

    table = pd.DataFrame(rows).set_index("Group")
    return table


# ═══════════════════════════════════════════════════════════════════════════
# Full detail table (all estimators, flat)
# ═══════════════════════════════════════════════════════════════════════════

def inference_table(
    data: pd.DataFrame,
    cfg: SimulationConfig,
) -> pd.DataFrame:
    """
    Run all estimators on a single DataFrame and return a detailed table.
    This is the supplementary view — everything in one flat table.
    """
    rows = []
    rows.append(_result_to_row("Naive (all)", diff_in_means(data), "Diff-in-means"))

    has_new = data["Y_pre"].isna().any()
    if has_new:
        strat = cuped_stratified(data)
        rows.append(_result_to_row("CUPED — returning", strat.returning, "CUPED OLS"))
        rows.append(_result_to_row("Diff-in-means — new", strat.new, "Diff-in-means"))
        rows.append(_result_to_row("CUPED — pop-weighted", strat.population, "Stratified"))
        rows.append(_result_to_row("CUPED — prec-weighted", strat.precision, "Stratified"))
    else:
        rows.append(_result_to_row("CUPED (all)", cuped_estimator(data), "CUPED OLS"))

    rows.append(_result_to_row("Bayesian", bayesian_estimator(data, cfg), "Normal-normal"))
    return pd.DataFrame(rows).set_index("Group")


# ═══════════════════════════════════════════════════════════════════════════
# Console analysis
# ═══════════════════════════════════════════════════════════════════════════

_FMT = {
    "τ̂": "{:.6f}",
    "SE": "{:.6f}",
    "95% CI lower": "{:.6f}",
    "95% CI upper": "{:.6f}",
    "t-stat": "{:.3f}",
    "p-value": "{:.4f}",
    "n": "{:.0f}",
}
_FORMATTERS = {k: v.format for k, v in _FMT.items()}


def analyze(
    cfg: Optional[SimulationConfig] = None,
    seed: int = 42,
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Primary single-experiment workflow: simulate (or accept) data, print the report.

    Returns the experiment_report DataFrame.
    """
    if cfg is None:
        cfg = SimulationConfig()

    if data is None:
        rng = np.random.default_rng(seed)
        data = simulate_user_level_data(cfg, rng)

    n = len(data)
    n_treated = int(data["D"].sum())
    n_new = int(data["Y_pre"].isna().sum())
    n_ret = n - n_new

    print("=" * 78)
    print("  EXPERIMENT REPORT — Treatment Effect Analysis")
    print("=" * 78)
    print(f"\n  Total users     : {n:,}")
    print(f"  Treated / Control: {n_treated:,} / {n - n_treated:,}")
    print(f"  Returning users  : {n_ret:,}  ({100 * n_ret / n:.1f}%)")
    print(f"  New users        : {n_new:,}  ({100 * n_new / n:.1f}%)")

    if hasattr(cfg, "tau_true"):
        rho = cfg.effective_autocorrelation()
        print(f"\n  [DGP] true τ = {cfg.tau_true},  ρ(Y_pre, Y) ≈ {rho:.4f}")

    report = experiment_report(data, cfg)

    print("\n" + "─" * 78)
    print("  RESULTS")
    print("─" * 78 + "\n")
    print(report.to_string(formatters=_FORMATTERS))
    print("\n  Signif. codes:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")

    if n_new > 0:
        strat = cuped_stratified(data)
        print(f"\n  μ̂ (coefficient on Y_pre among returning users) = {strat.mu_hat:.4f}")
        print(f"  Population weights: w_ret = {n_ret/n:.3f},  w_new = {n_new/n:.3f}")

    print()
    return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Single-experiment treatment-effect analysis.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_users", type=int, default=10_000)
    p.add_argument("--tau_true", type=float, default=0.05)
    p.add_argument("--beta_pre", type=float, default=0.8)
    p.add_argument("--sigma_post", type=float, default=1.0)
    p.add_argument("--p_new_users", type=float, default=0.20)
    p.add_argument("--sigma_heterogeneity", type=float, default=0.0)
    p.add_argument("--prior_sd_tau", type=float, default=0.10)
    args = p.parse_args()

    cfg = SimulationConfig(
        n_users=args.n_users,
        tau_true=args.tau_true,
        beta_pre=args.beta_pre,
        sigma_post=args.sigma_post,
        p_new_users=args.p_new_users,
        sigma_heterogeneity=args.sigma_heterogeneity,
        prior_sd_tau=args.prior_sd_tau,
    )
    analyze(cfg, seed=args.seed)


if __name__ == "__main__":
    _cli()
