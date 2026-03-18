"""
Full evaluation script: run all 21 regimes × B replications.

Produces Tables 3 (RMSE), 4 (Bias), 5 (RE) from the paper,
plus coverage, SE/SD ratio, and CI width.

Usage:
  python eval_full.py              # B=300, all 21 regimes
  python eval_full.py --B 100      # quick run
  python eval_full.py --latex      # also write LaTeX tables to stdout
  python eval_full.py --csv        # save results to CSV
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from config import SimulationConfig, build_all_regimes
from simulation import simulate_user_level_data
from estimators import (
    EstimatorResult, naive_dim, cuped_ols, stratified_ols, eb_cuped,
)

ESTIMATOR_ORDER = ["Naive DIM", "CUPED OLS", "Stratified OLS", "EB CUPED"]
REGIME_LABELS = {
    "baseline":       "baseline",
    "high_autocorr":  "high autocorr",
    "low_autocorr":   "low autocorr",
    "heavy_tail":     "heavy tail",
    "skewed":         "skewed",
    "homogeneous":    "homogeneous",
    "heterogeneous":  "heterogeneous",
    "large_effects":  "large effects",
    "small_effects":  "small effects",
    "small_sample":   "small sample",
    "medium_sample":  "medium sample",
    "large_sample":   "large sample",
    "few_strata":     "few strata",
    "many_strata":    "many strata",
    "balanced":       "balanced",
    "imbalanced":     "imbalanced",
    "sparse":         "sparse treatment",
    "low_noise":      "low noise",
    "high_noise":     "high noise",
    "worst_case":     "worst case",
    "best_case":      "best case",
}


def _run_one(cfg: SimulationConfig, rng: np.random.Generator) -> Dict[str, EstimatorResult]:
    data = simulate_user_level_data(cfg, rng)
    r1 = naive_dim(data)
    r2 = cuped_ols(data)
    r3 = stratified_ols(data)
    r4, _ = eb_cuped(data)
    return {r1.label: r1, r2.label: r2, r3.label: r3, r4.label: r4}


def evaluate_regime(
    name: str, cfg: SimulationConfig, B: int, base_seed: int,
) -> pd.DataFrame:
    """Run B replications for one regime; return one-row-per-estimator summary."""
    tau_star = cfg.true_pate()
    records: Dict[str, List[dict]] = {e: [] for e in ESTIMATOR_ORDER}

    for b in range(B):
        rng = np.random.default_rng(base_seed + b)
        results = _run_one(cfg, rng)
        for label in ESTIMATOR_ORDER:
            r = results[label]
            records[label].append({
                "tau_hat": r.tau_hat, "se": r.se,
                "covers": r.covers(tau_star),
                "ci_width": r.ci_upper - r.ci_lower,
            })

    rows = []
    naive_var = None
    for label in ESTIMATOR_ORDER:
        recs = records[label]
        th = np.array([r["tau_hat"] for r in recs])
        ses = np.array([r["se"] for r in recs])
        cov = np.array([r["covers"] for r in recs])
        ciw = np.array([r["ci_width"] for r in recs])

        bias = float(np.mean(th - tau_star))
        var_ = float(np.var(th, ddof=1))
        rmse = float(np.sqrt(bias**2 + var_))
        mae_ = float(np.mean(np.abs(th - tau_star)))
        mse_ = float(np.mean(ses))
        sd_  = float(np.std(th, ddof=1))
        sesd = mse_ / sd_ if sd_ > 0 else float("nan")
        covg = float(np.mean(cov))
        mciw = float(np.mean(ciw))

        if label == "Naive DIM":
            naive_var = var_
        re = naive_var / var_ if (naive_var and var_ > 0) else 1.0

        rows.append({
            "Regime": name,
            "Estimator": label,
            "Bias": bias,
            "Variance": var_,
            "RMSE": rmse,
            "MAE": mae_,
            "Mean_SE": mse_,
            "SE/SD": sesd,
            "Coverage": covg,
            "RE": re,
            "CI_Width": mciw,
        })

    return pd.DataFrame(rows)


def run_all_regimes(B: int = 300, seed: int = 42) -> pd.DataFrame:
    """Evaluate all 21 regimes and concatenate results."""
    regimes = build_all_regimes()
    frames = []
    total = len(regimes)

    for i, (name, cfg) in enumerate(regimes.items(), 1):
        t0 = time.time()
        print(f"[{i:2d}/{total}] {REGIME_LABELS.get(name, name):20s} "
              f"(n={cfg.n_users}, ρ={cfg.rho}, K={cfg.n_strata}, "
              f"p={cfg.p_treatment}, σ={cfg.sigma}, "
              f"noise={cfg.noise_dist}, PATE={cfg.true_pate():.3f}) ... ",
              end="", flush=True)
        df = evaluate_regime(name, cfg, B, seed)
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# =====================================================================
# Pivot into paper-style tables
# =====================================================================
def _pivot(full: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to regime × estimator table for one metric."""
    tbl = full.pivot(index="Regime", columns="Estimator", values=metric)
    regime_order = list(REGIME_LABELS.keys())
    tbl = tbl.reindex([r for r in regime_order if r in tbl.index])
    tbl = tbl[ESTIMATOR_ORDER]
    tbl.index = [REGIME_LABELS.get(r, r) for r in tbl.index]
    return tbl


def format_tables(full: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return {metric_name: pivot_table} for RMSE, Bias, RE, Coverage."""
    tables = {}
    for m in ["RMSE", "Bias", "RE", "Coverage", "SE/SD", "CI_Width"]:
        tables[m] = _pivot(full, m)

    # Add a "Mean" row to RMSE and Bias tables
    for m in ["RMSE", "Bias"]:
        if m == "Bias":
            tables[m].loc["Mean |Bias|"] = tables[m].abs().mean()
        else:
            tables[m].loc["Mean"] = tables[m].mean()

    return tables


def _bold_min_row(row: pd.Series) -> List[str]:
    """Bold the minimum value in each row (for RMSE)."""
    min_val = row.min()
    return [f"\\textbf{{{v:.4f}}}" if v == min_val else f"{v:.4f}" for v in row]


def to_latex_rmse(tbl: pd.DataFrame) -> str:
    """Generate LaTeX for the RMSE table (Table 3)."""
    lines = []
    lines.append(r"\begin{tabular}{@{}l rrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Regime & E1 Naive & E2 CUPED OLS & E3 Strat.\ OLS & E4 EB CUPED \\")
    lines.append(r"\midrule")
    for regime in tbl.index:
        row = tbl.loc[regime]
        min_val = row.min()
        cells = []
        for v in row:
            s = f"{v:.4f}"
            if v == min_val:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(f"{regime} & {' & '.join(cells)} \\\\")
        if regime in ("baseline", "low autocorr", "skewed",
                      "small effects", "large sample",
                      "many strata", "sparse treatment", "high noise"):
            pass
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def to_latex_bias(tbl: pd.DataFrame) -> str:
    """Generate LaTeX for the Bias table (Table 4)."""
    lines = []
    lines.append(r"\begin{tabular}{@{}l rrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Regime & E1 Naive & E2 CUPED OLS & E3 Strat.\ OLS & E4 EB CUPED \\")
    lines.append(r"\midrule")
    for regime in tbl.index:
        row = tbl.loc[regime]
        cells = [f"{v:+.4f}" if regime != "Mean |Bias|" else f"{v:.4f}" for v in row]
        lines.append(f"{regime} & {' & '.join(cells)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def to_latex_re(tbl: pd.DataFrame) -> str:
    """Generate LaTeX for the RE table (Table 5)."""
    lines = []
    lines.append(r"\begin{tabular}{@{}l rrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Regime & E2 CUPED OLS & E3 Strat.\ OLS & E4 EB CUPED \\")
    lines.append(r"\midrule")
    for regime in tbl.index:
        row = tbl.loc[regime]
        cells = [f"{row['CUPED OLS']:.2f}",
                 f"{row['Stratified OLS']:.2f}",
                 f"{row['EB CUPED']:.2f}"]
        lines.append(f"{regime} & {' & '.join(cells)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full 21-regime evaluation")
    parser.add_argument("--B", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latex", action="store_true", help="Print LaTeX tables")
    parser.add_argument("--csv", action="store_true", help="Save to eval_results.csv")
    args = parser.parse_args()

    full = run_all_regimes(B=args.B, seed=args.seed)
    tables = format_tables(full)

    print("\n" + "=" * 90)
    print("TABLE 3: RMSE across 21 simulation regimes")
    print("=" * 90)
    print(tables["RMSE"].to_string(float_format="{:.4f}".format))

    print("\n" + "=" * 90)
    print("TABLE 4: Bias across 21 simulation regimes")
    print("=" * 90)
    print(tables["Bias"].to_string(float_format="{:+.4f}".format))

    print("\n" + "=" * 90)
    print("TABLE 5: Relative efficiency vs Naive DIM")
    print("=" * 90)
    print(tables["RE"].to_string(float_format="{:.2f}".format))

    print("\n" + "=" * 90)
    print("Coverage (nominal = 0.95)")
    print("=" * 90)
    print(tables["Coverage"].to_string(float_format="{:.3f}".format))

    print("\n" + "=" * 90)
    print("SE/SD ratio (ideal = 1.00)")
    print("=" * 90)
    print(tables["SE/SD"].to_string(float_format="{:.3f}".format))

    if args.csv:
        full.to_csv("eval_results.csv", index=False)
        print("\nSaved eval_results.csv")

    if args.latex:
        print("\n\n% ====== LaTeX: Table 3 (RMSE) ======")
        print(to_latex_rmse(tables["RMSE"]))
        print("\n% ====== LaTeX: Table 4 (Bias) ======")
        print(to_latex_bias(tables["Bias"]))
        print("\n% ====== LaTeX: Table 5 (RE) ======")
        print(to_latex_re(tables["RE"]))
