"""
CLI entry point for the Bayesian CUPED simulation pipeline.

Usage:
  python main.py                                # Monte Carlo, default regime
  python main.py --single                       # Single-experiment report
  python main.py --all_regimes                  # Full 21-regime sweep
  python main.py --regime worst_case --B 100    # Specific regime
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SimulationConfig, build_all_regimes


def _build_config(args: argparse.Namespace) -> SimulationConfig:
    regimes = build_all_regimes()
    if args.regime in regimes:
        cfg = regimes[args.regime]
    else:
        cfg = regimes["baseline"]

    if args.n is not None:
        cfg.n_users = args.n
    if args.rho is not None:
        cfg.rho = args.rho
    if args.K is not None:
        cfg.n_strata = args.K
        if cfg.stratum_effects is not None:
            from config import _het_effects
            cfg.stratum_effects = _het_effects(args.K)
    if args.sigma is not None:
        cfg.sigma = args.sigma
    if args.p is not None:
        cfg.p_treatment = args.p
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Bayesian CUPED pipeline")
    parser.add_argument("--single", action="store_true", help="Single-experiment report")
    parser.add_argument("--all_regimes", action="store_true", help="Full 21-regime sweep")
    parser.add_argument("--mcmc", action="store_true", help="Include Bayesian MCMC (slow)")
    parser.add_argument("--regime", type=str, default="baseline",
                        choices=list(build_all_regimes().keys()) + ["custom"])
    parser.add_argument("--B", type=int, default=300, help="Monte Carlo replications")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX tables")
    parser.add_argument("--csv", action="store_true", help="Save CSV")
    args = parser.parse_args()

    if args.all_regimes:
        from eval_full import run_all_regimes, format_tables
        full = run_all_regimes(B=args.B, seed=args.seed)
        tables = format_tables(full)

        for name, tbl in tables.items():
            print(f"\n{'='*80}\n{name}\n{'='*80}")
            fmt = "{:+.4f}".format if name == "Bias" else "{:.4f}".format
            print(tbl.to_string(float_format=fmt))

        if args.csv:
            full.to_csv("eval_results.csv", index=False)
            print("\nSaved eval_results.csv")

        if args.latex:
            from eval_full import to_latex_rmse, to_latex_bias, to_latex_re
            print("\n% LaTeX RMSE Table")
            print(to_latex_rmse(tables["RMSE"]))
            print("\n% LaTeX Bias Table")
            print(to_latex_bias(tables["Bias"]))
            print("\n% LaTeX RE Table")
            print(to_latex_re(tables["RE"]))
        return

    cfg = _build_config(args)
    cfg.random_seed = args.seed

    if args.single:
        from inference import analyze
        analyze(cfg, seed=args.seed, run_mcmc=args.mcmc)
    else:
        from evaluation import run_monte_carlo
        from plots import plot_summary_bars

        print(f"Running {args.B} Monte Carlo replications (regime: {args.regime})...")
        summary = run_monte_carlo(cfg, B=args.B, seed=args.seed)
        print("\n" + summary.to_string(float_format="{:.4f}".format))
        print(f"\nTrue PATE = {cfg.true_pate():.4f}")

        if not args.no_plots:
            fig = plot_summary_bars(summary)
            fig.savefig("mc_summary.png", dpi=150, bbox_inches="tight")
            print("\nPlot saved to mc_summary.png")


if __name__ == "__main__":
    main()
