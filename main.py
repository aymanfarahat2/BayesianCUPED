"""
Entry point for the treatment-effect evaluation pipeline.

Two modes:
    python main.py                 # Monte Carlo (1000 sims, with plots)
    python main.py --single        # single-dataset inference (no MC)
    python main.py --single --seed 99 --p_new_users 0.3
"""

import argparse

from config import SimulationConfig
from evaluation import run_simulations
from inference import analyze
from plots import plot_estimator_distributions, plot_summary_bars, plot_se_calibration


def _parse_args():
    p = argparse.ArgumentParser(description="Treatment-effect evaluation pipeline.")
    p.add_argument("--single", action="store_true",
                   help="Single-dataset inference (no Monte Carlo).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_sim", type=int, default=1000)
    p.add_argument("--n_users", type=int, default=10_000)
    p.add_argument("--tau_true", type=float, default=0.05)
    p.add_argument("--beta_pre", type=float, default=0.8)
    p.add_argument("--sigma_post", type=float, default=1.0)
    p.add_argument("--p_new_users", type=float, default=0.20)
    p.add_argument("--sigma_heterogeneity", type=float, default=0.0)
    p.add_argument("--prior_sd_tau", type=float, default=0.10)
    p.add_argument("--no_plots", action="store_true",
                   help="Skip plots (useful for headless runs).")
    return p.parse_args()


def main():
    args = _parse_args()

    cfg = SimulationConfig(
        n_users=args.n_users,
        p_treatment=0.5,
        tau_true=args.tau_true,
        mu_pre=1.0,
        sigma_pre=0.5,
        beta_pre=args.beta_pre,
        sigma_post=args.sigma_post,
        sigma_heterogeneity=args.sigma_heterogeneity,
        p_new_users=args.p_new_users,
        random_seed=args.seed,
        prior_mean_tau=0.0,
        prior_sd_tau=args.prior_sd_tau,
    )

    # ── Single-dataset mode ───────────────────────────────────────────
    if args.single:
        analyze(cfg, seed=args.seed)
        return

    # ── Monte Carlo mode ──────────────────────────────────────────────
    n_sim = args.n_sim
    df, summary = run_simulations(cfg, n_sim=n_sim)

    print("=" * 78)
    print("  Stratified CUPED — Monte Carlo Evaluation")
    print("=" * 78)
    print(f"\n  DGP:  Y = {cfg.beta_pre}·Y_pre + τ_i·D + ε,   ε ~ N(0, {cfg.sigma_post}²)")
    print(f"        Y_pre ~ N({cfg.mu_pre}, {cfg.sigma_pre}²)")
    print(f"        τ_i = {cfg.tau_true} + {cfg.sigma_heterogeneity}·z_i,  z_i ~ N(0,1)")
    print(f"        Pr(new user) = {cfg.p_new_users}")
    print(f"        ρ(Y_pre, Y) ≈ {cfg.effective_autocorrelation():.4f}")
    print(f"        n = {cfg.n_users},  Pr(D=1) = {cfg.p_treatment},  S = {n_sim}")
    print(f"\n  Bayesian prior: τ ~ N({cfg.prior_mean_tau}, {cfg.prior_sd_tau}²)")
    print()
    print(summary.to_string(float_format="{:.5f}".format))
    print()

    if not args.no_plots:
        plot_estimator_distributions(df, cfg)
        plot_summary_bars(summary)
        plot_se_calibration(df, cfg)


if __name__ == "__main__":
    main()
