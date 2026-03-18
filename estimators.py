"""
Five estimators from Farahat (2026) — Bayesian CUPED paper.

E1. Naive DIM:          Y_T_bar - Y_C_bar
E2. CUPED OLS:          OLS of Y on (1, D, X) — targets VWATT
E3. Stratified OLS:     OLS with stratum fixed effects — targets VWATT
E4. Empirical Bayes:    James-Stein shrinkage across strata — targets PATE
E5. Bayesian MCMC:      Hierarchical model τ_k ~ N(μ_τ, σ²_τ) — targets PATE

Every estimator returns an EstimatorResult with point estimate, SE, CI, p-value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config import SimulationConfig


ALPHA = 0.05
Z_CRIT = sp_stats.norm.ppf(1 - ALPHA / 2)   # ≈ 1.96


def _two_sided_p(z: float) -> float:
    return float(2 * sp_stats.norm.sf(abs(z)))


# ---------------------------------------------------------------------------
# EstimatorResult: standardized output for every estimator
# ---------------------------------------------------------------------------
@dataclass
class EstimatorResult:
    label: str
    tau_hat: float
    se: float
    n: int
    ci_lower: float = field(init=False)
    ci_upper: float = field(init=False)
    p_value: float = field(init=False)
    rejects_null: bool = field(init=False)

    def __post_init__(self):
        self.ci_lower = self.tau_hat - Z_CRIT * self.se
        self.ci_upper = self.tau_hat + Z_CRIT * self.se
        z = self.tau_hat / self.se if self.se > 0 else float("inf")
        self.p_value = _two_sided_p(z) if math.isfinite(z) else 0.0
        self.rejects_null = self.p_value < ALPHA

    @property
    def t_stat(self) -> float:
        return self.tau_hat / self.se if self.se > 0 else float("inf")

    def covers(self, tau_true: float) -> bool:
        return self.ci_lower <= tau_true <= self.ci_upper


# ---------------------------------------------------------------------------
# Stratum-level results (for EB CUPED and Bayesian)
# ---------------------------------------------------------------------------
@dataclass
class StratumResult:
    """Holds stratum-level estimates for the hierarchical estimators."""
    tau_k_raw: np.ndarray          # raw within-stratum DIM
    se_k: np.ndarray               # SE of within-stratum DIM
    s2_k: np.ndarray               # variance of within-stratum DIM
    n_k: np.ndarray                # stratum sizes
    lambda_k: np.ndarray           # VWATT weights (precision-weighted)
    w_k: np.ndarray                # population weights (n_k / n)
    tau_k_shrunk: Optional[np.ndarray] = None
    B_k: Optional[np.ndarray] = None
    mu_tau_hat: Optional[float] = None
    sigma2_tau_hat: Optional[float] = None


# ====================================================================
# E1: Naive Difference-in-Means
# ====================================================================
def naive_dim(data: pd.DataFrame) -> EstimatorResult:
    """ȳ_T - ȳ_C with Welch SE."""
    t = data.loc[data["D"] == 1, "Y"]
    c = data.loc[data["D"] == 0, "Y"]
    tau_hat = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return EstimatorResult("Naive DIM", tau_hat, se, len(data))


# ====================================================================
# E2: CUPED OLS  (targets VWATT)
# ====================================================================
def cuped_ols(data: pd.DataFrame) -> EstimatorResult:
    """OLS: Y = α + β·D + γ·X + ε. The coefficient β targets the VWATT."""
    df = data.dropna(subset=["X"]).copy()
    n = len(df)
    Y = df["Y"].values
    D = df["D"].values
    X = df["X"].values
    Z = np.column_stack([np.ones(n), D, X])
    beta_hat, residuals, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    resid = Y - Z @ beta_hat
    sigma2 = np.sum(resid**2) / (n - 3)
    var_beta = sigma2 * np.linalg.inv(Z.T @ Z)
    se_beta = np.sqrt(var_beta[1, 1])
    return EstimatorResult("CUPED OLS", beta_hat[1], se_beta, n)


# ====================================================================
# E3: Stratified OLS (stratum fixed effects)
# ====================================================================
def stratified_ols(data: pd.DataFrame) -> EstimatorResult:
    """OLS with stratum dummies: Y = Σ_k α_k·1_{s=k} + β·D + ε."""
    df = data.loc[data["stratum"] >= 0].copy()
    n = len(df)
    K = df["stratum"].nunique()
    Y = df["Y"].values
    D = df["D"].values
    strata = df["stratum"].values

    # Build design matrix: K dummies (no intercept) + treatment
    Z = np.zeros((n, K + 1))
    for k in range(K):
        Z[strata == k, k] = 1.0
    Z[:, K] = D

    beta_hat, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    resid = Y - Z @ beta_hat
    sigma2 = np.sum(resid**2) / (n - K - 1)
    var_beta = sigma2 * np.linalg.inv(Z.T @ Z)
    se_beta = np.sqrt(var_beta[K, K])
    return EstimatorResult("Stratified OLS", beta_hat[K], se_beta, n)


# ====================================================================
# Stratum-level helpers
# ====================================================================
def _compute_stratum_stats(data: pd.DataFrame) -> StratumResult:
    """Compute within-stratum DIM, SE, and both VWATT and PATE weights."""
    df = data.loc[data["stratum"] >= 0].copy()
    strata = sorted(df["stratum"].unique())
    K = len(strata)
    n_total = len(df)

    tau_k = np.zeros(K)
    se_k = np.zeros(K)
    s2_k = np.zeros(K)
    n_k = np.zeros(K)
    pk_arr = np.zeros(K)

    for idx, k in enumerate(strata):
        sk = df[df["stratum"] == k]
        t = sk.loc[sk["D"] == 1, "Y"]
        c = sk.loc[sk["D"] == 0, "Y"]
        n_T, n_C = len(t), len(c)
        n_k[idx] = len(sk)
        pk_arr[idx] = n_T / n_k[idx] if n_k[idx] > 0 else 0.5

        tau_k[idx] = t.mean() - c.mean() if (n_T > 0 and n_C > 0) else 0.0
        var_T = t.var(ddof=1) if n_T > 1 else 0.0
        var_C = c.var(ddof=1) if n_C > 1 else 0.0
        s2_k[idx] = (var_T / max(n_T, 1)) + (var_C / max(n_C, 1))
        se_k[idx] = np.sqrt(s2_k[idx])

    # VWATT (precision) weights: λ_k ∝ n_k * p_k * (1 - p_k)
    raw_lambda = n_k * pk_arr * (1 - pk_arr)
    lambda_k = raw_lambda / raw_lambda.sum() if raw_lambda.sum() > 0 else np.ones(K) / K

    # Population weights
    w_k = n_k / n_total if n_total > 0 else np.ones(K) / K

    return StratumResult(
        tau_k_raw=tau_k, se_k=se_k, s2_k=s2_k,
        n_k=n_k, lambda_k=lambda_k, w_k=w_k,
    )


# ====================================================================
# CUPED-as-matching (Theorem 1): explicit VWATT from stratum estimates
# ====================================================================
def cuped_matching_vwatt(data: pd.DataFrame) -> Tuple[EstimatorResult, StratumResult]:
    """Theorem 1: β̂_CUPED = Σ_k λ_k τ̂_k (precision-weighted, targets VWATT)."""
    sr = _compute_stratum_stats(data)
    tau_hat = float(sr.lambda_k @ sr.tau_k_raw)
    se = float(np.sqrt(np.sum(sr.lambda_k**2 * sr.s2_k)))
    return EstimatorResult("CUPED Matching (VWATT)", tau_hat, se, int(sr.n_k.sum())), sr


# ====================================================================
# E4: Empirical Bayes CUPED (Algorithm 1, targets PATE)
# ====================================================================
def eb_cuped(data: pd.DataFrame) -> Tuple[EstimatorResult, StratumResult]:
    """
    Closed-form EB CUPED (Algorithm 1 in the paper).

    1. Within-stratum DIM: τ̂_k, s²_k.
    2. Precision-weighted global mean: μ̂_τ = Σ(τ̂_k / s²_k) / Σ(1/s²_k).
    3. MoM heterogeneity:  σ̂²_τ = max(0, 1/(K-1) Σ(τ̂_k - μ̂_τ)² - 1/K Σ s²_k).
    4. Shrinkage: B_k = s²_k / (s²_k + σ̂²_τ).
    5. Shrunk estimate: τ̂_k^EB = (1 - B_k)τ̂_k + B_k μ̂_τ.
    6. PATE: τ̂_EB = Σ_k (n_k/n) τ̂_k^EB.
    """
    sr = _compute_stratum_stats(data)
    K = len(sr.tau_k_raw)
    inv_s2 = np.where(sr.s2_k > 0, 1.0 / sr.s2_k, 0.0)

    # Step 2: precision-weighted global mean
    mu_hat = float(np.sum(inv_s2 * sr.tau_k_raw) / np.sum(inv_s2)) if np.sum(inv_s2) > 0 else 0.0

    # Step 3: method-of-moments heterogeneity
    if K > 1:
        sigma2_tau = max(
            0.0,
            np.sum((sr.tau_k_raw - mu_hat)**2) / (K - 1) - np.mean(sr.s2_k),
        )
    else:
        sigma2_tau = 0.0

    # Step 4: shrinkage factors
    B_k = sr.s2_k / (sr.s2_k + sigma2_tau) if sigma2_tau > 0 else np.ones(K)

    # Step 5: shrunk estimates
    tau_k_eb = (1 - B_k) * sr.tau_k_raw + B_k * mu_hat

    # Step 6: population-weighted PATE
    tau_eb = float(sr.w_k @ tau_k_eb)

    # SE via delta-method (population-weighted combination of shrunk SEs)
    var_k_eb = (1 - B_k)**2 * sr.s2_k
    se_eb = float(np.sqrt(np.sum(sr.w_k**2 * var_k_eb)))

    # Store in StratumResult
    sr.tau_k_shrunk = tau_k_eb
    sr.B_k = B_k
    sr.mu_tau_hat = mu_hat
    sr.sigma2_tau_hat = sigma2_tau

    return EstimatorResult("EB CUPED", tau_eb, se_eb, int(sr.n_k.sum())), sr


# ====================================================================
# E5: Bayesian MCMC  (hierarchical model, targets PATE)
# ====================================================================
def bayesian_mcmc(
    data: pd.DataFrame,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
) -> Tuple[EstimatorResult, dict]:
    """
    Full Bayesian hierarchical model from Section 3.3 of the paper.

    τ_k ~ N(μ_τ, σ²_τ),  Y_{ik} ~ N(α_k + τ_k D_{ik}, σ²_y).
    Returns posterior mean of pop-weighted ATE and its posterior SD as SE.
    """
    import pymc as pm
    import arviz as az

    df = data.loc[data["stratum"] >= 0].copy()
    strata = sorted(df["stratum"].unique())
    K = len(strata)
    stratum_map = {s: i for i, s in enumerate(strata)}
    stratum_idx = df["stratum"].map(stratum_map).values.astype(int)
    treat = df["D"].values.astype(float)
    y = df["Y"].values.astype(float)
    strata_shares = np.array([np.mean(stratum_idx == k) for k in range(K)])

    with pm.Model() as model:
        mu_tau = pm.Normal("mu_tau", mu=0, sigma=1)
        sigma_tau = pm.Exponential("sigma_tau", lam=1)
        alpha_k = pm.Normal("alpha_k", mu=0, sigma=2, shape=K)
        tau_k = pm.Normal("tau_k", mu=mu_tau, sigma=sigma_tau, shape=K)
        sigma_y = pm.Exponential("sigma_y", lam=1)
        mu_y = alpha_k[stratum_idx] + tau_k[stratum_idx] * treat
        pm.Normal("y_obs", mu=mu_y, sigma=sigma_y, observed=y)
        pm.Deterministic("shrinkage", sigma_y**2 / (sigma_y**2 + sigma_tau**2))
        pm.Deterministic("pop_ate", pm.math.dot(strata_shares, tau_k))

        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept,
                          progressbar=True, return_inferencedata=True)

    pop_ate_samples = trace.posterior["pop_ate"].values.flatten()
    tau_hat = float(pop_ate_samples.mean())
    se = float(pop_ate_samples.std())

    # Summaries
    tau_k_post = trace.posterior["tau_k"].values.reshape(-1, K)
    summary_dict = {
        "trace": trace,
        "mu_tau_post_mean": float(trace.posterior["mu_tau"].values.mean()),
        "sigma_tau_post_mean": float(trace.posterior["sigma_tau"].values.mean()),
        "sigma_y_post_mean": float(trace.posterior["sigma_y"].values.mean()),
        "shrinkage_post_mean": float(trace.posterior["shrinkage"].values.mean()),
        "tau_k_post_mean": tau_k_post.mean(axis=0),
        "tau_k_post_sd": tau_k_post.std(axis=0),
        "tau_k_hdi": az.hdi(trace, var_names=["tau_k"], hdi_prob=0.94),
        "pop_ate_hdi": az.hdi(trace, var_names=["pop_ate"], hdi_prob=0.94),
        "rhat": az.rhat(trace),
        "ess": az.ess(trace),
    }

    return EstimatorResult("Bayesian MCMC", tau_hat, se, len(df)), summary_dict


# ====================================================================
# Convenience: run all fast estimators (E1–E4) on one dataset
# ====================================================================
def run_all_fast(data: pd.DataFrame) -> Dict[str, EstimatorResult]:
    """Run E1–E4 and return {label: EstimatorResult}."""
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
