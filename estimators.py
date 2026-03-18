"""
Treatment-effect estimators with full inference (τ̂, SE, CI, p-value).

CUPED model (single covariate):
    Y = α + β·D + μ·Y_pre

When new users (no Y_pre) are present we use stratified estimation:
  Stratum R (returning): OLS on Y = α + β·D + μ·Y_pre  →  (β̂_R, SE_R)
  Stratum N (new):       diff-in-means on raw Y          →  (τ̂_N, SE_N)

  Population-weighted PATE:
      τ̂_pop = w_R·β̂_R + w_N·τ̂_N        where w_R = n_R/n, w_N = n_N/n
      SE(τ̂_pop) = √(w_R²·SE_R² + w_N²·SE_N²)

  Precision-weighted (minimum-variance):
      τ̂_prec = (β̂_R/V_R + τ̂_N/V_N) / (1/V_R + 1/V_N)
      SE(τ̂_prec) = √(1 / (1/V_R + 1/V_N))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import SimulationConfig

ALPHA = 0.05
Z_CRIT = 1.959964  # scipy.stats.norm.ppf(1 - ALPHA/2)


def _two_sided_p(z: float) -> float:
    """Two-sided p-value from standard normal, without scipy."""
    return math.erfc(abs(z) / math.sqrt(2.0))


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EstimatorResult:
    """Full inference output for one estimator on one dataset."""
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


@dataclass
class StratifiedResult:
    """Full inference from stratified CUPED with new users."""
    returning: EstimatorResult
    new: Optional[EstimatorResult]
    population: EstimatorResult
    precision: EstimatorResult
    mu_hat: float  # OLS coefficient on Y_pre


# ═══════════════════════════════════════════════════════════════════════════
# Diff-in-means
# ═══════════════════════════════════════════════════════════════════════════

def diff_in_means(data: pd.DataFrame, y_col: str = "Y",
                  label: str = "naive") -> EstimatorResult:
    """τ̂ = Ȳ_T − Ȳ_C  with Welch SE."""
    t = data.loc[data["D"] == 1, y_col]
    c = data.loc[data["D"] == 0, y_col]
    n_t, n_c = len(t), len(c)
    tau = float(t.mean() - c.mean())
    var = float(t.var(ddof=1) / max(n_t, 1) + c.var(ddof=1) / max(n_c, 1))
    return EstimatorResult(label=label, tau_hat=tau, se=np.sqrt(var), n=n_t + n_c)


# ═══════════════════════════════════════════════════════════════════════════
# CUPED (OLS)
# ═══════════════════════════════════════════════════════════════════════════

def _ols_cuped(Y: np.ndarray, D: np.ndarray,
               Y_pre: np.ndarray) -> tuple[float, float, float]:
    """
    OLS: Y = α + β·D + μ·Y_pre.
    Returns (β̂, μ̂, SE(β̂)).
    """
    n = len(Y)
    X = np.column_stack([np.ones(n), D, Y_pre])
    coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
    beta_hat = float(coeffs[1])
    mu_hat = float(coeffs[2])

    resid = Y - X @ coeffs
    s2 = float(np.sum(resid ** 2) / max(n - 3, 1))
    try:
        cov_matrix = s2 * np.linalg.inv(X.T @ X)
        se_beta = float(np.sqrt(max(cov_matrix[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        se_beta = float("inf")

    return beta_hat, mu_hat, se_beta


def cuped_estimator(data: pd.DataFrame) -> EstimatorResult:
    """CUPED regression on all users with non-missing Y_pre. Returns EstimatorResult."""
    mask = data["Y_pre"].notna()
    n = int(mask.sum())
    if n < 4:
        return diff_in_means(data, label="cuped_fallback")
    Y = data.loc[mask, "Y"].values
    D = data.loc[mask, "D"].values
    Y_pre_arr = data.loc[mask, "Y_pre"].values
    beta_hat, _mu, se_beta = _ols_cuped(Y, D, Y_pre_arr)
    return EstimatorResult(label="cuped", tau_hat=beta_hat, se=se_beta, n=n)


# ═══════════════════════════════════════════════════════════════════════════
# Stratified CUPED (returning + new users)
# ═══════════════════════════════════════════════════════════════════════════

def cuped_stratified(data: pd.DataFrame) -> StratifiedResult:
    """
    Stratified estimation when some users have no Y_pre.

    Returning stratum → OLS CUPED.
    New stratum        → diff-in-means.
    Combined           → population-weighted and precision-weighted.

    Returns a StratifiedResult carrying full inference for every level.
    """
    returning = data["Y_pre"].notna()
    n = len(data)
    n_ret = int(returning.sum())
    n_new = n - n_ret
    data_ret = data.loc[returning]
    data_new = data.loc[~returning]

    # --- Returning: CUPED regression → (β̂, SE) ---
    if n_ret >= 4:
        Y_r = data_ret["Y"].values
        D_r = data_ret["D"].values
        Yp_r = data_ret["Y_pre"].values
        beta_hat, mu_hat, se_ret = _ols_cuped(Y_r, D_r, Yp_r)
        res_ret = EstimatorResult(
            label="cuped_returning", tau_hat=beta_hat, se=se_ret, n=n_ret)
    else:
        res_ret = diff_in_means(data_ret, label="cuped_returning")
        mu_hat = 0.0

    # --- New: diff-in-means → (τ̂, SE) ---
    if n_new >= 2 and data_new["D"].nunique() == 2:
        res_new = diff_in_means(data_new, label="dim_new")
    else:
        res_new = EstimatorResult(
            label="dim_new", tau_hat=0.0, se=float("inf"), n=n_new)

    # --- Population-weighted PATE ---
    w_ret = n_ret / n if n > 0 else 0.5
    w_new = n_new / n if n > 0 else 0.5
    tau_pop = w_ret * res_ret.tau_hat + w_new * res_new.tau_hat
    se_pop = np.sqrt(w_ret ** 2 * res_ret.se ** 2 + w_new ** 2 * res_new.se ** 2)
    res_pop = EstimatorResult(
        label="cuped_population", tau_hat=tau_pop, se=se_pop, n=n)

    # --- Precision-weighted (minimum variance) ---
    v_ret = res_ret.se ** 2
    v_new = res_new.se ** 2
    prec_ret = 1.0 / v_ret if v_ret > 0 and np.isfinite(v_ret) else 0.0
    prec_new = 1.0 / v_new if v_new > 0 and np.isfinite(v_new) else 0.0
    total_prec = prec_ret + prec_new
    if total_prec > 0:
        tau_prec = (prec_ret * res_ret.tau_hat + prec_new * res_new.tau_hat) / total_prec
        se_prec = np.sqrt(1.0 / total_prec)
    else:
        tau_prec = tau_pop
        se_prec = se_pop
    res_prec = EstimatorResult(
        label="cuped_precision", tau_hat=tau_prec, se=se_prec, n=n)

    return StratifiedResult(
        returning=res_ret, new=res_new,
        population=res_pop, precision=res_prec,
        mu_hat=mu_hat,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian normal-normal
# ═══════════════════════════════════════════════════════════════════════════

def bayesian_estimator(data: pd.DataFrame, cfg: SimulationConfig,
                       y_col: str = "Y",
                       label: str = "bayes") -> EstimatorResult:
    """
    Normal-normal conjugate update.
    Prior: τ ~ N(m₀, v₀).  Likelihood: τ̂|τ ~ N(τ, V̂).
    Posterior: τ|data ~ N(μ_post, σ²_post).
    Returns posterior mean as point estimate, posterior SD as SE.
    """
    res_freq = diff_in_means(data, y_col)
    tau_hat = res_freq.tau_hat
    var_hat = res_freq.se ** 2
    if var_hat <= 0:
        var_hat = 1e-12
    v0 = cfg.prior_sd_tau ** 2
    m0 = cfg.prior_mean_tau
    var_post = 1.0 / (1.0 / v0 + 1.0 / var_hat)
    mu_post = var_post * (m0 / v0 + tau_hat / var_hat)
    return EstimatorResult(
        label=label, tau_hat=float(mu_post),
        se=float(np.sqrt(var_post)), n=res_freq.n)
