"""
Data-generating process from Farahat (2026).

X_i ~ Gamma(shape, 1), discretized into K quantile-based strata.
D_i ~ Bernoulli(p).
Y_i = rho * X_i + tau_{s(i)} * D_i + epsilon_i.

Strata are assigned by pd.qcut on X.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SimulationConfig


def simulate_user_level_data(
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = cfg.n_users
    K = cfg.n_strata
    tau_k = cfg.get_stratum_effects()

    # Pre-period covariate
    X = rng.gamma(cfg.x_gamma_shape, 1.0, size=n)

    # Treatment assignment (independent of X)
    D = rng.binomial(1, cfg.p_treatment, size=n).astype(np.float64)

    # Quantile-based stratification
    stratum = pd.qcut(X, K, labels=False, duplicates="drop")
    stratum = np.asarray(stratum, dtype=int)

    # Stratum-level treatment effects
    tau_i = tau_k[stratum]

    # Noise
    if cfg.noise_dist == "t3":
        eps = rng.standard_t(df=3, size=n) * cfg.sigma
    elif cfg.noise_dist == "lognormal":
        raw = rng.lognormal(0, 0.5, size=n)
        eps = (raw - np.exp(0.5 * 0.5**2)) * cfg.sigma
    else:
        eps = rng.normal(0.0, cfg.sigma, size=n)

    # Outcome
    Y = cfg.rho * X + tau_i * D + eps

    # Observed X (NaN for new users)
    is_new = rng.binomial(1, cfg.p_new_users, size=n).astype(np.float64)
    X_obs = np.where(is_new == 1, np.nan, X)
    stratum_obs = np.where(is_new == 1, -1, stratum)

    return pd.DataFrame({
        "D": D,
        "X": X_obs,
        "Y": Y,
        "stratum": stratum_obs.astype(int),
        "is_new_user": is_new,
    })
