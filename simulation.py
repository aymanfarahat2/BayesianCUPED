"""
User-level data simulation: pre/post revenue, treatment, optional new users and heterogeneity.
"""

import numpy as np
import pandas as pd

from config import SimulationConfig


def simulate_user_level_data(
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate user-level A/B test data.

    - Y_pre: pre-period outcome (NaN for new users). This is the only covariate in CUPED.
    - D: treatment assignment.
    - Y: post-period outcome.
    - is_new_user: 1 if no pre-period observation.

    DGP: Y = beta_pre * Y_pre + tau_i * D + noise. New users have Y_pre unobserved (NaN).

    Heterogeneity: tau_i = tau_true + sigma_heterogeneity * z_i, z_i ~ N(0,1). PATE = tau_true.
    """
    n = cfg.n_users

    D = rng.binomial(1, cfg.p_treatment, size=n).astype(np.float64)
    is_new = rng.binomial(1, cfg.p_new_users, size=n).astype(np.float64)
    Y_pre_full = rng.normal(cfg.mu_pre, cfg.sigma_pre, size=n)
    z = rng.standard_normal(n)
    tau_i = cfg.tau_true + cfg.sigma_heterogeneity * z
    noise = rng.normal(0.0, cfg.sigma_post, size=n)
    Y = cfg.beta_pre * Y_pre_full + tau_i * D + noise
    Y_pre = np.where(is_new == 1, np.nan, Y_pre_full)

    return pd.DataFrame({
        "D": D,
        "Y_pre": Y_pre,
        "Y": Y,
        "is_new_user": is_new,
    })
