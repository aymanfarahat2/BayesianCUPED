"""
Configuration for the treatment effect simulation pipeline.
Tune autocorrelation, heterogeneous effects, and new-user fraction here.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationConfig:
    """Parameters for data generation and estimators."""

    # --- Sample ---
    n_users: int = 10_000
    p_treatment: float = 0.5
    random_seed: int = 123

    # --- Pre-period outcome Y_pre (the only covariate in CUPED) ---
    mu_pre: float = 1.0
    sigma_pre: float = 0.5

    # --- Autocorrelation (Y_pre vs post Y) ---
    # CUPED model: Y = α + β*Treat + μ*Y_pre. Stronger link => more variance reduction.
    # beta_pre: slope of Y_pre in DGP (post Y = beta_pre*Y_pre + ...).
    # sigma_post: residual std of post-period.
    beta_pre: float = 0.8
    sigma_post: float = 1.0

    # --- Treatment effect ---
    tau_true: float = 0.05   # population average treatment effect (PATE)

    # --- Heterogeneous treatment effect ---
    # tau_i = tau_true + sigma_heterogeneity * z_i, z_i ~ N(0, 1).
    # Set to 0 for homogeneous effect.
    sigma_heterogeneity: float = 0.0

    # --- New users (no pre-period observation) ---
    # Fraction with no Y_pre (missing); they still have D and Y.
    p_new_users: float = 0.0

    # --- Bayesian prior (normal-normal on tau) ---
    prior_mean_tau: float = 0.0
    prior_sd_tau: float = 0.1

    def effective_autocorrelation(self) -> float:
        """Approximate correlation between Y_pre and Y (under homogeneity, no D)."""
        var_pre = self.sigma_pre ** 2
        var_post = self.beta_pre ** 2 * var_pre + self.sigma_post ** 2
        cov = self.beta_pre * var_pre
        return cov / (np.sqrt(var_pre * var_post)) if var_post > 0 else 0.0
