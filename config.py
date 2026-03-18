"""
Configuration for the Bayesian CUPED simulation pipeline.

DGP: X_i ~ Gamma(2,1), discretized into K quantile-based strata,
     stratum-specific τ_k, Y = ρ·X + τ_{s(i)}·D + ε.

All 21 simulation regimes from Table 2 of Farahat (2026).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SimulationConfig:
    """All parameters for data generation, estimation, and evaluation."""

    # --- Sample ---
    n_users: int = 2_000
    p_treatment: float = 0.5
    random_seed: int = 42

    # --- Pre-period covariate ---
    x_gamma_shape: float = 2.0
    n_strata: int = 5

    # --- Autocorrelation ---
    rho: float = 0.5

    # --- Noise ---
    sigma: float = 1.0
    noise_dist: str = "normal"  # "normal", "t3", "lognormal"

    # --- Treatment effects ---
    tau_true: float = 0.20
    stratum_effects: Optional[List[float]] = None
    effect_scale: float = 1.0

    # --- New users ---
    p_new_users: float = 0.0

    # --- Bayesian prior ---
    prior_mean_tau: float = 0.0
    prior_sd_tau: float = 0.10

    def __post_init__(self):
        if self.stratum_effects is not None and len(self.stratum_effects) != self.n_strata:
            raise ValueError(
                f"stratum_effects length ({len(self.stratum_effects)}) "
                f"must equal n_strata ({self.n_strata})")

    def get_stratum_effects(self) -> np.ndarray:
        """Return K-length array of true stratum-level treatment effects."""
        if self.stratum_effects is not None:
            return np.array(self.stratum_effects) * self.effect_scale
        return np.full(self.n_strata, self.tau_true)

    def true_pate(self) -> float:
        """True population average treatment effect."""
        return float(self.get_stratum_effects().mean())

    def effective_autocorrelation(self) -> float:
        """Approximate ρ(X, Y) under the DGP (ignoring treatment)."""
        var_x = self.x_gamma_shape
        var_y = self.rho ** 2 * var_x + self.sigma ** 2
        cov = self.rho * var_x
        return cov / np.sqrt(var_x * var_y) if var_y > 0 else 0.0


# =====================================================================
# Heterogeneous effect helpers
# =====================================================================
_HET5 = [0.05, 0.10, 0.20, 0.30, 0.45]  # default K=5


def _het_effects(K: int) -> List[float]:
    """Spread heterogeneous effects from 0.05 to 0.45 across K strata."""
    if K == 5:
        return list(_HET5)
    return list(np.linspace(0.05, 0.45, K).round(4))


# =====================================================================
# All 21 regimes from Table 2 of the paper
# =====================================================================
def _base(**overrides) -> SimulationConfig:
    """Baseline with optional overrides."""
    kw = dict(
        n_users=2000, rho=0.5, n_strata=5, p_treatment=0.5,
        sigma=1.0, effect_scale=1.0, noise_dist="normal",
        stratum_effects=list(_HET5),
    )
    kw.update(overrides)
    if "n_strata" in overrides and "stratum_effects" not in overrides:
        kw["stratum_effects"] = _het_effects(kw["n_strata"])
    return SimulationConfig(**kw)


def build_all_regimes() -> "OrderedDict[str, SimulationConfig]":
    """Return the 21 regimes from Table 2 of Farahat (2026)."""
    regimes: OrderedDict[str, SimulationConfig] = OrderedDict()

    # --- Autocorrelation ---
    regimes["baseline"]       = _base()
    regimes["high_autocorr"]  = _base(rho=0.9)
    regimes["low_autocorr"]   = _base(rho=0.1)

    # --- Distribution ---
    regimes["heavy_tail"]     = _base(noise_dist="t3")
    regimes["skewed"]         = _base(noise_dist="lognormal")

    # --- Heterogeneity ---
    regimes["homogeneous"]    = SimulationConfig(
        n_users=2000, rho=0.5, n_strata=5, p_treatment=0.5,
        sigma=1.0, tau_true=0.20, stratum_effects=None,
    )
    regimes["heterogeneous"]  = _base(rho=0.7)  # same effects, stronger covariate
    regimes["large_effects"]  = _base(effect_scale=3.0)
    regimes["small_effects"]  = _base(effect_scale=0.2)

    # --- Sample size ---
    regimes["small_sample"]   = _base(n_users=300)
    regimes["medium_sample"]  = _base(n_users=1000)
    regimes["large_sample"]   = _base(n_users=10_000)

    # --- Stratification ---
    regimes["few_strata"]     = _base(n_strata=3)
    regimes["many_strata"]    = _base(n_strata=10)

    # --- Treatment balance ---
    regimes["balanced"]       = _base(p_treatment=0.50)  # same as baseline
    regimes["imbalanced"]     = _base(p_treatment=0.20)
    regimes["sparse"]         = _base(p_treatment=0.05)

    # --- Noise scale ---
    regimes["low_noise"]      = _base(sigma=0.3)
    regimes["high_noise"]     = _base(sigma=3.0)

    # --- Compound ---
    regimes["worst_case"]     = SimulationConfig(
        n_users=300, rho=0.1, n_strata=5, p_treatment=0.20,
        sigma=2.0, noise_dist="t3",
        stratum_effects=list(_HET5),
    )
    regimes["best_case"]      = SimulationConfig(
        n_users=5000, rho=0.95, n_strata=5, p_treatment=0.5,
        sigma=0.3,
        stratum_effects=list(_HET5),
    )

    return regimes


# Convenience shortcuts
def paper_default()      -> SimulationConfig: return build_all_regimes()["baseline"]
def paper_homogeneous()  -> SimulationConfig: return build_all_regimes()["homogeneous"]
def paper_worst_case()   -> SimulationConfig: return build_all_regimes()["worst_case"]
def paper_best_case()    -> SimulationConfig: return build_all_regimes()["best_case"]
