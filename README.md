# Bayesian CUPED: Simulation Study

**Companion code for:**
Farahat, A. (2026). *Bayesian CUPED: Hierarchical Shrinkage for Variance Reduction in Online Experiments.*

---

## Overview

This repository implements and evaluates five estimators for treatment effects in online A/B tests, building on the insight that CUPED (Controlled Experiments Using Pre-Experiment Data) is equivalent to a precision-weighted matching estimator when the pre-experiment covariate is discretized into strata.

### Estimators

| # | Estimator | Estimand | Method |
|---|-----------|----------|--------|
| E1 | Naive DIM | ATE | ȳ_T − ȳ_C |
| E2 | CUPED OLS | VWATT | OLS: Y ~ D + X |
| E3 | Stratified OLS | VWATT | OLS with stratum fixed effects |
| E4 | **EB CUPED** (proposed) | **PATE** | James–Stein shrinkage, closed-form |
| E5 | **Bayesian MCMC** (proposed) | **PATE** | Hierarchical τ_k ~ N(μ_τ, σ²_τ) |

### Key Results (B=300 Monte Carlo, 21 regimes)

| Estimator | Mean RMSE | Mean RE vs Naive | Key advantage |
|-----------|-----------|------------------|---------------|
| CUPED OLS | 0.0781 | 2.50 | Best when ρ is large |
| EB CUPED | 0.0777 | 1.59 | Targets PATE; robust under heavy tails |
| Stratified OLS | 0.0809 | 1.60 | Unbiased within-stratum |
| Naive DIM | 0.0919 | 1.00 | Baseline |

- Under **heavy tails** and **worst-case** conditions, EB CUPED is the only fast estimator that provides meaningful gains
- In the **worst case** (t₃, ρ=0.1, n=300, p=0.2, σ=2): EB CUPED achieves RMSE 0.448 vs CUPED OLS 0.527
- Classical CUPED targets **VWATT**, not **PATE** — a distinction that matters under heterogeneous treatment effects

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Single-Experiment Analysis (Primary Use Case)

```bash
python main.py --single
```

Or in Python:

```python
from config import paper_default
from simulation import simulate_user_level_data
from inference import experiment_report
import numpy as np

cfg = paper_default()
rng = np.random.default_rng(42)
data = simulate_user_level_data(cfg, rng)
stratum_df, summary_df = experiment_report(data, cfg)
print(summary_df)
```

### Monte Carlo Simulation

```bash
# Default: 300 replications, baseline regime
python main.py

# Custom: worst-case regime, 100 replications
python main.py --regime worst_case --B 100

# Full 21-regime sweep (Tables 3-5 from the paper)
python eval_full.py --B 300 --csv --latex

# All regimes via main.py
python main.py --all_regimes --B 300

# With MCMC (slow)
python main.py --single --mcmc
```

### Interactive Notebook

```bash
jupyter notebook experiment.ipynb
```

---

## Project Structure

```
AymanSimulation/
├── config.py              # SimulationConfig + all 21 regimes (build_all_regimes)
├── simulation.py           # DGP: X~Gamma, quantile strata, Y = ρX + τD + ε
├── estimators.py           # E1–E5 with SE, CI, p-value
├── inference.py            # Single-experiment report (experiment_report)
├── evaluation.py           # Monte Carlo harness (run_monte_carlo)
├── eval_full.py            # Full 21-regime sweep → Tables 3-5
├── plots.py                # Shrinkage, forest, distribution, summary plots
├── main.py                 # CLI entry point (--single, --all_regimes)
├── experiment.ipynb         # Interactive Jupyter notebook
├── simulation_study.tex     # LaTeX companion document
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data-Generating Process

```
X_i ~ Gamma(2, 1)              # Pre-experiment covariate
s(i) = quantile_bin(X_i, K)    # K quantile-based strata
D_i ~ Bernoulli(p)             # Treatment assignment
Y_i = ρ·X_i + τ_{s(i)}·D_i + ε_i   # Outcome

ε_i ~ Normal(0, σ²)  or  t_3  or  LogNormal(centered)
```

Default stratum effects: τ_k ∈ {0.05, 0.10, 0.20, 0.30, 0.45}.

---

## Using with Your Own Data

```python
import pandas as pd
from inference import experiment_report

# Your data must have columns: D (0/1), X (pre-period), Y (outcome), stratum (int >= 0)
df = pd.read_csv("your_experiment.csv")
stratum_df, summary_df = experiment_report(df)
print(summary_df)
```

If you don't have strata, discretize your pre-period metric:

```python
df["stratum"] = pd.qcut(df["X"], q=5, labels=False)
```

---

## Citation

```bibtex
@article{farahat2026bayesian,
  title={Bayesian {CUPED}: Hierarchical Shrinkage for Variance Reduction in Online Experiments},
  author={Farahat, Ayman},
  year={2026}
}
```

---

## License

MIT
