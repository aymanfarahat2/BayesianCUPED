# AymanSimulation — Treatment Effect Evaluation Pipeline

Modular pipeline to compare **naive**, **CUPED**, and **Bayesian** treatment effect estimators with configurable autocorrelation, heterogeneous effects, and new users.

## Layout

| File | Role |
|------|------|
| `config.py` | `SimulationConfig`: all parameters (autocorrelation, heterogeneity, new-user %, etc.) |
| `simulation.py` | `simulate_user_level_data()`: Y_pre, Y (pre/post outcome), D, new-user flag |
| `estimators.py` | `naive_diff_in_means`, `cuped_estimator_with_new_users`, `bayesian_normal_normal_estimator` |
| `evaluation.py` | `run_single_simulation`, `run_simulations`, `summarize_estimator` |
| `plots.py` | `plot_estimator_distributions`, `plot_bias_variance_rmse` |
| `main.py` | Entry point: build config, run 1000 sims, print summary, show plots |

## Parameters (in `config.py` or `main.py`)

- **Autocorrelation (pre vs post):** `beta_pre` (slope), `sigma_post` (noise). Higher `beta_pre` or lower `sigma_post` ⇒ stronger correlation.
- **Heterogeneous treatment:** `sigma_heterogeneity`. Treatment effect per user: `tau_i = tau_true + sigma_heterogeneity * N(0,1)`. Set to `0` for homogeneous.
- **New users:** `p_new_users` (fraction with no pre-period). CUPED is modified to estimate μ on returning users only; new users use raw Y in the diff-in-means.

## CUPED model (no other covariates)

We posit **Y = α + β·Treat + μ·Y_pre**. There are no other covariates; the only covariate is the pre-period outcome Y_pre.

- **(α, β, μ)** are estimated by OLS on **returning** users (non-missing Y_pre). **β** is the treatment effect; **μ** is the coefficient on Y_pre.
- **Returning:** Y_cuped = Y − μ·(Y_pre − mean(Y_pre)).
- **New:** Y_cuped = Y (no adjustment).
- **τ̂** = mean(Y_cuped | D=1) − mean(Y_cuped | D=0) over **all** users.

## Run

```bash
cd /Users/afarahat/AymanPython/CondaWork/AymanSimulation
pip install -r requirements.txt   # if needed
python main.py
```

Or: `python treatment_evaluation_pipeline.py` (delegates to `main.py`).
