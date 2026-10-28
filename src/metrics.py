"""
Statistical Evaluation Metrics and Significance Testing.

Includes:
  - R², RMSE, MAE, normalized RMSE (nRMSE %), and Bias
  - Non-parametric Bootstrap Confidence Intervals (95% CI)
  - Paired Permutation Significance Tests
  - Multi-target evaluation table reporting
"""

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
except ImportError:
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    def mean_squared_error(y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    def mean_absolute_error(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

def calculate_regression_metrics(y_true, y_pred):
    """
    Computes standard remote sensing regression metrics:
      R², RMSE, MAE, nRMSE (%), and Bias.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid) or len(y_true[valid]) < 2:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "nRMSE_%": np.nan, "Bias": np.nan}

    yt = y_true[valid]
    yp = y_pred[valid]

    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mean_obs = np.mean(yt)
    nrmse = (rmse / mean_obs * 100.0) if mean_obs > 0 else np.nan
    bias = np.mean(yp - yt)

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "nRMSE_%": float(nrmse),
        "Bias": float(bias)
    }

def compute_bootstrap_ci(y_true, y_pred, metric_fn=mean_squared_error, is_rmse=True,
                         n_bootstraps=1000, alpha=0.05, seed=42):
    """Computes non-parametric bootstrap 95% confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_stats = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_yt = y_true[idx]
        sample_yp = y_pred[idx]
        val = metric_fn(sample_yt, sample_yp)
        if is_rmse:
            val = np.sqrt(val)
        boot_stats.append(val)

    lower = np.percentile(boot_stats, 100.0 * (alpha / 2.0))
    upper = np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)

def paired_permutation_test(y_true, y_pred_a, y_pred_b, n_permutations=2000, seed=42):
    """
    Paired two-sided permutation test to evaluate statistical significance between models.
    """
    rng = np.random.RandomState(seed)
    loss_a = (y_true - y_pred_a) ** 2
    loss_b = (y_true - y_pred_b) ** 2
    observed_diff = np.mean(loss_a - loss_b)

    n = len(y_true)
    count = 0
    for _ in range(n_permutations):
        flips = rng.choice([-1, 1], size=n)
        perm_diff = np.mean(flips * (loss_a - loss_b))
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return float(observed_diff), float(p_value)

def evaluate_multi_target(actual, estimated, target_cols=None, compute_ci=False):
    """Evaluate multi-column ground truth against predictions."""
    target_cols = target_cols or ["AGB", "GSV", "Carbon"]
    records = []
    for i, col in enumerate(target_cols):
        yt = actual[:, i]
        yp = estimated[:, i]
        m = calculate_regression_metrics(yt, yp)
        row = {"Target": col, **m}
        if compute_ci:
            ci_low, ci_high = compute_bootstrap_ci(yt, yp, is_rmse=True)
            row["RMSE_95CI_Low"] = ci_low
            row["RMSE_95CI_High"] = ci_high
        records.append(row)
    return pd.DataFrame(records)
