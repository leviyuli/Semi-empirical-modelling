#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kinetic fit under one-lambda model (lambda fixed), with bounds:
  0.10 < b < 0.15
  5e-5 < j0 < 5e-4

Model:
  eta_kin(BD; b, j0) = b*log10(j/j0) + b*log10(0.5 + 1/(lambda*BD))

Input : 'input_fit_kin_nls.xlsx' (two columns: BD, eta_kin; no header)
Output: 'result_fit_kin_nls.xlsx' with sheets: params, data_fit, metrics, settings
"""
import pandas as pd
import numpy as np
from scipy.optimize import least_squares

# ----- Settings -----
INPUT_PATH  = "input_fit_kin_nls.xlsx"
RESULT_PATH = "result_fit_kin_nls.xlsx"
lambda_fixed_mm = 0.0105   # 10.05 um
j   = 4.0                  # A/cm^2 (applied current density)
# Initial guesses (must lie inside bounds)
b_init   = 0.11            # V/dec
j0_init  = 0.0002          # A/cm^2
use_relative_residuals = False

# Bounds: 0.10 < b < 0.15 ; 5e-5 < j0 < 5e-4
LOWER_BOUNDS = np.array([0.10, 5e-5], dtype=float)
UPPER_BOUNDS = np.array([0.15, 5e-4], dtype=float)
# --------------------

def eta_kin_model(BD, b, j0):
    return b*np.log10(j/j0) + b*np.log10(0.5 + 1.0/(lambda_fixed_mm*BD))

def fit_b_j0(BD, y, x0=(0.11, 2e-4)):
    # Ensure the starting point is within bounds
    x0 = np.clip(np.array(x0, dtype=float), LOWER_BOUNDS + 1e-12, UPPER_BOUNDS - 1e-12)

    def resid_fun(theta):
        b, j0 = theta
        # least_squares enforces bounds, but keep a guard for numerical safety
        if (b <= LOWER_BOUNDS[0]) or (b >= UPPER_BOUNDS[0]) or (j0 <= LOWER_BOUNDS[1]) or (j0 >= UPPER_BOUNDS[1]):
            return 1e6*np.ones_like(y)
        yhat = eta_kin_model(BD, b, j0)
        r = y - yhat
        return r/np.maximum(np.abs(y),1e-9) if use_relative_residuals else r

    return least_squares(
        resid_fun,
        x0=x0,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        method="trf"
    )

def main():
    df = pd.read_excel(INPUT_PATH, header=None).dropna()
    if df.shape[1] < 2:
        raise ValueError("Expect two columns: BD, eta_kin.")
    BD = df.iloc[:,0].to_numpy(dtype=float)
    y  = df.iloc[:,1].to_numpy(dtype=float)
    n  = y.size

    # Full-data fit
    sol = fit_b_j0(BD, y, x0=(b_init, j0_init))
    b_hat, j0_hat = map(float, sol.x)

    # Covariance & SE from LM Jacobian on absolute residuals
    r_abs = y - eta_kin_model(BD, b_hat, j0_hat)
    p = 2; dof = max(n - p, 1)
    sigma2 = float(r_abs @ r_abs) / dof
    J = sol.jac
    try:
        JTJ_inv = np.linalg.inv(J.T @ J)
        cov = sigma2 * JTJ_inv
        se_b, se_j0 = np.sqrt(np.diag(cov)).astype(float)
    except np.linalg.LinAlgError:
        cov = np.full((2,2), np.nan)
        se_b, se_j0 = np.nan, np.nan

    # 95% CIs (normal approx)
    tval = 1.96
    ci_b  = (b_hat  - tval*se_b,  b_hat  + tval*se_b)
    ci_j0 = (j0_hat - tval*se_j0, j0_hat + tval*se_j0)

    # R² on original scale
    yhat = eta_kin_model(BD, b_hat, j0_hat)
    R2 = 1.0 - float(np.sum((y - yhat)**2))/float(np.sum((y - np.mean(y))**2))

    # Q² via LOO re-fitting (respecting the same bounds)
    yhat_loo = np.full(n, np.nan)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        BD_tr, y_tr = BD[idx], y[idx]
        try:
            sol_i = fit_b_j0(BD_tr, y_tr, x0=(b_hat, j0_hat))  # warm start inside bounds
            b_i, j0_i = map(float, sol_i.x)
            yhat_loo[i] = eta_kin_model(np.array([BD[i]]), b_i, j0_i)[0]
        except Exception:
            yhat_loo[i] = np.nan
    ok = np.isfinite(yhat_loo)
    if ok.sum() >= 3:
        ybar = float(np.nanmean(y))
        ss_res_loo = float(np.nansum((y - yhat_loo)**2))
        ss_tot = float(np.nansum((y - ybar)**2))
        Q2 = 1.0 - ss_res_loo/ss_tot if ss_tot > 0 else np.nan
    else:
        Q2 = np.nan

    # Save
    with pd.ExcelWriter(RESULT_PATH, engine="openpyxl") as writer:
        pd.DataFrame({
            "param": ["b","j0","lambda_mm","j"],
            "value": [b_hat, j0_hat, lambda_fixed_mm, j],
            "se":    [se_b,  se_j0, np.nan, np.nan],
            "ci_low":[ci_b[0], ci_j0[0], np.nan, np.nan],
            "ci_high":[ci_b[1], ci_j0[1], np.nan, np.nan],
            "note":  ["NLS (95% CI)","NLS (95% CI)","",""]
        }).to_excel(writer, sheet_name="params", index=False)

        pd.DataFrame({
            "BD": BD, "eta_kin_obs": y, "eta_kin_fit": yhat, "residual": y - yhat
        }).to_excel(writer, sheet_name="data_fit", index=False)

        pd.DataFrame({
            "metric": ["R2","Q2","n_used","loo_preds_ok"],
            "value":  [R2, Q2, n, int(ok.sum())]
        }).to_excel(writer, sheet_name="metrics", index=False)

        pd.DataFrame({
            "setting": ["use_relative_residuals","b_init","j0_init","b_lb","b_ub","j0_lb","j0_ub"],
            "value":   [use_relative_residuals, b_init, j0_init, LOWER_BOUNDS[0], UPPER_BOUNDS[0], LOWER_BOUNDS[1], UPPER_BOUNDS[1]]
        }).to_excel(writer, sheet_name="settings", index=False)

if __name__ == "__main__":
    main()
