#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Catalyst-layer fit under one-lambda model (lambda fixed):
eta_RCL = (b/alpha) * log10( 1 + [ (j ln 10)/(2 b) * R_CL_s * (0.5 + 1/(lambda*BD)) ]^alpha )

Input : 'input_fit_rcl_nls.xlsx' (two cols: BD, eta_RCL; no header)
Output: 'result_fit_rcl_nls.xlsx' with params, data_fit, metrics, settings
Note   : b should be set to the intrinsic b from the kinetics fit (fit_kin_nls.py).
"""
import pandas as pd
import numpy as np
from scipy.optimize import least_squares

# ----- Settings -----
INPUT_PATH  = "input_fit_rcl_nls.xlsx"
RESULT_PATH = "result_fit_rcl_nls.xlsx"
lambda_fixed_mm = 0.0105
alpha = 1.1982
j     = 4.0
# Set this to the b fitted from fit_kin_nls.py (intrinsic Tafel slope).
# Keep a sensible default so the script can run independently.
b     = 0.10
use_relative_residuals = False
# --------------------

LN10 = np.log(10.0)

def eta_rcl_model(BD, R_s):
    factor = (j * LN10) / (2.0 * b)                    # scalar
    mult   = (0.5 + 1.0/(lambda_fixed_mm*BD))          # (n,)
    inner  = factor * R_s * mult                        # (n,)
    return (b/alpha) * np.log10(1.0 + np.power(inner, alpha))

def fit_Rs(BD, y, R0=0.05):
    def resid_fun(theta):
        R_s = theta[0]
        if R_s <= 0:
            return 1e6*np.ones_like(y)
        yhat = eta_rcl_model(BD, R_s)
        r = y - yhat
        return r/np.maximum(np.abs(y),1e-9) if use_relative_residuals else r
    return least_squares(resid_fun, x0=np.array([R0], dtype=float),
                         bounds=(np.array([1e-9]), np.array([np.inf])),
                         method="trf")

def main():
    df = pd.read_excel(INPUT_PATH, header=None).dropna()
    if df.shape[1] < 2:
        raise ValueError("Expect two columns: BD, eta_RCL.")
    BD = df.iloc[:,0].to_numpy(dtype=float)
    y  = df.iloc[:,1].to_numpy(dtype=float)
    n  = y.size

    # Full-data fit
    sol = fit_Rs(BD, y, R0=0.05)
    R_s_hat = float(sol.x[0])

    # Covariance & SE
    r_abs = y - eta_rcl_model(BD, R_s_hat)
    p = 1; dof = max(n - p, 1)
    sigma2 = float(r_abs @ r_abs) / dof
    J = sol.jac                               # (n,1)
    try:
        JTJ_inv = np.linalg.inv(J.T @ J)      # (1,1)
        cov = sigma2 * JTJ_inv
        se_Rs = float(np.sqrt(cov[0,0]))
    except np.linalg.LinAlgError:
        cov = np.array([[np.nan]])
        se_Rs = np.nan

    # 95% CI
    tval = 1.96
    ci_Rs = (R_s_hat - tval*se_Rs, R_s_hat + tval*se_Rs)

    # R^2 on original scale
    yhat = eta_rcl_model(BD, R_s_hat)
    R2 = 1.0 - float(np.sum((y - yhat)**2))/float(np.sum((y - np.mean(y))**2))

    # Q^2 via LOO refit
    yhat_loo = np.full(n, np.nan)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        BD_tr, y_tr = BD[idx], y[idx]
        try:
            sol_i = fit_Rs(BD_tr, y_tr, R0=R_s_hat)
            Rs_i = float(sol_i.x[0])
            yhat_loo[i] = eta_rcl_model(np.array([BD[i]]), Rs_i)[0]
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
            "param": ["R_CL_s","alpha","b_used","j","lambda_mm"],
            "value": [R_s_hat, alpha, b,       j,  lambda_fixed_mm],
            "se":    [se_Rs,   np.nan, np.nan, np.nan, np.nan],
            "ci_low":[ci_Rs[0], np.nan, np.nan, np.nan, np.nan],
            "ci_high":[ci_Rs[1], np.nan, np.nan, np.nan, np.nan],
            "note":  ["NLS (95% CI)","","","",""]
        }).to_excel(writer, sheet_name="params", index=False)

        pd.DataFrame({
            "BD": BD, "eta_RCL_obs": y, "eta_RCL_fit": yhat, "residual": y - yhat
        }).to_excel(writer, sheet_name="data_fit", index=False)

        pd.DataFrame({
            "metric": ["R2","Q2","n_used","loo_preds_ok"],
            "value":  [R2, Q2, n, int(ok.sum())]
        }).to_excel(writer, sheet_name="metrics", index=False)

        pd.DataFrame({
            "setting": ["use_relative_residuals","b_source"],
            "value":   [use_relative_residuals, "Set to kinetics-fit b"]
        }).to_excel(writer, sheet_name="settings", index=False)

if __name__ == "__main__":
    main()
