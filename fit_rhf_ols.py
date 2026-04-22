#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OLS for R_HF vs 1/BD under one-lambda model:
R_HF(BD) = R_HF0 + R_HF_s*(0.5 + 1/(lambda*BD))
        = a + b*(1/BD), with a = R_HF0 + R_HF_s/2, b = R_HF_s/lambda

Input : 'input_fit_rhf_ols.xlsx' (two cols: BD, eta_RHF (V))
Output: 'result_fit_rhf_ols.xlsx' with params, data_fit, metrics
"""
import pandas as pd
import numpy as np

# ----- Settings -----
INPUT_PATH  = "input_fit_rhf_ols.xlsx"
RESULT_PATH = "result_fit_rhf_ols.xlsx"
j               = 4.0        # A/cm^2
lambda_fixed_mm = 0.0105     # 10.05 um
# --------------------

def main():
    df = pd.read_excel(INPUT_PATH, header=None).dropna()
    if df.shape[1] < 2:
        raise ValueError("Expect two columns: BD, eta_RHF.")
    BD  = df.iloc[:,0].to_numpy(dtype=float)
    eta = df.iloc[:,1].to_numpy(dtype=float)

    R_HF = eta / j                  # Ω cm^2
    X = 1.0 / BD                    # predictor
    n = R_HF.size

    # OLS: R_HF = a + b * (1/BD)
    Xd = np.column_stack([np.ones(n), X])      # (n,2)
    beta, *_ = np.linalg.lstsq(Xd, R_HF, rcond=None)  # (2,)
    a_hat, b_hat = float(beta[0]), float(beta[1])
    yhat = Xd @ beta
    resid = R_HF - yhat

    ybar = float(R_HF.mean())
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((R_HF - ybar)**2))
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    # Covariance of (a,b)
    p = 2
    dof = max(n-p, 1)
    sigma2 = ss_res / dof
    XtX = Xd.T @ Xd
    try:
        XtX_inv = np.linalg.inv(XtX)
        cov = sigma2 * XtX_inv      # (2,2)
    except np.linalg.LinAlgError:
        cov = np.full((2,2), np.nan)
    se_a, se_b = np.sqrt(np.diag(cov)).astype(float)

    # Derived parameters: R_HF_s, R_HF0
    R_HF_s_hat = b_hat * lambda_fixed_mm
    R_HF0_hat  = a_hat - 0.5 * R_HF_s_hat

    # Delta-method SEs for derived params
    # For R_HF_s = lambda * b: var = lambda^2 * Var(b)
    se_RHF_s = abs(lambda_fixed_mm) * se_b
    # For R_HF0 = a - (lambda/2) b: grad = [1, -lambda/2]
    g = np.array([1.0, -0.5*lambda_fixed_mm])
    var_RHF0 = float(g @ cov @ g) if np.all(np.isfinite(cov)) else np.nan
    se_RHF0  = float(np.sqrt(var_RHF0)) if np.isfinite(var_RHF0) else np.nan

    # 95% CIs
    tval = 1.96
    ci_a       = (a_hat - tval*se_a,      a_hat + tval*se_a)
    ci_b       = (b_hat - tval*se_b,      b_hat + tval*se_b)
    ci_RHF_s   = (R_HF_s_hat - tval*se_RHF_s, R_HF_s_hat + tval*se_RHF_s)
    ci_RHF0    = (R_HF0_hat  - tval*se_RHF0,  R_HF0_hat  + tval*se_RHF0)

    # LOO-CV Q² (refit)
    yhat_loo = np.full(n, np.nan)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        Xi = X[idx]; Yi = R_HF[idx]
        Xdi = np.column_stack([np.ones(Xi.size), Xi])
        try:
            bet_i, *_ = np.linalg.lstsq(Xdi, Yi, rcond=None)
            yhat_loo[i] = float(bet_i[0] + bet_i[1]*X[i])
        except Exception:
            yhat_loo[i] = np.nan
    ok = np.isfinite(yhat_loo)
    if ok.sum() >= 3:
        ybar = float(R_HF.mean())
        ss_res_loo = float(np.nansum((R_HF - yhat_loo)**2))
        ss_tot = float(np.sum((R_HF - ybar)**2))
        Q2 = 1.0 - ss_res_loo/ss_tot if ss_tot > 0 else np.nan
    else:
        Q2 = np.nan

    # Save
    with pd.ExcelWriter(RESULT_PATH, engine="openpyxl") as writer:
        pd.DataFrame({
            "param": ["a (intercept)","b (slope 1/BD)","R_HF_s","R_HF0","j","lambda_mm"],
            "value": [a_hat, b_hat, R_HF_s_hat, R_HF0_hat, j, lambda_fixed_mm],
            "se":    [se_a,  se_b,  se_RHF_s,  se_RHF0,   np.nan, np.nan],
            "ci_low":[ci_a[0], ci_b[0], ci_RHF_s[0], ci_RHF0[0], np.nan, np.nan],
            "ci_high":[ci_a[1], ci_b[1], ci_RHF_s[1], ci_RHF0[1], np.nan, np.nan],
            "note": ["OLS (95% CI)","OLS (95% CI)","delta-method","delta-method","",""]
        }).to_excel(writer, sheet_name="params", index=False)

        pd.DataFrame({
            "BD": BD, "inv_BD": X, "R_HF_obs": R_HF, "R_HF_fit": yhat, "residual": R_HF - yhat
        }).to_excel(writer, sheet_name="data_fit", index=False)

        pd.DataFrame({
            "metric": ["R2","Q2","n_used","loo_preds_ok"],
            "value":  [R2, Q2, n, int(ok.sum())]
        }).to_excel(writer, sheet_name="metrics", index=False)

if __name__ == "__main__":
    main()
