#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A lightweight, single-file implementation of the OLS+LOO-CV screening.
- Reads input from 'input_screening_singleX_ols.xlsx' in the same folder.
- Evaluates four model families per (y, x):
    * linear:           y = b0 + b1 * x
    * log10x:           y = b0 + b1 * log10(x)             [x > 0]
    * reciprocalx:      y = b0 + b1 * (1/x)                [x != 0]
    * expY_linearx:     ln(y) = b0 + b1 * x  =>  y_hat = exp(b0 + b1*x)   [y > 0]
- Reports OLS 95% CIs (standard SE-based) for b0 and b1.
- Computes in-sample R^2 and LOO-CV Q^2 on the original y-scale.
- Writes all results to 'result_screening_singleX_ols.xlsx'. No figures.
"""
import pandas as pd
import numpy as np
from itertools import product

RESULT_PATH = "result_screening_singleX_ols.xlsx"
INPUT_PATH  = "input_screening_singleX_ols.xlsx"
FAMILIES = ["linear", "log10x", "reciprocalx", "expY_linearx"]
X_COLS = ["x1", "x2", "x3"]
Y_COLS = ["y1", "y2", "y3", "y4", "y5"]

def _ols_fit(X, y):
    """
    OLS with intercept. Returns params, SEs, R2, predictions, residuals, cov.
    Uses 1-D arrays to avoid NumPy 1.25 scalar-conversion warnings.
    """
    X = np.asarray(X, dtype=float).reshape(-1)   # 1-D
    y = np.asarray(y, dtype=float).reshape(-1)   # 1-D
    n = y.size

    # Design with intercept
    Xd = np.column_stack([np.ones(n), X])        # (n,2)

    # Coefficients
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)    # shape (2,)
    beta = np.asarray(beta).reshape(-1)              # ensure 1-D

    # Fitted values and residuals (1-D)
    yhat = Xd @ beta
    resid = y - yhat

    # R^2
    ybar = float(y.mean())
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - ybar, y - ybar))
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    # Covariance & SEs (classic OLS)
    p = Xd.shape[1]
    dof = max(n - p, 1)
    sigma2 = ss_res / dof
    XtX = Xd.T @ Xd
    try:
        XtX_inv = np.linalg.inv(XtX)
        cov = sigma2 * XtX_inv
        se = np.sqrt(np.diag(cov))               # 1-D
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
        se  = np.full(p, np.nan)

    return {
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "se0":   float(se[0]) if np.isfinite(se[0]) else np.nan,
        "se1":   float(se[1]) if np.isfinite(se[1]) else np.nan,
        "R2":    R2,
        "yhat":  yhat,            # 1-D
        "resid": resid,           # 1-D
        "cov":   cov,
    }

def _loo_q2(y, fit_func):
    """
    Leave-one-out Q^2 computed on original y-scale.
    fit_func(idx_mask) must return a scalar prediction for the held-out element.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size
    yhat_loo = np.full(n, np.nan)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        try:
            yhat_loo[i] = float(fit_func(idx))
        except Exception:
            yhat_loo[i] = np.nan
    ybar = float(np.nanmean(y))
    ss_res = float(np.nansum((y - yhat_loo)**2))
    ss_tot = float(np.nansum((y - ybar)**2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

def main():
    df = pd.read_excel(INPUT_PATH)
    for c in X_COLS + Y_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {INPUT_PATH}. Found: {df.columns.tolist()}")

    results = []
    for ycol, xcol, fam in product(Y_COLS, X_COLS, FAMILIES):
        x_raw = df[xcol].to_numpy(dtype=float)
        y_raw = df[ycol].to_numpy(dtype=float)
        valid_mask = np.isfinite(x_raw) & np.isfinite(y_raw)
        x = x_raw.copy(); y = y_raw.copy()

        # Transform definitions
        if fam == "linear":
            def transform(idx): return x[idx], y[idx]
        elif fam == "log10x":
            valid_mask &= (x_raw > 0)
            def transform(idx): return np.log10(x[idx]), y[idx]
        elif fam == "reciprocalx":
            valid_mask &= (x_raw != 0)
            def transform(idx): return 1.0/x[idx], y[idx]
        elif fam == "expY_linearx":
            valid_mask &= (y_raw > 0)
            def transform(idx): return x[idx], np.log(y[idx])
        else:
            continue

        if not np.any(valid_mask):
            continue
        x = x[valid_mask]; y = y[valid_mask]
        n = y.size
        if n < 3:
            continue

        # Fit on full data
        X_fit, y_fit = transform(np.ones(n, dtype=bool))
        model = _ols_fit(X_fit, y_fit)
        beta0, beta1, se0, se1 = model["beta0"], model["beta1"], model["se0"], model["se1"]

        if fam == "expY_linearx":
            yhat_full = np.exp(beta0 + beta1*X_fit)
            R2 = 1.0 - float(np.sum((y - yhat_full)**2))/float(np.sum((y - np.mean(y))**2))
        else:
            yhat_full = model["yhat"]
            R2 = model["R2"]

        # LOO prediction function (returns scalar prediction for held-out i on original y-scale)
        def fit_func(idx_mask):
            X_tr, y_tr = transform(idx_mask)
            mod = _ols_fit(X_tr, y_tr)
            i = np.where(~idx_mask)[0][0]
            x_i = x[i]
            if fam == "expY_linearx":
                return float(np.exp(mod["beta0"] + mod["beta1"]*x_i))
            elif fam == "linear":
                return float(mod["beta0"] + mod["beta1"]*x_i)
            elif fam == "log10x":
                return float(mod["beta0"] + mod["beta1"]*np.log10(x_i))
            else:  # reciprocalx
                return float(mod["beta0"] + mod["beta1"]*(1.0/x_i))

        Q2 = _loo_q2(y, fit_func)

        # 95% CIs for betas (normal approx; two-sided 1.96)
        tval = 1.96
        ci0 = (beta0 - tval*se0, beta0 + tval*se0) if np.isfinite(se0) else (np.nan, np.nan)
        ci1 = (beta1 - tval*se1, beta1 + tval*se1) if np.isfinite(se1) else (np.nan, np.nan)

        # Human-readable formula
        if fam == "linear":
            formula = f"y = {beta0:.6g} + {beta1:.6g} * x"
        elif fam == "log10x":
            formula = f"y = {beta0:.6g} + {beta1:.6g} * log10(x)"
        elif fam == "reciprocalx":
            formula = f"y = {beta0:.6g} + {beta1:.6g} * (1/x)"
        else:
            formula = f"ln(y) = {beta0:.6g} + {beta1:.6g} * x   (y_hat = exp(...))"

        results.append({
            "y": ycol, "x": xcol, "family": fam,
            "beta0": beta0, "beta1": beta1,
            "se_beta0": se0, "se_beta1": se1,
            "ci_beta0_low": ci0[0], "ci_beta0_high": ci0[1],
            "ci_beta1_low": ci1[0], "ci_beta1_high": ci1[1],
            "R2": R2, "Q2": Q2, "n_used": n,
            "formula": formula
        })

    res_df = pd.DataFrame(results).sort_values(["y","Q2"], ascending=[True, False])
    # Best per y by Q2
    best_df = (res_df.sort_values("Q2", ascending=False)
                      .groupby("y", as_index=False).head(1))

    # Q2 pivot (rows: y, cols: x|family)
    if len(res_df):
        res_df["xfam"] = res_df["x"] + "|" + res_df["family"]
        q2_pivot = res_df.pivot_table(index="y", columns="xfam", values="Q2")
    else:
        q2_pivot = pd.DataFrame()

    with pd.ExcelWriter(RESULT_PATH, engine="openpyxl") as writer:
        res_df.to_excel(writer, sheet_name="all_fits", index=False)
        best_df.to_excel(writer, sheet_name="best_single_Q2", index=False)
        q2_pivot.to_excel(writer, sheet_name="Q2_pivot")
        pd.DataFrame({"key": ["families","input_path"],
                      "value": [", ".join(FAMILIES), INPUT_PATH]}).to_excel(writer, sheet_name="settings", index=False)

if __name__ == "__main__":
    main()
