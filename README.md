# Boundary-Density Semi‑Empirical Modeling (BD‑SEM)

Small, single‑purpose Python scripts for component‑wise **semi‑empirical** regression under **boundary density** (BD). The workflow uses a **single, fixed** benchmark transport length $\lambda$ (derived from morphology) across components and reports both in‑sample $R^2$ and LOO‑CV $Q^2$ on the native response scale.

- Semi‑empirical fits for $\eta_{\mathrm{kin}},\ \eta_{\mathrm{RHF}},\ \eta_{\mathrm{RCL}},\ \eta_{\mathrm{res}}$ with **one shared $\lambda$**.
- Metrics: in‑sample $R^2$ and predictive $Q^2$ (leave‑one‑out refitting).
- Each script reads **one** Excel input and writes **one** Excel output (no figures).

## One $\lambda$ benchmark (context)

We adopt a single effective transport length $\lambda$ (default $0.0105\ \mathrm{mm}=10.05\ \mu\mathrm{m}$) derived from active‑fraction arguments: $f(B_D;\lambda)=1-e^{-\lambda B_D}$ with $B_D$ in $\mathrm{mm}^{-1}$ and $\lambda$ in $\mathrm{mm}$. For model closure we use the approximation $\dfrac{1}{f}\approx\dfrac{\lambda B_D+2}{2\lambda B_D}=\dfrac12+\dfrac{1}{\lambda B_D}$, corresponding to inward‑only activation.

## Component models (fixed $\lambda$)

Let $B_D$ be in $\mathrm{mm}^{-1}$ and $\lambda$ in $\mathrm{mm}$.

### Kinetics (fit $b,\ j_0$)
$\eta_{\mathrm{kin}}(B_D; b, j_0)=b\,\log_{10}\left(\dfrac{j}{j_0}\right)+b\,\log_{10}\left(\dfrac12+\dfrac{1}{\lambda B_D}\right)$.  
**Fitting bounds:** $0.10<b<0.15$ (V/dec), $5\times10^{-5}<j_0<5\times10^{-4}$ (A cm $^{-2}$).

### High‑frequency ohmic (fit $R_{\mathrm{HF},0},\ R_{\mathrm{HF},s}$)
Work on $R_{\mathrm{HF}}=\eta_{\mathrm{RHF}}/j$. With fixed $\lambda$:

Latex code:
R_{\mathrm{HF}}(B_D)=R_{\mathrm{HF},0}+R_{\mathrm{HF},s}\left(\dfrac12+\dfrac{1}{\lambda B_D}\right)=\underbrace{\big(R_{\mathrm{HF},0}+\tfrac12 R_{\mathrm{HF},s}\big)}_{a}+\underbrace{\left(\dfrac{R_{\mathrm{HF},s}}{\lambda}\right)}_{b}\dfrac{1}{B_D}

### Catalyst‑layer (fit $R_{\mathrm{CL},s}$; use $b$ from kinetics)
$$\eta_{\mathrm{RCL}}(B_D; R_{\mathrm{CL},s}) = \frac{b}{\alpha}\,\log_{10}（\{1 + (\frac{j \ln 10}{2b}\,R_{\mathrm{CL},s}( \frac{1}{2} + \frac{1}{\lambda B_D} ))^{\alpha}\}）$$

### Residual term (fit $\eta_{\mathrm{con}},\ k_\theta$; use $b$ from kinetics)
To keep dimensions consistent with $B_D$ in $\mathrm{mm}^{-1}$, we use $j=4/100\ \mathrm{A\,mm^{-2}}$ (i.e., $4\ \mathrm{A\,cm^{-2}}=0.04\ \mathrm{A\,mm^{-2}}$), making $k_\theta$ naturally in $\mathrm{mm\,A^{-1}}$:  
$\eta_{\mathrm{res}}(B_D;\eta_{\mathrm{con}},k_\theta)=\eta_{\mathrm{con}}+b\,\log_{10}\left(\dfrac{B_D}{B_D-j\,k_\theta}\right)$, with $B_D\,[\mathrm{mm}^{-1}]$, $j\,[\mathrm{A\,mm^{-2}}]$, $k_\theta\,[\mathrm{mm/A}]$.  
**Positivity constraint:** ensure $B_D-j\,k_\theta>0$ for all data using $0<k_\theta<(1-\varepsilon)\,\dfrac{\min_i B_{D,i}}{j}$ with $\varepsilon\approx0.2$.

## Metrics

We report on the native response scale (V or $\Omega\,\mathrm{cm}^2$):  
$R^2=1-\dfrac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2},\quad Q^2=1-\dfrac{\sum_i (y_i-\hat y_{(-i)})^2}{\sum_i (y_i-\bar y)^2}$.  
$Q^2$ uses leave‑one‑out **refitting**.

## Repository contents

- `screening_singleX_ols.py` — Single‑predictor OLS across four families (linear, $\log_{10}x$, $1/x$, $\ln y$ vs $x$), with LOO‑CV $Q^2$, in‑sample $R^2$, and 95% CIs for $\beta_0,\beta_1$.  
  **Input:** `input_screening_singleX_ols.xlsx` with columns `x1 x2 x3 y1 y2 y3 y4 y5`  
  **Output:** `result_screening_singleX_ols.xlsx` with `all_fits`, `best_single_Q2`, `Q2_pivot`, `settings`.

- `fit_kin_nls.py` — NLS for $(b,j_0)$ under the kinetics model; $\lambda$ fixed; bounded box constraints.  
  **Input:** `input_fit_kin_nls.xlsx` (two columns: $B_D$, $\eta_{\mathrm{kin}}$)  
  **Output:** `result_fit_kin_nls.xlsx` with `params`, `data_fit`, `metrics`, `settings`.

- `fit_rhf_ols.py` — OLS on $R_{\mathrm{HF}}$ vs $1/B_D$ to recover $R_{\mathrm{HF},0}, R_{\mathrm{HF},s}$; $\lambda$ fixed; delta‑method uncertainties for derived parameters.  
  **Input:** `input_fit_rhf_ols.xlsx` (two columns: $B_D$, $\eta_{\mathrm{RHF}}$)  
  **Output:** `result_fit_rhf_ols.xlsx` with `params`, `data_fit`, `metrics`.

- `fit_rcl_nls.py` — NLS for $R_{\mathrm{CL},s}$ in the catalyst‑layer model; $\lambda$ fixed; uses $b$ from kinetics.  
  **Input:** `input_fit_rcl_nls.xlsx` (two columns: $B_D$, $\eta_{\mathrm{RCL}}$)  
  **Output:** `result_fit_rcl_nls.xlsx` with `params`, `data_fit`, `metrics`, `settings`.

- `fit_res_nls.py` — NLS for $(\eta_{\mathrm{con}}, k_\theta)$ with positivity bound on $k_\theta$; uses $b$ from kinetics; $j=0.04\ \mathrm{A\,mm^{-2}}$ so $k_\theta$ is in $\mathrm{mm/A}$.  
  **Input:** `input_fit_res_nls.xlsx` (two columns: $B_D$, $\eta_{\mathrm{res}}$)  
  **Output:** `result_fit_res_nls.xlsx` with `params`, `data_fit`, `metrics`, `settings`.

Each script writes exactly one Excel file named `result_<script>.xlsx`. No figures.

## Installation

```bash
python -m pip install --upgrade numpy pandas scipy openpyxl
```

## Usage

```bash
# Exploratory screening (with 95% CIs)
python screening_singleX_ols.py

# Semi‑empirical component fits (λ fixed; each script independent)
python fit_kin_nls.py
python fit_rhf_ols.py
python fit_rcl_nls.py
python fit_res_nls.py
```

## Conventions and knobs

- **Units:** $B_D$ in $\mathrm{mm}^{-1}$; $\lambda$ in $\mathrm{mm}$; thus $1/(\lambda B_D)$ is dimensionless.
- **Shared $\lambda$:** default $\lambda=0.0105\ \mathrm{mm}$ . Change at the top of each fit script.
- **Kinetics bounds:** $0.10<b<0.15$, $5\times10^{-5}<j_0<5\times10^{-4}$ ($A\ cm^{-2}$).
- **Residual positivity:** $0<k_\theta<(1-\varepsilon)\,\min_i B_{D,i}/j$ with $j=0.04\ \mathrm{A\ mm^{-2}}$, $\varepsilon\approx0.2$.
- **Independence:** each script carries its own constants (`j`, `lambda_mm`, `alpha`, etc.) and a `use_relative_residuals` toggle.
