# PEM Electrolyzer Voltage Degradation Analysis

A Python research codebase for analysing long-term voltage degradation in Proton
Exchange Membrane Water Electrolysis (PEMWE) stacks. Developed as part of a
master's thesis project.

The core idea is to estimate the **voltage under reference operating conditions**
(U_rc) over time, separate the ohmic drift from process-variable effects (current
density, temperature, operating hours), and quantify a statistically robust
degradation rate in uV/h.

---

## Repository Structure

```
MA_code/
+-- degradation_toolbox/          # Installable Python library (core models & utilities)
|   +-- Urc/                      # All Urc model implementations
|   +-- pol_curve_extraction/     # Staircase polarisation-curve extraction helper
|   +-- utils/                    # Shared calculation and data utilities
|
+-- master_arbeit_Di/             # Thesis-specific analysis scripts and notebooks
|   +-- explore/                  # Pre-processing and comparison framework
|   +-- utils/                    # Stability evaluation, outlier detection, sensitivity
|   +-- ground_truth/             # Ground-truth computation and alignment
|   +-- ada_with_min_num/         # Adaptive model experiments on G1M1_new & G6M2
|   +-- C3_ln(oh)/                # C3 voltage-drop model experiments 
|                                 # (C3 Method for reference only.
|                                 #  Due to parameter differences, 
|                                 # it is not possible to compare with the baseline.)
|   +-- c4_c6/                    # Cbrt and Sigmoid model experiments
|   +-- GPR/                      # GPR and CbrtGPR model experiments
|   +-- huber_ridge/              # Huber robust model experiments
|   +-- iref/                     # Iref filtering model experiments
|   +-- surface/                  # Surface global model experiments
|   +-- surface_stats/            # Surface Stats (RLM) model experiments
|   +-- plots/                    # Output plots from all experiment notebooks
|
+-- explore_data/                 # Training datasets: G1M1_new (large) and G6M2 (small).
|                                 # Included in the repo so notebooks can be run and
|                                 # results reproduced directly without external data.
+-- instability_demo/             # Outlier investigation notebooks
+-- requirements.txt
+-- README.md
```

---

## Core Library -- degradation_toolbox/

### Urc/ -- Model Family

All models share the five-parameter linear voltage model in scaled feature space:

    U - OCV(T) = c1*I_s + c2*(I*T)_s + c3*log(h)_s + c4*I2_s + c5

The baseline model is **Urc1**, based on the work of Xuqian Yan (main branch).
Key extensions introduced in this project:

1. **Shared preprocessing** (`GMpreprocess`) — all derived models share the same
   data-cleaning pipeline so results across models are directly comparable.
2. **Unified parameter interface** — models accept `ref_config` (low / medium / high
   reference conditions derived from GT analysis) as well as model-specific extra
   parameters, enabling consistent cross-model benchmarking.

| File | Model | Key idea |
|---|---|---|
| Urc1.py | Baseline | Fixed-width sliding window OLS, per-interval fitting (from Xuqian Yan) |
| Urc1_adaptive.py | Adaptive | Expanding window that merges intervals with insufficient data |
| Urc1_c3.py | C3 / voltage-drop | Detects and drops voltage-step intervals to isolate slow drift |
| Urc1_c6_sigmoid.py | C6 Sigmoid | Replaces linear I term with sigmoid activation for non-linear kinetics |
| Urc1_cbrt_gpr.py | CbrtGPR | Cube-root current feature combined with Gaussian Process coefficient smoothing |
| Urc1_gpr.py | GPR | Gaussian Process Regression smoothing of per-interval coefficient trajectories |
| Urc1_huber.py | Huber | Robust per-interval fitting using Huber-weighted least squares |
| Urc1_Iref.py | Iref | Filters data to a narrow band around each reference current for cleaner estimates |
| Urc1_surface.py | Surface | Global time-varying least-squares across all intervals simultaneously |
| Urc1_surface_stats.py | Surface Stats | Global RLM (statsmodels HuberT) variant of the Surface model |
| Urc.py | Wrapper | Thin wrapper that groups Baseline + Adaptive for legacy compatibility |
| config.py | Config | Shared hyperparameter defaults (scaler bounds, filter thresholds) |
| helpers.py | Helpers | Shared feature scaling and OCV functions |

### pol_curve_extraction/

Extracts voltage-current polarisation curves from staircase current-step protocols.
Provides the `PolarizationCurve` class for optional use with structured test data.

### utils/

| File | Purpose |
|---|---|
| calculation_basics.py | `calc_h_since_last_start` -- cumulative operating hours since last start |
| data_helpers.py | Shared loading and column-normalisation utilities |
| ground_truth.py | GT reference-condition voltage computation helpers |

---

## Thesis Research Code -- master_arbeit_Di/

### explore/

| File | Purpose |
|---|---|
| GMpreprocess.py | `GMpreprocess` -- loads and cleans raw electrolyser parquet data |
| UnifiedModelComparator.py | Runs all Urc model variants, aligns to ground truth, computes stability metrics, exports comparison plots |
| Comparison_tools.py | Lower-level comparison plotting utilities (called by UnifiedModelComparator) |
| deg_rate.py | `DegradationRate` dataclass; linear regression of Urc time series -> uV/h estimate |

### utils/

| File | Purpose |
|---|---|
| stability_evaluation.py | `evaluate_urc_stability()` -- computes coefficient CV, condition number, and GT RMSE across all models |
| outlier_detection.py | `UrcOutlierDetector` -- identifies top-N% outlier intervals by residual distance |
| investigate.py | `Urc1OutlierInvestigator` -- diagnoses outlier root causes (sparse data, collinearity, anchoring) |
| cal_monotonicity.py | Spearman rank and sign-monotonicity metrics for degradation trend quality |
| para_sensitivity_analysis.py | `ParameterSensitivityAnalysis` -- one-at-a-time sensitivity analysis for Adaptive model hyperparameters |

### ground_truth/

Notebooks and scripts for computing the ground-truth reference voltage from
the G1M1_new and G6M2 datasets. GT trajectories are used as the external
validation target for all model variants.


!!NOTE: For a detailed process and flowchart of generating gt, see `master_arbeit_Di/ground_truth/GT_WORKFLOW_DOCUMENTATION.md`

### Experiment folders

Each folder contains Jupyter notebooks that train and evaluate a model variant on
both G1M1_new and G6M2. Output plots are saved to `master_arbeit_Di/plots/`.

| Folder | Model(s) covered | Notebooks |
|---|---|---|
| ada_with_min_num/ | Baseline + Adaptive | G1M1 & G6M2 training, parameter sensitivity analysis |
| C3_ln(oh)/ | C3 voltage-drop | G1M1 & G6M2 experiments |
| c4_c6/ | C6 Sigmoid | G1M1 & G6M2 experiments |
| GPR/ | GPR + CbrtGPR | G1M1 & G6M2 experiments |
| huber_ridge/ | Huber | G1M1 & G6M2 training + sensitivity analysis |
| iref/ | Iref | G1M1 & G6M2 experiments |
| surface/ | Surface | G1M1 & G6M2 comparison |
| surface_stats/ | Surface Stats | G1M1 & G6M2 comparison |

### plots folders

Each folder contains the results and plots that are generated from experiment folders.

---

## Comparison Framework

`UnifiedModelComparator` (in `master_arbeit_Di/explore/`) is the main entry point
for multi-model evaluation:

1. Instantiates and runs all enabled Urc model variants on a dataset.
2. Aligns each model's Urc time series against the ground-truth reference voltage.
3. Computes RMSE, MAE, stability score, and condition-number statistics.
4. Produces interactive Plotly dashboards (coefficient trends, residuals, coverage).

---

## Key Concepts

**Urc (Voltage under Reference Conditions)**
The voltage a cell would exhibit at a fixed reference operating point. The low,
medium, and high values of I_ref, T_ref, and h_ref are derived from GT analysis
of the data distribution. Urc is estimated by regressing out operating-variable
effects from the measured voltage.

**Ground Truth (GT)**
By analysing the data distribution, the three reference conditions (high, medium,
and low) with the most data coverage were identified. Ground truth Urc trajectories
were generated from these conditions and are applied across both datasets
(G1M1_new, G6M2). GT is used to benchmark model accuracy.

---

## Installation

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # Linux / macOS

# Install the core library in editable mode
pip install -e .

# Install all runtime dependencies
pip install -r requirements.txt
```

---

## Quick Start

```python
from degradation_toolbox.Urc.Urc1 import Urc1
from master_arbeit_Di.explore.GMpreprocess import GMpreprocess

# Load and preprocess data
preprocessor = GMpreprocess(file_path="explore_data/G6M2.parquet",
                             output_dir="explore_data/output")
data = preprocessor.run()

# Run Baseline model
model = Urc1(data=data)
model.run()

# Inspect results
print(model.fitting_results_reliable.head())
model.get_degradation_rate(target_i=1.5)
```

---

## Running Experiments

Each method folder under `master_arbeit_Di/` contains Jupyter notebooks that can
be run with the `venv` kernel. The datasets are already included in `explore_data/`
so results can be reproduced without any additional data setup.

Training progress and evaluation results appear in the notebook cell outputs.
Generated plots are saved to `master_arbeit_Di/plots/<method>/`. (The `plots/`
directory already contains results from the final training run.)

