# Ground Truth (GT) Workflow Documentation

This document describes the current GT pipeline under `master_arbeit_Di/ground_truth`,
following the `a_1 -> a_2 -> b -> c -> d (optional) -> final packaging` convention.
It is intended to make the full data and decision flow explicit, including
human-in-the-loop quality control.

## 1. End-to-End Workflow

### Step a_1: Reference-condition distribution analysis

- Notebook: `a_1_dataset_distribution.ipynb`
- Module: `a_1_data_distribution.py`
- Entry function: `analyze_reference_condition_distribution()`
- Main output directory: `master_arbeit_Di/ground_truth/output_backup/output_histrogram/`
- Purpose: identify dense candidate regions in `(Iref, Tref, OHref)` space.

Human checkpoint:

- Review hotspot tables and 1D/3D distributions.
- Select candidate reference anchors and initial tolerance windows.

### Step a_2: Sensitivity-based tolerance estimation

- Notebook: `a_2_find_gt_tolerance_ranges.ipynb`
- Canonical module: `a_2_find_gt_tolerance_ranges.py`
- Main entry: `run_range_estimation()`
- Output: console report with sensitivities, selected interval counts, and suggested windows.

Human checkpoint:

- Use suggested windows as guidance.
- Finalize engineering tolerance ranges for extraction.

### Step b: Raw GT extraction and diagnostic overlays

- Notebook: `b_dataset_gt_distribution.ipynb`
- Module: `b_gt_distribution.py`
- Entry: `export_gt_diagrams_for_many_datasets()`
- Main output directory: `master_arbeit_Di/ground_truth/output_backup/gt_html/`
- Typical artifacts: raw GT CSV files and interactive HTML overlays.

Human checkpoint:

- Inspect overlays and remove invalid GT points when necessary.
- Save manually curated CSV files to `master_arbeit_Di/ground_truth/output_backup/gt_raw/`.

### Step c: Daily aggregation and single-line regression

- Notebook: `c_dataset_gt_raw_process.ipynb`
- Module: `c_gt_raw_process.py`
- Main entries:
  - `process_many_gt_raw_files()`
  - `process_many_gt_raw_regressions()`
- Output directory: `master_arbeit_Di/ground_truth/output_backup/gt_raw_processed/`

Implementation note:

- For days with multiple raw GT points, the current implementation aggregates to one
  value per day using the **daily mean**.

Human checkpoint:

- Validate daily overlays and regression quality.
- If single-line regression is not adequate, continue to segmented regression.

### Step d (optional): Segmented regression for special cases

Use this step when a specific time interval shows an obviously abnormal degradation
rate compared with the periods before and after it (for example, a suspicious block
in `G1M1` high-load GT). In this case, the unreliable interval is manually removed
from the CSV and treated as a no-GT region. During model-vs-GT evaluation, only the
retained GT intervals are compared.

- Notebook: `d_g1m1_high_load_segmented_regression.ipynb`
- Module: `d_gt_raw_segmented.py`
- Entry: `process_segmented_gt_raw()`
- Input: daily-mean CSV from `output_backup/gt_raw_processed/`
- Typical artifacts:
  - `*__segmented_regression_full_coverage.csv`
  - `*__segmented_regression.html`

### Final packaging: Build GT deliverables

- Script: `build_ground_truth_files.py`
- Main entry: `generate_gt_files()` (called by `main()`)
- Output directories:
  - `master_arbeit_Di/ground_truth/output_backup/generated_gt/`
  - `master_arbeit_Di/ground_truth/output_backup/graph_html/`
- Purpose: generate final GT CSV/JSON metadata and HTML verification plots per Iref.

## 2. Stage-to-Artifact Mapping

- `a_1`: Distribution analysis artifacts for reference-condition candidate discovery.
- `a_2`: Sensitivity reports and suggested tolerance windows.
- `b`: Raw GT extraction CSV + diagnostic overlays (before/after manual cleaning).
- `c`: Daily aggregated GT, single-line regression outputs, full-coverage daily GT series.
- `d` (optional): Segmented regression outputs for cases where one-line regression is not valid.
- Final packaging: final deliverables consumed by downstream modeling and evaluation.

## 3. Detailed Flowchart: Final GT Generation Logic

```mermaid
flowchart TD
  A["Input parquet datasets"] --> B["Step a_1: Distribution analysis<br/>Select candidate Iref/Tref/OHref regions"]
  B --> C["Step a_2: Sensitivity analysis<br/>Propose tolerance windows"]
  C --> D{"Human decision:<br/>Finalize extraction windows?"}
  D -->|Yes| E["Step b: Raw GT extraction<br/>export_gt_diagrams_for_many_datasets"]
  D -->|No| C

  E --> E1["Raw GT CSV + overlay HTML<br/>output_backup/gt_html"]
  E1 --> F{"Human QC:<br/>Invalid GT points removed?"}
  F -->|No| E1
  F -->|Yes| F1["Curated GT raw CSV<br/>output_backup/gt_raw"]

  F1 --> G["Step c-1: Daily aggregation<br/>process_many_gt_raw_files"]
  G --> G1["Daily GT CSV + raw/daily overlays<br/>output_backup/gt_raw_processed"]

  G1 --> H["Step c-2: Single-line regression<br/>process_many_gt_raw_regressions"]
  H --> H1["Full-coverage daily regression CSV + overlay<br/>output_backup/gt_raw_processed"]

  H1 --> I{"Regression quality acceptable?"}
  I -->|Yes| J["Use single-line full-coverage GT"]
  I -->|No| K["Step d optional: Segmented regression<br/>process_segmented_gt_raw"]
  K --> K1["Segmented full-coverage GT + segmented overlay"]

  J --> L["Final packaging<br/>build_ground_truth_files.py / generate_gt_files"]
  K1 --> L

  L --> M["Final GT deliverables<br/>output_backup/generated_gt<br/>CSV + JSON metadata"]
  L --> N["Verification visualizations<br/>output_backup/graph_html"]

  M --> O["Downstream model-vs-GT evaluation<br/>Compare only retained and valid GT intervals"]
  N --> O
```


