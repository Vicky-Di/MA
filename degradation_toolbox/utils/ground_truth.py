"""
Ground Truth Extraction from Raw Dataset

When a dataset does not contain a pre-computed Uref column (e.g. G6M2),
this module extracts "pseudo ground truth" by:

1. Filtering raw data points whose operating conditions (current density,
   temperature, h_since_last_start) are close to a given reference condition
   within user-specified tolerance bounds.
2. Taking the measured voltage at those points as the ground truth Urc.
3. Linearly interpolating across the full time axis to fill gaps.

The output can be fed directly into ``Urc1_Surface_Stats_GT`` as ``gt_uref``.

Note
----
The input ``data`` **must already contain** a ``h_since_last_start`` column.
If your raw data doesn't have it yet, compute it first::

    from degradation_toolbox.utils.calculation_basics import calc_h_since_last_start
    data = calc_h_since_last_start(data, "currentDensity>0.1 or voltage>1.3", 1)

Then call ``extract_ground_truth(data, ...)``.

Usage Example
-------------
>>> from degradation_toolbox.utils.ground_truth import extract_ground_truth
>>> from degradation_toolbox.utils.calculation_basics import calc_h_since_last_start
>>>
>>> # Step 1: Compute h_since_last_start
>>> data = calc_h_since_last_start(data, "currentDensity>0.1 or voltage>1.3", 1)
>>>
>>> # Step 2: Extract ground truth
>>> gt_uref, data = extract_ground_truth(
...     data=data,                           # DataFrame with DatetimeIndex
...     ref_I=1.0, tol_I=(0.95, 1.05),      # I ∈ [0.95, 1.05]
...     ref_T=60,  tol_T=(58, 62),           # T ∈ [58, 62]

...     dataset_name="G6M2",
...     plot=True,
... )
>>>
>>> # Step 3: Feed into model (drop helper columns before model input)
>>> data_for_model = data.drop(columns=["gt_uref_raw", "gt_uref"])
>>> model = Urc1_Surface_Stats_GT(data=data_for_model, gt_uref=gt_uref, ...)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _format_iref_for_filename(iref: float) -> str:
    """Convert 0.6 -> '0p6' for stable filenames."""
    return f"{float(iref):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _parse_iref_from_filename_token(token: str) -> float:
    """Convert '0p6' back to 0.6."""
    return float(token.replace("p", "."))


def extract_ground_truth(
    data: pd.DataFrame,
    ref_I: float,
    tol_I: tuple[float, float],
    ref_T: float,
    tol_T: tuple[float, float],
    ref_OH: float,
    tol_OH: tuple[float, float],
    col_I: str = "currentDensity",
    col_T: str = "temperature",
    col_U: str = "voltage",
    col_OH: str = "h_since_last_start",
    dataset_name: str = "Dataset",
    plot: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Extract pseudo ground truth from raw data by filtering near reference conditions.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with DatetimeIndex. Must contain columns for current density,
        temperature, voltage, and h_since_last_start.
    ref_I : float
        Reference current density [A/cm²].
    tol_I : tuple(float, float)
        (lower_bound, upper_bound) for current density filtering.
    ref_T : float
        Reference temperature [°C].
    tol_T : tuple(float, float)
        (lower_bound, upper_bound) for temperature filtering.
    ref_OH : float
        Reference operational hours since last start [h].
    tol_OH : tuple(float, float)
        (lower_bound, upper_bound) for h_since_last_start filtering.
    col_I : str
        Column name for current density (default: 'currentDensity').
    col_T : str
        Column name for temperature (default: 'temperature').
    col_U : str
        Column name for voltage (default: 'voltage').
    col_OH : str
        Column name for h_since_last_start (default: 'h_since_last_start').
    dataset_name : str
        Name of the dataset (used in plot titles).
    plot : bool
        Whether to generate diagnostic plots (default: True).

    Returns
    -------
    gt_uref : pd.Series
        Ground truth voltage series with DatetimeIndex.
        Index = timestamp, values = voltage [V].
        Contains linearly interpolated values for all timestamps in `data`.
        Can be passed directly to ``Urc1_Surface_Stats_GT(gt_uref=...)``
    data : pd.DataFrame
        The input DataFrame with two additional columns:
        - ``gt_uref_raw``: voltage at real GT points, NaN elsewhere.
        - ``gt_uref``: linearly interpolated GT for all rows.
    """
    # =====================================================================
    # Validate input
    # =====================================================================
    required_cols = [col_I, col_T, col_U, col_OH]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(
            f"Missing columns in data: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(
            "data.index must be a DatetimeIndex. "
            f"Got {type(data.index).__name__}."
        )

    I_lo, I_hi = tol_I
    T_lo, T_hi = tol_T
    OH_lo, OH_hi = tol_OH
    # =====================================================================
    # Step 1: Filter data points near reference conditions
    # =====================================================================
    mask = (
        (data[col_I] >= I_lo) & (data[col_I] <= I_hi) &
        (data[col_T] >= T_lo) & (data[col_T] <= T_hi) &
        (data[col_OH] >= OH_lo) & (data[col_OH] <= OH_hi)
    )

    gt_points = data.loc[mask, col_U].copy()
    gt_points.name = "gt_uref_raw"

    n_total = len(data)
    n_gt = mask.sum()

    print(f"{'=' * 60}")
    print(f"  Ground Truth Extraction: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"  Reference condition:")
    print(f"    I  = {ref_I} A/cm²   (filter: [{I_lo}, {I_hi}])")
    print(f"    T  = {ref_T} °C      (filter: [{T_lo}, {T_hi}])")
    print(f"  Total data points:  {n_total}")
    print(f"  GT points found:    {n_gt}  ({n_gt / n_total * 100:.2f}%)")

    if n_gt == 0:
        raise ValueError(
            "No data points match the reference condition. "
            "Consider widening the tolerance bounds."
        )

    print(f"  GT voltage range:   [{gt_points.min():.4f}, {gt_points.max():.4f}] V")
    print(f"  GT time span:       {gt_points.index[0]} → {gt_points.index[-1]}")
    print(f"{'=' * 60}")

    # =====================================================================
    # Step 2: Add GT columns to data
    # =====================================================================
    # Column 1: raw GT (only at real GT points, NaN elsewhere)
    data["gt_uref_raw"] = np.nan
    data.loc[mask, "gt_uref_raw"] = gt_points.values

    # Column 2: interpolated GT (real GT + linear interpolation for gaps)
    data["gt_uref"] = data["gt_uref_raw"].interpolate(method="time")

    # Mark points before the first GT and after the last GT as NaN
    # (extrapolation is unreliable)
    first_gt_idx = gt_points.index[0]
    last_gt_idx = gt_points.index[-1]
    data.loc[data.index < first_gt_idx, "gt_uref"] = np.nan
    data.loc[data.index > last_gt_idx, "gt_uref"] = np.nan

    n_interp = data["gt_uref"].notna().sum()
    print(f"  Interpolated GT coverage: {n_interp} / {n_total} "
          f"({n_interp / n_total * 100:.1f}%)")

    # =====================================================================
    # Step 3: Prepare output Series for Urc1_Surface_Stats_GT
    # =====================================================================
    # gt_uref as a Series with DatetimeIndex (the format expected by the model)
    gt_uref_series = data["gt_uref"].dropna().copy()
    gt_uref_series.name = "Uref"

    # =====================================================================
    # Step 4: Diagnostic plots
    # =====================================================================
    if plot:
        ref_str = f"I={ref_I} A/cm², T={ref_T}°C, OH={ref_OH}h"

        _plot_raw_gt(data, gt_points, col_U, dataset_name, ref_str)
        _plot_interpolated_gt(data, gt_points, col_U, dataset_name, ref_str)

    return gt_uref_series, data


def save_ground_truth_series(
    gt_uref: pd.Series,
    output_path: str,
    dataset_name: str,
    iref: float,
    ref_T: float | None = 60.0,
    ref_OH: float | None = 72.0,
    i_range: tuple[float, float] | None = None,
    t_range: tuple[float, float] | None = None,
    oh_range: tuple[float, float] | None = None,
    source: str = "extracted",
) -> dict:
    """
    Save one GT series to disk as CSV plus sidecar JSON metadata.

    CSV keeps the time/value payload simple and portable.
    JSON stores reference conditions and filtering ranges so future notebooks
    can verify they are loading the intended GT.
    """
    if gt_uref is None or len(gt_uref) == 0:
        raise ValueError("gt_uref is empty. Nothing to save.")

    series = gt_uref.copy()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series.name = "gt_uref"

    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = csv_path.with_suffix(".json")

    payload = series.rename("gt_uref").to_frame().reset_index()
    timestamp_col = payload.columns[0]
    payload = payload.rename(columns={timestamp_col: "timestamp"})
    payload.to_csv(csv_path, index=False)

    metadata = {
        "dataset_name": dataset_name,
        "iref": float(iref),
        "ref_T": None if ref_T is None else float(ref_T),
        "ref_OH": None if ref_OH is None else float(ref_OH),
        "i_range": list(i_range) if i_range is not None else None,
        "t_range": list(t_range) if t_range is not None else None,
        "oh_range": list(oh_range) if oh_range is not None else None,
        "source": source,
        "n_points": int(series.notna().sum()),
        "time_start": series.index.min().isoformat(),
        "time_end": series.index.max().isoformat(),
        "csv_file": csv_path.name,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    return metadata


def load_ground_truth_series(input_path: str) -> tuple[pd.Series, dict]:
    """
    Load one GT series from CSV and, if present, its sidecar JSON metadata.
    """
    csv_path = Path(input_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if "gt_uref" not in df.columns:
        value_cols = [c for c in df.columns if c != "timestamp"]
        if not value_cols:
            raise ValueError(f"No GT value column found in {csv_path}")
        df = df.rename(columns={value_cols[0]: "gt_uref"})

    series = df.set_index("timestamp")["gt_uref"].dropna().sort_index()
    series.name = "Uref"

    meta_path = csv_path.with_suffix(".json")
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    if "iref" not in metadata:
        stem_parts = csv_path.stem.split("__iref_")
        if len(stem_parts) == 2:
            metadata["iref"] = _parse_iref_from_filename_token(stem_parts[1])

    return series, metadata


def save_ground_truth_bundle(
    gt_by_iref: dict[float, pd.Series],
    output_dir: str,
    dataset_name: str,
    ref_T: float | None = 60.0,
    ref_OH: float | None = 72.0,
    i_tolerance: float | None = None,
    t_range: tuple[float, float] | None = None,
    oh_range: tuple[float, float] | None = None,
    source: str = "extracted",
) -> dict[float, dict]:
    """
    Save multiple GT series, one file per Iref.

    Filename pattern:
        {dataset_name}__gt__iref_{0p6}.csv
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_by_iref: dict[float, dict] = {}
    for iref, gt_uref in gt_by_iref.items():
        iref_token = _format_iref_for_filename(float(iref))
        csv_path = output_root / f"{dataset_name}__gt__iref_{iref_token}.csv"
        i_range = None
        if i_tolerance is not None:
            i_range = (float(iref) - float(i_tolerance), float(iref) + float(i_tolerance))

        metadata = save_ground_truth_series(
            gt_uref=gt_uref,
            output_path=str(csv_path),
            dataset_name=dataset_name,
            iref=float(iref),
            ref_T=ref_T,
            ref_OH=ref_OH,
            i_range=i_range,
            t_range=t_range,
            oh_range=oh_range,
            source=source,
        )
        metadata_by_iref[float(iref)] = metadata

    return metadata_by_iref


def load_ground_truth_bundle(
    input_dir: str,
    dataset_name: str | None = None,
) -> tuple[dict[float, pd.Series], dict[float, dict]]:
    """
    Load all GT files from a directory.

    Returns
    -------
    gt_by_iref : dict
        Mapping {iref: gt_series}
    metadata_by_iref : dict
        Mapping {iref: metadata}
    """
    input_root = Path(input_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {input_root}")

    pattern = "*__gt__iref_*.csv" if dataset_name is None else f"{dataset_name}__gt__iref_*.csv"
    files = sorted(input_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ground truth CSV files found in {input_root} matching {pattern}")

    gt_by_iref: dict[float, pd.Series] = {}
    metadata_by_iref: dict[float, dict] = {}
    for csv_path in files:
        series, metadata = load_ground_truth_series(str(csv_path))
        if "iref" not in metadata:
            raise ValueError(f"Unable to determine iref for GT file: {csv_path}")

        iref = float(metadata["iref"])
        gt_by_iref[iref] = series
        metadata_by_iref[iref] = metadata

    return gt_by_iref, metadata_by_iref


# =========================================================================
# Plotting helpers
# =========================================================================

def _plot_raw_gt(
    data: pd.DataFrame,
    gt_points: pd.Series,
    col_U: str,
    dataset_name: str,
    ref_str: str,
):
    """Plot 1: Only the real (filtered) GT data points."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.10,
        subplot_titles=[
            f"Raw Ground Truth Points ({len(gt_points)} pts)",
            "GT Voltage Distribution",
        ],
    )

    # --- Row 1: GT time series ---
    # Background: all raw voltage as light grey
    fig.add_trace(
        go.Scattergl(
            x=data.index,
            y=data[col_U],
            mode="markers",
            marker=dict(size=1, color="lightgrey", opacity=0.3),
            name="All Data",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Foreground: GT points highlighted
    fig.add_trace(
        go.Scattergl(
            x=gt_points.index,
            y=gt_points.values,
            mode="markers",
            marker=dict(size=3, color="crimson", opacity=0.6),
            name="GT Points",
        ),
        row=1, col=1,
    )

    # --- Row 2: GT voltage histogram ---
    fig.add_trace(
        go.Histogram(
            x=gt_points.values,
            nbinsx=100,
            marker_color="crimson",
            name="GT Distribution",
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Timestamp", row=1, col=1)
    fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
    fig.update_xaxes(title_text="Voltage [V]", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_layout(
        title=f"{dataset_name} – Raw Ground Truth  ({ref_str})",
        height=700,
        width=1000,
        template="plotly_white",
    )
    fig.show()


def _plot_interpolated_gt(
    data: pd.DataFrame,
    gt_points: pd.Series,
    col_U: str,
    dataset_name: str,
    ref_str: str,
):
    """Plot 2: Full interpolated GT curve + raw GT points."""
    gt_interp = data["gt_uref"].dropna()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.10,
        subplot_titles=[
            "Interpolated Ground Truth",
            "Interpolated GT Distribution",
        ],
    )

    # --- Row 1: Interpolated GT curve ---
    # Background: all raw voltage as light grey
    fig.add_trace(
        go.Scattergl(
            x=data.index,
            y=data[col_U],
            mode="markers",
            marker=dict(size=1, color="lightgrey", opacity=0.2),
            name="All Data",
        ),
        row=1, col=1,
    )

    # Interpolated GT line
    fig.add_trace(
        go.Scattergl(
            x=gt_interp.index,
            y=gt_interp.values,
            mode="lines",
            line=dict(color="steelblue", width=1.5),
            name="GT (interpolated)",
        ),
        row=1, col=1,
    )

    # Real GT points on top
    fig.add_trace(
        go.Scattergl(
            x=gt_points.index,
            y=gt_points.values,
            mode="markers",
            marker=dict(size=3, color="crimson", opacity=0.6),
            name="GT Points (real)",
        ),
        row=1, col=1,
    )

    # --- Row 2: Distribution ---
    fig.add_trace(
        go.Histogram(
            x=gt_interp.values,
            nbinsx=150,
            marker_color="steelblue",
            name="Interpolated",
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Timestamp", row=1, col=1)
    fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
    fig.update_xaxes(title_text="Voltage [V]", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_layout(
        title=f"{dataset_name} – Interpolated Ground Truth  ({ref_str})",
        height=700,
        width=1000,
        template="plotly_white",
    )
    fig.show()
