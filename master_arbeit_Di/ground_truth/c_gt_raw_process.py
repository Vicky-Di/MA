"""Daily aggregation and regression for raw ground-truth (GT) voltage points.

After the human review step (see GT_WORKFLOW_DOCUMENTATION.md), cleaned raw GT
CSVs stored in ``gt_raw/`` are read by this module.  The module then:
1. Aggregates intra-day scatter to a single daily mean.
2. Fits a global linear regression on the daily GT series.
3. Generates a full-coverage daily GT trajectory spanning the entire dataset.

These outputs feed the next pipeline stage (``d_gt_raw_segmented.py`` for
special cases, or direct export to ``output_backup/generated_gt/``).

Main public API:
    load_preprocessed_dataset()                Load and preprocess a parquet dataset.
    read_gt_raw_csv()                          Load a cleaned raw GT CSV.
    aggregate_gt_raw_daily()                   Reduce to one GT point per day.
    fit_daily_linear_regression()              Fit a linear voltage–time trend.
    generate_full_coverage_daily_gt_from_regression()
                                               Densify weekly GT to every day.
    process_one_gt_raw_file()                  Daily-agg pipeline for one file.
    process_many_gt_raw_files()                Batch daily-agg pipeline.
    process_one_gt_raw_regression()            Regression pipeline for one file.
    process_many_gt_raw_regressions()          Batch regression pipeline.

Typical output locations:
    output_backup/gt_raw_processed/  Daily-mean CSVs, overlay HTMLs, regression CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from degradation_toolbox.Urc.Urc1 import Urc1

try:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess
except ImportError:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Preprocessing filters applied uniformly before GT regression.

    Attributes:
        i_off: Current offset correction [A/cm²].
        u_off: Voltage offset correction [V].
        resample: Resampling stride (1 = no resampling).
        data_filter_i_min: Minimum current density threshold [A/cm²].
        data_filter_U_min: Minimum voltage threshold [V].
        data_filter_U_max: Maximum voltage threshold [V].
        data_filter_T_min: Minimum temperature threshold [°C].
        data_filter_T_max: Maximum temperature threshold [°C].
        data_filter_h_since_last_start_min: Minimum hours since last start [h].
    """

    i_off: float = 0.1
    u_off: float = 1.3
    resample: int = 1
    data_filter_i_min: float = 0.1
    data_filter_U_min: float = 1.4
    data_filter_U_max: float = 2.3
    data_filter_T_min: float = 50.0
    data_filter_T_max: float = 65.0
    data_filter_h_since_last_start_min: float = 0.5


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str | Path) -> Path:
    """Create a directory and all parents if absent.

    Args:
        path: Target directory path.

    Returns:
        Resolved Path object.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _normalize_index_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort coercion of a DataFrame index to DatetimeIndex.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with DatetimeIndex or original index on failure.
    """
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out
    try:
        out.index = pd.to_datetime(out.index)
        return out
    except Exception:
        pass
    out.index = pd.to_datetime(out.index, unit="ms")
    return out


def _safe_stem(path: str | Path) -> str:
    """Return the filename stem (without extension) of a path.

    Args:
        path: File path.

    Returns:
        Stem string.
    """
    return Path(path).stem


def _parse_timestamps_flexible(ts: pd.Series) -> pd.Series:
    """Parse a Series of timestamp strings using two strategies and pick the better.

    Handles both ISO-format and day-first locale strings (e.g., ``14/04/2021``).

    Args:
        ts: Series of raw timestamp strings.

    Returns:
        Series of parsed Timestamps; NaT where parsing fails.
    """
    ts_text = ts.astype(str).str.strip()
    parsed_default = pd.to_datetime(ts_text, errors="coerce", format="mixed", dayfirst=False)
    parsed_dayfirst = pd.to_datetime(ts_text, errors="coerce", format="mixed", dayfirst=True)
    if parsed_dayfirst.notna().sum() > parsed_default.notna().sum():
        return parsed_dayfirst
    return parsed_default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_preprocessed_dataset(
    dataset_path: str,
    preprocess_output_dir: str,
    preprocess_config: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load a parquet dataset, apply GMpreprocess, and run Urc1 secondary filters.

    Args:
        dataset_path: Path to the input parquet file.
        preprocess_output_dir: Directory for GMpreprocess intermediate outputs.
        preprocess_config: Preprocessing thresholds; defaults applied if None.

    Returns:
        Tuple of (filtered_df, dataset_name).

    Raises:
        ValueError: If GMpreprocess returns empty or None data.
    """
    cfg = preprocess_config or PreprocessConfig()
    preprocessor = GMpreprocess(file_path=dataset_path, output_dir=preprocess_output_dir)
    raw_df = preprocessor.run()
    if raw_df is None or raw_df.empty:
        raise ValueError(f"GMpreprocess returned empty data for dataset: {dataset_path}")

    filtered_df = Urc1.preprocess_once(
        raw_df,
        i_off=cfg.i_off,
        u_off=cfg.u_off,
        resample=cfg.resample,
        data_filter_i_min=cfg.data_filter_i_min,
        data_filter_U_min=cfg.data_filter_U_min,
        data_filter_U_max=cfg.data_filter_U_max,
        data_filter_T_min=cfg.data_filter_T_min,
        data_filter_T_max=cfg.data_filter_T_max,
        data_filter_h_since_last_start_min=cfg.data_filter_h_since_last_start_min,
    )
    filtered_df = _normalize_index_to_datetime(filtered_df)
    return filtered_df, preprocessor.name


def read_gt_raw_csv(gt_raw_csv_path: str) -> pd.DataFrame:
    """Read a manually curated raw GT CSV and validate required columns.

    The CSV must contain at least ``timestamp`` and ``gt_uref_raw`` columns.
    Timestamps are parsed with locale-tolerant logic.

    Args:
        gt_raw_csv_path: Path to the raw GT CSV.

    Returns:
        Cleaned DataFrame sorted by timestamp with invalid rows removed.

    Raises:
        ValueError: If required columns are missing or no valid rows remain.
    """
    df = pd.read_csv(gt_raw_csv_path)
    if "timestamp" not in df.columns or "gt_uref_raw" not in df.columns:
        raise ValueError(
            f"GT raw CSV must contain ['timestamp', 'gt_uref_raw'], got: {list(df.columns)}"
        )
    out = df.copy()
    out["timestamp"] = _parse_timestamps_flexible(out["timestamp"])
    out = out.dropna(subset=["timestamp", "gt_uref_raw"]).sort_values("timestamp")
    if out.empty:
        raise ValueError(f"No valid rows after timestamp parsing: {gt_raw_csv_path}")
    return out


def aggregate_gt_raw_daily(gt_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw GT points to at most one (daily mean) point per calendar day.

    This step removes intra-day scatter so that the regression step sees a
    cleaner signal.  Metadata columns (iref, tref, label, etc.) are preserved
    via ``first`` aggregation.

    Args:
        gt_raw_df: Raw GT DataFrame with a ``timestamp`` column.

    Returns:
        Aggregated DataFrame with one row per day, sorted by timestamp.
    """
    if gt_raw_df.empty:
        return gt_raw_df.copy()

    out = gt_raw_df.copy()
    out["date"] = out["timestamp"].dt.floor("D")

    keep_cols = [
        c for c in [
            "dataset_name", "label", "iref", "tref", "ohref",
            "i_range_low", "i_range_high",
            "t_range_low", "t_range_high",
            "oh_range_low", "oh_range_high",
        ] if c in out.columns
    ]

    agg_spec: dict[str, Any] = {"gt_uref_raw": "mean"}
    for col in keep_cols:
        agg_spec[col] = "first"

    daily = out.groupby("date", as_index=False).agg(agg_spec)
    daily = daily.rename(columns={"date": "timestamp"})
    return daily.sort_values("timestamp")


def build_overlay_figure(
    filtered_df: pd.DataFrame,
    gt_points_df: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Create a Plotly overlay of background voltage and GT points.

    Args:
        filtered_df: Full filtered voltage DataFrame indexed by DatetimeIndex.
        gt_points_df: GT rows with ``timestamp`` and ``gt_uref_raw`` columns.
        title: Figure title string.

    Returns:
        Interactive Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=filtered_df.index,
            y=filtered_df["voltage"],
            mode="markers",
            marker=dict(size=2, color="rgba(150,150,150,0.28)"),
            name="All filtered voltage",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=gt_points_df["timestamp"],
            y=gt_points_df["gt_uref_raw"],
            mode="markers",
            marker=dict(size=5, color="#111111"),
            name="GT raw points",
            hovertemplate="<b>GT raw</b><br>Voltage=%{y:.4f} V<br>Time=%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=620,
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Voltage [V]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def fit_daily_linear_regression(
    daily_df: pd.DataFrame,
    origin_ts: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Fit a linear voltage–time trend on the daily GT series.

    The model is ``V = slope * t_days + intercept``, where ``t_days`` is
    measured from ``origin_ts``.

    Args:
        daily_df: Daily-mean GT with a ``timestamp`` column and ``gt_uref_raw``
            values.
        origin_ts: Reference timestamp for t = 0.  Defaults to the earliest
            daily timestamp.

    Returns:
        Dict with keys: origin_ts, slope_v_per_day, slope_uv_per_h,
        intercept_v, r2.

    Raises:
        ValueError: If fewer than 2 valid daily points are present.
    """
    if daily_df.empty or len(daily_df) < 2:
        raise ValueError("Need at least 2 daily points for linear regression.")

    work = daily_df.copy().sort_values("timestamp")
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    origin = pd.Timestamp(origin_ts) if origin_ts is not None else work["timestamp"].min()

    # t_days: elapsed time in days from the origin
    t_days = (work["timestamp"] - origin) / pd.Timedelta(days=1)
    y = work["gt_uref_raw"].astype(float)

    slope_v_per_day, intercept_v = np.polyfit(t_days.to_numpy(), y.to_numpy(), deg=1)
    pred = slope_v_per_day * t_days + intercept_v

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "origin_ts": origin,
        "slope_v_per_day": float(slope_v_per_day),
        "slope_uv_per_h": float(slope_v_per_day * 1e6 / 24.0),
        "intercept_v": float(intercept_v),
        "r2": r2,
    }


def build_regression_overlay_figure(
    filtered_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    regression: dict[str, Any],
    title: str,
) -> go.Figure:
    """Extend a GT overlay figure with a full-span regression line.

    Builds on :func:`build_overlay_figure` and appends a dashed regression
    line spanning the entire filtered dataset time window.

    Args:
        filtered_df: Full filtered voltage DataFrame.
        daily_df: Daily-mean GT points.
        regression: Regression summary from :func:`fit_daily_linear_regression`.
        title: Figure title string.

    Returns:
        Interactive Plotly figure with added regression trace.
    """
    fig = build_overlay_figure(filtered_df, daily_df, title)

    line_x = pd.to_datetime([filtered_df.index.min(), filtered_df.index.max()])
    t_days = (line_x - regression["origin_ts"]) / pd.Timedelta(days=1)
    line_y = regression["slope_v_per_day"] * t_days + regression["intercept_v"]

    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color="#1f77b4", width=3, dash="dash"),
            name="Daily regression",
            hovertemplate="<b>Regression</b><br>Voltage=%{y:.4f} V<br>Time=%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"{title}<br><sup>slope={regression['slope_v_per_day']:.6g} V/day "
            f"({regression['slope_uv_per_h']:.2f} uV/h), R2={regression['r2']:.4f}</sup>"
        )
    )
    return fig


def generate_full_coverage_daily_gt_from_regression(
    filtered_df: pd.DataFrame,
    regression: dict[str, Any],
    meta_row: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate one regression-based GT value for every calendar day in the dataset.

    The output time span matches the full filtered dataset (not just the days
    where raw GT points existed), providing dense GT coverage for model evaluation.

    Args:
        filtered_df: Full filtered DataFrame; its index defines the date range.
        regression: Regression summary from :func:`fit_daily_linear_regression`.
        meta_row: Optional dict of metadata columns (e.g., iref, label) to
            propagate into the output DataFrame.

    Returns:
        DataFrame with columns: timestamp, gt_uref_regression,
        slope_v_per_day, slope_uv_per_h, intercept_v, r2.
    """
    start_day = pd.Timestamp(filtered_df.index.min()).floor("D")
    end_day = pd.Timestamp(filtered_df.index.max()).floor("D")
    all_days = pd.date_range(start=start_day, end=end_day, freq="D")

    t_days = (all_days - regression["origin_ts"]) / pd.Timedelta(days=1)
    gt_daily = regression["slope_v_per_day"] * t_days + regression["intercept_v"]

    out = pd.DataFrame({
        "timestamp": all_days,
        "gt_uref_regression": gt_daily,
        "slope_v_per_day": regression["slope_v_per_day"],
        "slope_uv_per_h": regression["slope_uv_per_h"],
        "intercept_v": regression["intercept_v"],
        "r2": regression["r2"],
    })

    if meta_row:
        for key in [
            "dataset_name", "label", "iref", "tref", "ohref",
            "i_range_low", "i_range_high",
            "t_range_low", "t_range_high",
            "oh_range_low", "oh_range_high",
        ]:
            if key in meta_row:
                out[key] = meta_row[key]
    return out


def process_one_gt_raw_file(
    dataset_path: str,
    gt_raw_csv_path: str,
    preprocess_output_dir: str,
    output_dir: str,
    preprocess_config: PreprocessConfig | None = None,
    save_html: bool = True,
) -> dict[str, Any]:
    """Daily-aggregation pipeline for one raw GT CSV file.

    Reads the raw GT CSV, aggregates to daily means, optionally saves HTML
    overlays with the original voltage background, and writes the daily CSV.

    Args:
        dataset_path: Path to the parquet dataset (used for background voltage).
        gt_raw_csv_path: Path to the manually cleaned raw GT CSV.
        preprocess_output_dir: Intermediate preprocessing directory.
        output_dir: Destination for output files.
        preprocess_config: Optional preprocessing thresholds.
        save_html: Whether to save raw and daily-overlay HTML figures.

    Returns:
        Summary dict with paths to daily CSV and optional HTML files.
    """
    filtered_df, dataset_name = load_preprocessed_dataset(
        dataset_path=dataset_path,
        preprocess_output_dir=preprocess_output_dir,
        preprocess_config=preprocess_config,
    )
    gt_raw_df = read_gt_raw_csv(gt_raw_csv_path)
    daily_df = aggregate_gt_raw_daily(gt_raw_df)

    out_dir = _ensure_dir(output_dir)
    file_stem = _safe_stem(gt_raw_csv_path)

    daily_csv_path = out_dir / f"{file_stem}__daily_mean.csv"
    daily_df.to_csv(daily_csv_path, index=False)

    raw_html_path = daily_html_path = None
    if save_html:
        raw_html_path = out_dir / f"{file_stem}__raw_overlay.html"
        daily_html_path = out_dir / f"{file_stem}__daily_overlay.html"
        build_overlay_figure(filtered_df, gt_raw_df, f"{dataset_name} | Raw GT | {file_stem}").write_html(
            str(raw_html_path), include_plotlyjs="cdn"
        )
        build_overlay_figure(filtered_df, daily_df, f"{dataset_name} | Daily GT | {file_stem}").write_html(
            str(daily_html_path), include_plotlyjs="cdn"
        )

    return {
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "gt_raw_csv": gt_raw_csv_path,
        "raw_points": int(len(gt_raw_df)),
        "daily_points": int(len(daily_df)),
        "daily_csv": str(daily_csv_path),
        "raw_overlay_html": None if raw_html_path is None else str(raw_html_path),
        "daily_overlay_html": None if daily_html_path is None else str(daily_html_path),
    }


def process_many_gt_raw_files(
    items: list[dict[str, str]],
    preprocess_output_dir: str,
    output_dir: str,
    preprocess_config: PreprocessConfig | None = None,
    save_html: bool = True,
) -> pd.DataFrame:
    """Batch daily-aggregation pipeline for multiple GT CSV files.

    Each item in ``items`` must contain ``dataset_path`` and ``gt_raw_csv`` keys.

    Args:
        items: List of dicts mapping dataset_path and gt_raw_csv.
        preprocess_output_dir: Shared intermediate preprocessing directory.
        output_dir: Destination directory.
        preprocess_config: Shared preprocessing thresholds.
        save_html: Whether to save HTML overlays.

    Returns:
        Summary DataFrame saved as ``gt_raw_daily_process_summary.csv``.
    """
    rows = [
        process_one_gt_raw_file(
            dataset_path=item["dataset_path"],
            gt_raw_csv_path=item["gt_raw_csv"],
            preprocess_output_dir=preprocess_output_dir,
            output_dir=output_dir,
            preprocess_config=preprocess_config,
            save_html=save_html,
        )
        for item in items
    ]
    summary_df = pd.DataFrame(rows)
    out_dir = _ensure_dir(output_dir)
    summary_df.to_csv(out_dir / "gt_raw_daily_process_summary.csv", index=False)
    print(f"Saved daily-process summary: {out_dir / 'gt_raw_daily_process_summary.csv'}")
    return summary_df


def process_one_gt_raw_regression(
    dataset_path: str,
    gt_raw_csv_path: str,
    preprocess_output_dir: str,
    output_dir: str,
    preprocess_config: PreprocessConfig | None = None,
    save_html: bool = True,
) -> dict[str, Any]:
    """Full regression pipeline for one raw GT CSV file.

    Reads the cleaned CSV, aggregates daily, fits a linear trend, and produces
    a full-coverage daily GT that spans the entire dataset duration.

    Args:
        dataset_path: Path to the parquet dataset.
        gt_raw_csv_path: Path to the manually cleaned raw GT CSV.
        preprocess_output_dir: Intermediate preprocessing directory.
        output_dir: Destination for output files.
        preprocess_config: Optional preprocessing thresholds.
        save_html: Whether to save the regression overlay HTML.

    Returns:
        Summary dict with regression metrics and output file paths.
    """
    filtered_df, dataset_name = load_preprocessed_dataset(
        dataset_path=dataset_path,
        preprocess_output_dir=preprocess_output_dir,
        preprocess_config=preprocess_config,
    )
    gt_raw_df = read_gt_raw_csv(gt_raw_csv_path)
    daily_df = aggregate_gt_raw_daily(gt_raw_df)
    reg = fit_daily_linear_regression(daily_df, origin_ts=filtered_df.index.min())

    out_dir = _ensure_dir(output_dir)
    file_stem = _safe_stem(gt_raw_csv_path)

    meta_row = daily_df.iloc[0].to_dict() if not daily_df.empty else {}
    full_gt_df = generate_full_coverage_daily_gt_from_regression(filtered_df, reg, meta_row=meta_row)
    full_csv = out_dir / f"{file_stem}__daily_regression_full_coverage.csv"
    full_gt_df.to_csv(full_csv, index=False)

    reg_html_path = None
    if save_html:
        title = f"{dataset_name} | Daily GT + Regression | {file_stem}"
        build_regression_overlay_figure(filtered_df, daily_df, reg, title).write_html(
            str(out_dir / f"{file_stem}__daily_overlay_regression.html"),
            include_plotlyjs="cdn",
        )
        reg_html_path = out_dir / f"{file_stem}__daily_overlay_regression.html"

    return {
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "gt_raw_csv": gt_raw_csv_path,
        "daily_points": int(len(daily_df)),
        "slope_v_per_day": reg["slope_v_per_day"],
        "slope_uv_per_h": reg["slope_uv_per_h"],
        "intercept_v": reg["intercept_v"],
        "r2": reg["r2"],
        "daily_regression_full_coverage_csv": str(full_csv),
        "daily_regression_overlay_html": None if reg_html_path is None else str(reg_html_path),
    }


def process_many_gt_raw_regressions(
    items: list[dict[str, str]],
    preprocess_output_dir: str,
    output_dir: str,
    preprocess_config: PreprocessConfig | None = None,
    save_html: bool = True,
) -> pd.DataFrame:
    """Batch regression pipeline for multiple GT CSV files.

    Each item in ``items`` must contain ``dataset_path`` and ``gt_raw_csv`` keys.

    Args:
        items: List of dicts mapping dataset_path and gt_raw_csv.
        preprocess_output_dir: Shared intermediate preprocessing directory.
        output_dir: Destination directory.
        preprocess_config: Shared preprocessing thresholds.
        save_html: Whether to save regression HTML overlays.

    Returns:
        Summary DataFrame saved as ``gt_raw_regression_summary.csv``.
    """
    rows = [
        process_one_gt_raw_regression(
            dataset_path=item["dataset_path"],
            gt_raw_csv_path=item["gt_raw_csv"],
            preprocess_output_dir=preprocess_output_dir,
            output_dir=output_dir,
            preprocess_config=preprocess_config,
            save_html=save_html,
        )
        for item in items
    ]
    summary_df = pd.DataFrame(rows)
    out_dir = _ensure_dir(output_dir)
    summary_df.to_csv(out_dir / "gt_raw_regression_summary.csv", index=False)
    print(f"Saved regression summary: {out_dir / 'gt_raw_regression_summary.csv'}")
    return summary_df


def _example_main() -> None:
    """Minimal self-contained usage example.  Edit the paths before running."""
    from pathlib import Path as _P

    project_root = _P(__file__).resolve().parents[2]
    cfg = PreprocessConfig()
    items = [
        {
            "dataset_path": str(project_root / "explore_data" / "G6M2.parquet"),
            "gt_raw_csv": str(
                project_root
                / "master_arbeit_Di"
                / "ground_truth"
                / "output_backup"
                / "gt_raw"
                / "G6M2__gt_diagram__g6m2_mid_load__iref_1__tref_58__ohref_33.csv"
            ),
        }
    ]
    summary = process_many_gt_raw_files(
        items=items,
        preprocess_output_dir=str(project_root / "explore_data" / "output"),
        output_dir=str(
            project_root / "master_arbeit_Di" / "ground_truth" / "output_backup" / "gt_raw_processed"
        ),
        preprocess_config=cfg,
        save_html=True,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    _example_main()
