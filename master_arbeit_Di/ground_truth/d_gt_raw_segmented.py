"""Segmented regression for daily ground-truth (GT) voltage series.

This module fits two independent linear regressions around a user-defined time gap.
It is intended for cases where a middle period is known to be unreliable and should
be excluded from trend estimation.

Main outputs:
- Segmented full-coverage daily GT CSV.
- Interactive HTML figure with two regression lines and an excluded-gap overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


@dataclass
class SegmentedRegressionConfig:
    """Configuration for segmented GT regression.

    Args:
        gap_start_date: Start date of excluded interval, format "YYYY-MM-DD".
        gap_end_date: End date of excluded interval, format "YYYY-MM-DD".
        exclude_gap_start: If True, excludes `gap_start_date` itself.
        exclude_gap_end: If True, excludes `gap_end_date` itself.
    """

    gap_start_date: str
    gap_end_date: str
    exclude_gap_start: bool = False
    exclude_gap_end: bool = False


def read_daily_mean_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a daily-mean GT CSV and use timestamp as index.

    Args:
        csv_path: Input CSV path.

    Returns:
        DataFrame indexed by timestamp.
    """
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.set_index("timestamp")

    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    return df.set_index(df.columns[0])


def _resolve_gap_bounds(config: SegmentedRegressionConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve gap bounds according to inclusion/exclusion flags."""
    gap_start = pd.to_datetime(config.gap_start_date)
    gap_end = pd.to_datetime(config.gap_end_date)

    if not config.exclude_gap_start:
        gap_start = gap_start + timedelta(days=1)
    if not config.exclude_gap_end:
        gap_end = gap_end + timedelta(days=1)

    return gap_start, gap_end


def segment_daily_data(
    daily_df: pd.DataFrame,
    config: SegmentedRegressionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split daily data into two segments around the excluded gap.

    Args:
        daily_df: Daily GT DataFrame indexed by timestamp.
        config: Segmented regression configuration.

    Returns:
        Tuple of (segment_1_df, segment_2_df).
    """
    gap_start, gap_end = _resolve_gap_bounds(config)
    segment_1 = daily_df[daily_df.index < gap_start].copy()
    segment_2 = daily_df[daily_df.index >= gap_end].copy()
    return segment_1, segment_2


def fit_segment_regression(segment_df: pd.DataFrame, value_col: str | None = None) -> dict | None:
    """Fit linear regression for one data segment.

    Args:
        segment_df: Segment DataFrame indexed by timestamp.
        value_col: Optional target column name.

    Returns:
        Regression summary dictionary, or None if insufficient valid data.
    """
    if segment_df.empty:
        return None

    if value_col is None:
        for col in ["gt_uref_raw", "gt_uref_mean", "gt_uref"]:
            if col in segment_df.columns:
                value_col = col
                break
        if value_col is None:
            value_col = segment_df.columns[0]

    work = segment_df[[value_col]].dropna()
    if len(work) < 2:
        return None

    origin_ts = work.index.min()
    t_days = (work.index - origin_ts).total_seconds() / (24.0 * 3600.0)
    y = work[value_col].to_numpy(dtype=float)

    slope_v_per_day, intercept_v, r_value, _, _ = stats.linregress(t_days, y)
    if np.isnan(slope_v_per_day):
        return None

    return {
        "slope_v_per_day": float(slope_v_per_day),
        "slope_uv_per_h": float(slope_v_per_day * 1e6 / 24.0),
        "intercept_v": float(intercept_v),
        "r2": float(r_value**2) if not np.isnan(r_value) else np.nan,
        "n_points": int(len(work)),
        "from_date": work.index.min(),
        "to_date": work.index.max(),
        "origin_ts": origin_ts,
    }


def generate_full_coverage_segmented_gt(
    filtered_df: pd.DataFrame,
    seg1_regression: dict | None,
    seg2_regression: dict | None,
    gap_config: SegmentedRegressionConfig,
) -> pd.DataFrame:
    """Generate daily GT values across full coverage using segmented regressions.

    Args:
        filtered_df: Full filtered dataset, used to infer coverage window.
        seg1_regression: Regression result for pre-gap segment.
        seg2_regression: Regression result for post-gap segment.
        gap_config: Gap configuration.

    Returns:
        DataFrame indexed by timestamp with regression-based daily GT.
    """
    if filtered_df.empty:
        return pd.DataFrame()

    gap_start, gap_end = _resolve_gap_bounds(gap_config)

    start_day = filtered_df.index.min().date()
    end_day = filtered_df.index.max().date()
    daily_timestamps = pd.date_range(start=start_day, end=end_day, freq="D")

    rows: list[dict] = []
    for ts in daily_timestamps:
        if gap_start.date() <= ts.date() < gap_end.date():
            continue

        if ts.date() < gap_start.date():
            regression = seg1_regression
            segment_label = "segment1"
        else:
            regression = seg2_regression
            segment_label = "segment2"

        if regression is None:
            continue

        t_days = (ts - regression["origin_ts"]).days
        v_pred = regression["slope_v_per_day"] * t_days + regression["intercept_v"]

        rows.append(
            {
                "timestamp": ts,
                "gt_uref_regression": v_pred,
                "segment": segment_label,
                "slope_v_per_day": regression["slope_v_per_day"],
                "slope_uv_per_h": regression["slope_uv_per_h"],
                "intercept_v": regression["intercept_v"],
                "r2": regression["r2"],
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.set_index("timestamp")
    return result


def build_segmented_regression_figure(
    daily_df: pd.DataFrame,
    full_coverage_df: pd.DataFrame,
    seg1_regression: dict | None,
    seg2_regression: dict | None,
    gap_config: SegmentedRegressionConfig,
    dataset_name: str = "Dataset",
    value_col: str | None = None,
) -> go.Figure:
    """Build segmented regression visualization.

    Args:
        daily_df: Original daily GT points.
        full_coverage_df: Regression-generated full-coverage daily GT.
        seg1_regression: Segment-1 regression summary.
        seg2_regression: Segment-2 regression summary.
        gap_config: Gap configuration.
        dataset_name: Plot title prefix.
        value_col: Optional value column for daily points.

    Returns:
        Plotly figure.
    """
    if value_col is None:
        for col in ["gt_uref_raw", "gt_uref_mean", "gt_uref"]:
            if col in daily_df.columns:
                value_col = col
                break
        if value_col is None:
            value_col = daily_df.columns[0]

    gap_start = pd.to_datetime(gap_config.gap_start_date)
    gap_end = pd.to_datetime(gap_config.gap_end_date)

    fig = go.Figure()
    fig.add_vrect(
        x0=gap_start,
        x1=gap_end,
        fillcolor="lightgray",
        opacity=0.3,
        layer="below",
        line_width=0,
        name="Excluded Gap",
    )

    if seg1_regression is not None:
        seg1_data = daily_df[daily_df.index < gap_start]
        fig.add_trace(
            go.Scatter(
                x=seg1_data.index,
                y=seg1_data[value_col],
                mode="markers",
                name="Segment 1 (Daily Mean)",
                marker=dict(size=6, color="blue", opacity=0.6),
            )
        )
        seg1_coverage = full_coverage_df[full_coverage_df["segment"] == "segment1"]
        if not seg1_coverage.empty:
            fig.add_trace(
                go.Scatter(
                    x=seg1_coverage.index,
                    y=seg1_coverage["gt_uref_regression"],
                    mode="lines",
                    name=(
                        "Regression 1 "
                        f"(slope={seg1_regression['slope_uv_per_h']:.3f} uV/h, "
                        f"R2={seg1_regression['r2']:.4f})"
                    ),
                    line=dict(color="blue", dash="dash", width=2),
                )
            )

    if seg2_regression is not None:
        seg2_data = daily_df[daily_df.index >= gap_end]
        fig.add_trace(
            go.Scatter(
                x=seg2_data.index,
                y=seg2_data[value_col],
                mode="markers",
                name="Segment 2 (Daily Mean)",
                marker=dict(size=6, color="red", opacity=0.6),
            )
        )
        seg2_coverage = full_coverage_df[full_coverage_df["segment"] == "segment2"]
        if not seg2_coverage.empty:
            fig.add_trace(
                go.Scatter(
                    x=seg2_coverage.index,
                    y=seg2_coverage["gt_uref_regression"],
                    mode="lines",
                    name=(
                        "Regression 2 "
                        f"(slope={seg2_regression['slope_uv_per_h']:.3f} uV/h, "
                        f"R2={seg2_regression['r2']:.4f})"
                    ),
                    line=dict(color="red", dash="dash", width=2),
                )
            )

    fig.update_layout(
        title=f"{dataset_name} - Segmented Regression with Gap Exclusion",
        xaxis_title="Timestamp",
        yaxis_title="Voltage [V]",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1200,
    )
    return fig


def process_segmented_gt_raw(
    daily_mean_csv: str | Path,
    config: SegmentedRegressionConfig,
    filtered_df: Optional[pd.DataFrame] = None,
    output_dir: str | Path = ".",
    dataset_name: str = "Dataset",
    save_html: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run segmented regression workflow for one daily GT file.

    Args:
        daily_mean_csv: Input daily-mean GT CSV.
        config: Segmented regression configuration.
        filtered_df: Optional full filtered dataset for coverage range.
        output_dir: Output directory for CSV/HTML.
        dataset_name: Name used in plot title.
        save_html: Whether to save an HTML visualization.

    Returns:
        Tuple of (full_coverage_df, metrics).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    daily_df = read_daily_mean_csv(daily_mean_csv)
    seg1_daily, seg2_daily = segment_daily_data(daily_df, config)

    seg1_reg = fit_segment_regression(seg1_daily)
    seg2_reg = fit_segment_regression(seg2_daily)

    base_df = filtered_df if filtered_df is not None else daily_df
    full_coverage_df = generate_full_coverage_segmented_gt(base_df, seg1_reg, seg2_reg, config)

    fig = build_segmented_regression_figure(
        daily_df=daily_df,
        full_coverage_df=full_coverage_df,
        seg1_regression=seg1_reg,
        seg2_regression=seg2_reg,
        gap_config=config,
        dataset_name=dataset_name,
    )

    base_name = Path(daily_mean_csv).stem
    csv_out = output_path / f"{base_name}__segmented_regression_full_coverage.csv"
    full_coverage_df.to_csv(csv_out)
    print(f"Saved: {csv_out}")

    if save_html:
        html_out = output_path / f"{base_name}__segmented_regression.html"
        fig.write_html(str(html_out))
        print(f"Saved: {html_out}")

    metrics = {
        "segment1": {
            "n_points": seg1_reg["n_points"] if seg1_reg else 0,
            "slope_uv_per_h": seg1_reg["slope_uv_per_h"] if seg1_reg else np.nan,
            "r2": seg1_reg["r2"] if seg1_reg else np.nan,
            "from": seg1_reg["from_date"] if seg1_reg else None,
            "to": seg1_reg["to_date"] if seg1_reg else None,
        },
        "segment2": {
            "n_points": seg2_reg["n_points"] if seg2_reg else 0,
            "slope_uv_per_h": seg2_reg["slope_uv_per_h"] if seg2_reg else np.nan,
            "r2": seg2_reg["r2"] if seg2_reg else np.nan,
            "from": seg2_reg["from_date"] if seg2_reg else None,
            "to": seg2_reg["to_date"] if seg2_reg else None,
        },
        "gap_excluded": {
            "start": config.gap_start_date,
            "end": config.gap_end_date,
        },
        "full_coverage_points": int(len(full_coverage_df)),
    }

    return full_coverage_df, metrics


if __name__ == "__main__":
    print(__doc__)
