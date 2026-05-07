"""GT diagram export — raw ground-truth extraction per reference condition.

This module provides functions to extract raw GT candidate points from a
preprocessed dataset for one or more user-defined reference-condition windows
(Iref, Tref, OHref).  Each reference specification is saved as a CSV and
optionally as an interactive HTML overlay.

Main public API:
    load_preprocessed_dataset()           Load and preprocess a parquet dataset.
    extract_gt_for_reference()            Extract raw GT points for one reference spec.
    build_gt_overlay_figure()             Build an interactive voltage-vs-time overlay.
    export_gt_diagram()                   Run extraction + export for one reference.
    export_gt_diagrams_for_dataset()      Batch export for one dataset, multiple refs.
    export_gt_diagrams_for_many_datasets() Batch export across multiple datasets.

Typical output locations (configured in calling notebooks):
    output_backup/gt_html/  Intermediate raw GT CSVs and HTML overlays.
    output_backup/gt_raw/   Manually curated subset of raw GT CSVs (human-edited copies).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from degradation_toolbox.Urc.Urc1 import Urc1
from degradation_toolbox.utils.ground_truth import extract_ground_truth

try:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess
except ImportError:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Preprocessing filters applied uniformly before any GT extraction.

    Attributes:
        i_off: Current offset correction [A/cm²].
        u_off: Voltage offset correction [V].
        resample: Resampling stride (1 = no resampling).
        data_filter_i_min: Minimum current density threshold [A/cm²].
        data_filter_U_min: Minimum voltage threshold [V].
        data_filter_U_max: Maximum voltage threshold [V].
        data_filter_T_min: Minimum temperature threshold [°C].
        data_filter_T_max: Maximum temperature threshold [°C].
        data_filter_h_since_last_start_min: Minimum operating hours since last
            start, used to discard unstable warm-up periods [h].
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


@dataclass
class ReferenceConditionSpec:
    """Specification for one reference operating condition.

    Attributes:
        iref: Reference current density [A/cm²].
        tref: Reference temperature [°C].
        ohref: Reference operating hours since last start [h].
        i_range: Symmetric tolerance window around iref: (low, high) [A/cm²].
        t_range: Tolerance window around tref: (low, high) [°C].
        oh_range: Tolerance window for OHref: (low, high) [h].
        label: Optional human-readable label used in filenames and plot titles.
    """

    iref: float
    tref: float
    ohref: float
    i_range: tuple[float, float]
    t_range: tuple[float, float]
    oh_range: tuple[float, float]
    label: str | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str | Path) -> Path:
    """Create a directory (and all parents) if it does not already exist.

    Args:
        path: Target directory path.

    Returns:
        Resolved Path object pointing to the directory.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _safe_token(value: float) -> str:
    """Convert a float to a compact filename-safe token.

    Examples:
        1.5   -> "1p5"
        1.0   -> "1"
        0.6   -> "0p6"

    Args:
        value: Numeric value to encode.

    Returns:
        Filename-safe string representation.
    """
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _safe_label(label: str | None) -> str:
    """Convert an arbitrary label to a lowercase filesystem-safe token.

    Args:
        label: Raw label string, or None.

    Returns:
        Cleaned lowercase label suitable for use in filenames.
    """
    if not label:
        return ""
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned.lower()


def _normalize_index_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to coerce a DataFrame index to DatetimeIndex.

    Args:
        df: Input DataFrame with an unchecked index type.

    Returns:
        DataFrame with a DatetimeIndex (best-effort conversion).
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


def _save_raw_gt_points_csv(
    data_with_gt: pd.DataFrame,
    output_path: str | Path,
    dataset_name: str,
    spec: ReferenceConditionSpec,
) -> Path:
    """Persist raw GT voltage points with full reference metadata to CSV.

    Args:
        data_with_gt: DataFrame carrying a `gt_uref_raw` column produced by
            ``extract_ground_truth()``.
        output_path: Destination CSV file path.
        dataset_name: Human-readable dataset identifier.
        spec: Reference condition specification for sidecar metadata columns.

    Returns:
        Path to the written CSV file.
    """
    gt_raw = data_with_gt["gt_uref_raw"].dropna()
    out_df = gt_raw.rename("gt_uref_raw").to_frame().reset_index()
    time_col = out_df.columns[0]
    out_df = out_df.rename(columns={time_col: "timestamp"})
    out_df["dataset_name"] = dataset_name
    out_df["label"] = spec.label
    out_df["iref"] = spec.iref
    out_df["tref"] = spec.tref
    out_df["ohref"] = spec.ohref
    out_df["i_range_low"] = spec.i_range[0]
    out_df["i_range_high"] = spec.i_range[1]
    out_df["t_range_low"] = spec.t_range[0]
    out_df["t_range_high"] = spec.t_range[1]
    out_df["oh_range_low"] = spec.oh_range[0]
    out_df["oh_range_high"] = spec.oh_range[1]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return output_path


def _coerce_reference_spec(spec: ReferenceConditionSpec | dict[str, Any]) -> ReferenceConditionSpec:
    """Normalise a dict or existing spec to a :class:`ReferenceConditionSpec`.

    Args:
        spec: Raw spec as a dataclass or plain dict with identical keys.

    Returns:
        Validated :class:`ReferenceConditionSpec` instance.
    """
    if isinstance(spec, ReferenceConditionSpec):
        return spec
    return ReferenceConditionSpec(
        iref=float(spec["iref"]),
        tref=float(spec["tref"]),
        ohref=float(spec["ohref"]),
        i_range=(float(spec["i_range"][0]), float(spec["i_range"][1])),
        t_range=(float(spec["t_range"][0]), float(spec["t_range"][1])),
        oh_range=(float(spec["oh_range"][0]), float(spec["oh_range"][1])),
        label=spec.get("label"),
    )


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
        Tuple of (filtered_df, dataset_name) where filtered_df is indexed by
        DatetimeIndex and dataset_name is inferred from the preprocessor.

    Raises:
        ValueError: If GMpreprocess returns empty or None data.
    """
    cfg = preprocess_config or PreprocessConfig()
    preprocessor = GMpreprocess(file_path=dataset_path, output_dir=preprocess_output_dir)
    raw_df = preprocessor.run()
    if raw_df is None or raw_df.empty:
        raise ValueError("GMpreprocess returned empty data.")

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


def extract_gt_for_reference(
    preprocessed_df: pd.DataFrame,
    dataset_name: str,
    reference_spec: ReferenceConditionSpec | dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame, ReferenceConditionSpec]:
    """Extract raw GT voltage points that satisfy one reference window.

    All data rows whose (I, T, OH) values fall within the tolerance windows
    defined in ``reference_spec`` are treated as raw GT candidates.

    Args:
        preprocessed_df: Filtered and indexed DataFrame from
            :func:`load_preprocessed_dataset`.
        dataset_name: Dataset identifier for labelling outputs.
        reference_spec: Reference window definition as dataclass or dict.

    Returns:
        Tuple of (gt_uref, data_with_gt, spec):
            - gt_uref: Interpolated GT voltage Series.
            - data_with_gt: Full dataset DataFrame with added ``gt_uref_raw``
              column marking the matching points.
            - spec: Resolved :class:`ReferenceConditionSpec`.
    """
    spec = _coerce_reference_spec(reference_spec)
    gt_uref, data_with_gt = extract_ground_truth(
        data=preprocessed_df.copy(),
        ref_I=spec.iref,
        tol_I=spec.i_range,
        ref_T=spec.tref,
        tol_T=spec.t_range,
        ref_OH=spec.ohref,
        tol_OH=spec.oh_range,
        dataset_name=dataset_name,
        plot=False,
    )
    return gt_uref, data_with_gt, spec


def build_gt_overlay_figure(
    data_with_gt: pd.DataFrame,
    dataset_name: str,
    reference_spec: ReferenceConditionSpec,
) -> go.Figure:
    """Create a Plotly figure overlaying raw GT points on all filtered voltage.

    Args:
        data_with_gt: DataFrame with ``voltage`` and ``gt_uref_raw`` columns;
            rows without a GT value show as background voltage scatter.
        dataset_name: Plot title prefix.
        reference_spec: Used to annotate the title with reference metadata.

    Returns:
        Interactive Plotly figure.
    """
    gt_raw = data_with_gt["gt_uref_raw"].dropna()

    title_parts = [
        f"{dataset_name} Ground Truth Diagram",
        f"Iref={reference_spec.iref:.4f}",
        f"Tref={reference_spec.tref:.2f}",
        f"OHref={reference_spec.ohref:.2f}",
    ]
    if reference_spec.label:
        title_parts.insert(1, f"[{reference_spec.label}]")

    subtitle = (
        f"I in [{reference_spec.i_range[0]:.4f}, {reference_spec.i_range[1]:.4f}] | "
        f"T in [{reference_spec.t_range[0]:.2f}, {reference_spec.t_range[1]:.2f}] | "
        f"OH in [{reference_spec.oh_range[0]:.2f}, {reference_spec.oh_range[1]:.2f}] | "
        f"raw GT pts={len(gt_raw)}"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=data_with_gt.index,
            y=data_with_gt["voltage"],
            mode="markers",
            marker=dict(size=2, color="rgba(150,150,150,0.28)"),
            name="All filtered voltage",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=gt_raw.index,
            y=gt_raw.values,
            mode="markers",
            marker=dict(size=4, color="#080808"),
            name="GT raw points",
            hovertemplate="<b>GT raw</b><br>Voltage=%{y:.4f} V<br>Time=%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title="<br>".join([" ".join(title_parts), f"<sup>{subtitle}</sup>"]),
        template="plotly_white",
        height=620,
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Voltage [V]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def export_gt_diagram(
    dataset_path: str,
    preprocess_output_dir: str,
    output_dir: str,
    reference_spec: ReferenceConditionSpec | dict[str, Any],
    preprocess_config: PreprocessConfig | None = None,
    dataset_name_override: str | None = None,
    save_html: bool = False,
    save_gt_csv: bool = True,
    csv_output_dir: str | None = None,
) -> dict[str, Any]:
    """End-to-end GT extraction and export for a single reference condition.

    Args:
        dataset_path: Path to the input parquet file.
        preprocess_output_dir: Intermediate preprocessing output directory.
        output_dir: Destination directory for HTML outputs.
        reference_spec: Reference condition definition.
        preprocess_config: Optional preprocessing thresholds.
        dataset_name_override: Use this name instead of the preprocessor-inferred one.
        save_html: Whether to write an HTML diagnostic plot.
        save_gt_csv: Whether to write the raw GT CSV.
        csv_output_dir: Optional dedicated directory for GT CSV exports.
            If None, CSV files are written to ``output_dir``.

    Returns:
        Summary dict with keys: dataset_name, iref, tref, ohref, i_range,
        t_range, oh_range, n_gt_points, html_file, csv_file.
    """
    preprocessed_df, inferred_name = load_preprocessed_dataset(
        dataset_path=dataset_path,
        preprocess_output_dir=preprocess_output_dir,
        preprocess_config=preprocess_config,
    )
    dataset_name = dataset_name_override or inferred_name
    gt_uref, data_with_gt, spec = extract_gt_for_reference(
        preprocessed_df=preprocessed_df,
        dataset_name=dataset_name,
        reference_spec=reference_spec,
    )
    figure = build_gt_overlay_figure(data_with_gt, dataset_name, spec)
    html_out_dir = _ensure_dir(output_dir)
    csv_out_dir = _ensure_dir(csv_output_dir) if csv_output_dir else html_out_dir

    label_token = _safe_label(spec.label)
    stem_parts = [dataset_name, "gt_diagram"]
    if label_token:
        stem_parts.append(label_token)
    stem_parts.extend([
        f"iref_{_safe_token(spec.iref)}",
        f"tref_{_safe_token(spec.tref)}",
        f"ohref_{_safe_token(spec.ohref)}",
    ])
    file_stem = "__".join(stem_parts)

    html_path = None
    if save_html:
        html_path = html_out_dir / f"{file_stem}.html"
        figure.write_html(str(html_path), include_plotlyjs="cdn")

    csv_path = None
    if save_gt_csv:
        csv_path = csv_out_dir / f"{file_stem}.csv"
        _save_raw_gt_points_csv(data_with_gt, csv_path, dataset_name, spec)

    n_pts = int(data_with_gt["gt_uref_raw"].notna().sum())
    return {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "label": spec.label,
        "iref": spec.iref,
        "tref": spec.tref,
        "ohref": spec.ohref,
        "i_range": list(spec.i_range),
        "t_range": list(spec.t_range),
        "oh_range": list(spec.oh_range),
        "n_gt_points": n_pts,
        "html_file": None if html_path is None else str(html_path),
        "csv_file": None if csv_path is None else str(csv_path),
    }


def export_gt_diagrams_for_dataset(
    dataset_path: str,
    preprocess_output_dir: str,
    output_dir: str,
    reference_specs: list[ReferenceConditionSpec | dict[str, Any]],
    preprocess_config: PreprocessConfig | None = None,
    dataset_name_override: str | None = None,
    save_html: bool = False,
    save_gt_csv: bool = True,
    csv_output_dir: str | None = None,
) -> pd.DataFrame:
    """Export GT diagrams for all reference conditions of a single dataset.

    Preprocessing is performed once; then all reference windows are evaluated
    sequentially without reloading the data.

    Args:
        dataset_path: Path to the input parquet file.
        preprocess_output_dir: Intermediate preprocessing output directory.
        output_dir: Destination directory for HTML outputs and per-dataset summary.
        reference_specs: List of reference condition specifications.
        preprocess_config: Optional preprocessing thresholds.
        dataset_name_override: Override the dataset name used in filenames.
        save_html: Whether to write HTML diagnostic plots.
        save_gt_csv: Whether to write raw GT CSVs.
        csv_output_dir: Optional dedicated directory for raw GT CSV outputs.
            If None, CSV files are written to ``output_dir``.

    Returns:
        Summary DataFrame, one row per reference, saved as
        ``{dataset_name}__gt_diagram_summary.csv`` in ``output_dir``.
    """
    preprocessed_df, inferred_name = load_preprocessed_dataset(
        dataset_path=dataset_path,
        preprocess_output_dir=preprocess_output_dir,
        preprocess_config=preprocess_config,
    )
    dataset_name = dataset_name_override or inferred_name
    html_out_dir = _ensure_dir(output_dir)
    csv_out_dir = _ensure_dir(csv_output_dir) if csv_output_dir else html_out_dir

    rows: list[dict[str, Any]] = []
    for raw_spec in reference_specs:
        gt_uref, data_with_gt, spec = extract_gt_for_reference(
            preprocessed_df=preprocessed_df,
            dataset_name=dataset_name,
            reference_spec=raw_spec,
        )
        figure = build_gt_overlay_figure(data_with_gt, dataset_name, spec)

        label_token = _safe_label(spec.label)
        stem_parts = [dataset_name, "gt_diagram"]
        if label_token:
            stem_parts.append(label_token)
        stem_parts.extend([
            f"iref_{_safe_token(spec.iref)}",
            f"tref_{_safe_token(spec.tref)}",
            f"ohref_{_safe_token(spec.ohref)}",
        ])
        file_stem = "__".join(stem_parts)

        html_path = None
        if save_html:
            html_path = html_out_dir / f"{file_stem}.html"
            figure.write_html(str(html_path), include_plotlyjs="cdn")

        csv_path = None
        if save_gt_csv:
            csv_path = csv_out_dir / f"{file_stem}.csv"
            _save_raw_gt_points_csv(data_with_gt, csv_path, dataset_name, spec)

        rows.append({
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "label": spec.label,
            "iref": spec.iref,
            "tref": spec.tref,
            "ohref": spec.ohref,
            "i_range": list(spec.i_range),
            "t_range": list(spec.t_range),
            "oh_range": list(spec.oh_range),
            "n_gt_points": int(data_with_gt["gt_uref_raw"].notna().sum()),
            "html_file": None if html_path is None else str(html_path),
            "csv_file": None if csv_path is None else str(csv_path),
        })

    summary_df = pd.DataFrame(rows)
    summary_path = html_out_dir / f"{dataset_name}__gt_diagram_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved GT diagram summary: {summary_path}")
    return summary_df


def export_gt_diagrams_for_many_datasets(
    dataset_specs: list[dict[str, Any]],
    preprocess_output_dir: str,
    output_dir: str,
    preprocess_config: PreprocessConfig | None = None,
    save_html: bool = True,
    save_gt_csv: bool = True,
    csv_output_dir: str | None = None,
) -> pd.DataFrame:
    """Batch export GT diagrams across multiple datasets.

    Each entry in ``dataset_specs`` must contain:
        - ``dataset_path`` (str): Path to the parquet file.
        - ``references`` (list): Reference condition specs for that dataset.
        - ``dataset_name`` (str, optional): Override name.

    Args:
        dataset_specs: List of dataset specification dicts.
        preprocess_output_dir: Shared intermediate preprocessing directory.
        output_dir: Destination directory for HTML outputs and merged summary.
        preprocess_config: Shared preprocessing thresholds.
        save_html: Whether to write HTML overlays.
        save_gt_csv: Whether to write raw GT CSVs.
        csv_output_dir: Optional dedicated directory for raw GT CSV exports.
            If None, CSV files are written to ``output_dir``.

    Returns:
        Consolidated summary DataFrame, also saved as
        ``all_datasets__gt_diagram_summary.csv`` in ``output_dir``.
    """
    all_summaries: list[pd.DataFrame] = []
    for ds_spec in dataset_specs:
        summary_df = export_gt_diagrams_for_dataset(
            dataset_path=ds_spec["dataset_path"],
            preprocess_output_dir=preprocess_output_dir,
            output_dir=output_dir,
            reference_specs=ds_spec["references"],
            preprocess_config=preprocess_config,
            dataset_name_override=ds_spec.get("dataset_name"),
            save_html=save_html,
            save_gt_csv=save_gt_csv,
            csv_output_dir=csv_output_dir,
        )
        all_summaries.append(summary_df)

    if not all_summaries:
        return pd.DataFrame()

    merged = pd.concat(all_summaries, ignore_index=True)
    _ensure_dir(output_dir)
    merged_path = Path(output_dir) / "all_datasets__gt_diagram_summary.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged GT diagram summary: {merged_path}")
    return merged


def _example_main() -> None:
    """Minimal usage example configured via environment variables.

    Required environment variables:
        GT_DATASET_PATH
        GT_PREPROCESS_OUT
        GT_HTML_OUT

    Optional environment variables:
        GT_CSV_OUT (defaults to GT_HTML_OUT)
        GT_DATASET_NAME
    """
    import os

    dataset_path = os.environ.get("GT_DATASET_PATH")
    preprocess_out = os.environ.get("GT_PREPROCESS_OUT")
    html_out = os.environ.get("GT_HTML_OUT")
    csv_out = os.environ.get("GT_CSV_OUT")
    dataset_name_override = os.environ.get("GT_DATASET_NAME")

    required_vars = {
        "GT_DATASET_PATH": dataset_path,
        "GT_PREPROCESS_OUT": preprocess_out,
        "GT_HTML_OUT": html_out,
    }
    missing = [key for key, value in required_vars.items() if not value]
    if missing:
        missing_vars = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    dataset_path = str(dataset_path)
    preprocess_out = str(preprocess_out)
    html_out = str(html_out)
    csv_out = str(csv_out) if csv_out else None

    preprocess_cfg = PreprocessConfig()
    refs = [
        ReferenceConditionSpec(
            label="mid_load",
            iref=1.0,
            tref=60.0,
            ohref=72.0,
            i_range=(0.98, 1.02),
            t_range=(58.0, 62.0),
            oh_range=(24.0, 160.0),
        ),
        ReferenceConditionSpec(
            label="high_load",
            iref=1.5,
            tref=60.0,
            ohref=72.0,
            i_range=(1.46, 1.50),
            t_range=(58.0, 62.0),
            oh_range=(24.0, 160.0),
        ),
    ]
    summary = export_gt_diagrams_for_dataset(
        dataset_path=dataset_path,
        preprocess_output_dir=preprocess_out,
        output_dir=html_out,
        reference_specs=refs,
        preprocess_config=preprocess_cfg,
        dataset_name_override=dataset_name_override,
        save_html=False,
        save_gt_csv=True,
        csv_output_dir=csv_out,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    _example_main()
