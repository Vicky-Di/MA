"""Build ground-truth voltage files from a PEMFC dataset.

This module provides tools to extract, validate, and save GT voltage
time-series for one or more current-density reference points (Iref values).
Each extracted series is stored as both a CSV and a JSON metadata sidecar;
an interactive HTML overlay is generated so the user can visually confirm
that the chosen tolerance windows are appropriate.

Typical workflow (paired notebook: ``b_dataset_gt_distribution.ipynb``)
---------------------------------------------------------
1. Edit the configuration block inside ``main()`` (dataset path, Irefs, ranges).
2. Run ``generate_gt_files()`` (called automatically via ``main()``).
3. Inspect the HTML files written to *html_dir* and, if needed,
   narrow or widen the tolerance windows.

Human intervention point
^^^^^^^^^^^^^^^^^^^^^^^^
After this step a human opens each HTML file in a browser and deletes
any rows in the raw GT CSV that contain obvious artefacts (e.g., transient
spikes, stopped-stack readings).  The cleaned CSVs are then copied to
``gt_raw/`` for the next pipeline step.

Public API:
    generate_gt_files()   -- extract GT for multiple Iref values.
    main()                -- standalone entry-point with editable config.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from degradation_toolbox.Urc.Urc1 import Urc1
from degradation_toolbox.utils.ground_truth import extract_ground_truth, save_ground_truth_series

try:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess
except ImportError:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess


def _format_iref_token(iref: float) -> str:
    """Convert a current-density value to a filesystem-safe string token.

    Examples::

        _format_iref_token(1.0)  ->  "1"
        _format_iref_token(0.6)  ->  "0p6"
        _format_iref_token(1.5)  ->  "1p5"

    Args:
        iref: Current density [A/cm^2].

    Returns:
        Compact string representation with '.' replaced by 'p' and
        trailing zeros stripped.
    """
    return f"{float(iref):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _validate_range_pair(values: list[float] | tuple[float, float], name: str) -> tuple[float, float]:
    """Validate that *values* is a two-element (lo, hi) range with lo <= hi.

    Args:
        values: Sequence of exactly two numbers: lower and upper bound.
        name: Human-readable label for error messages.

    Returns:
        Validated ``(lo, hi)`` tuple.

    Raises:
        ValueError: If *values* does not contain exactly two numbers, or
            if lo > hi.
    """
    if len(values) != 2:
        raise ValueError(f"{name} requires exactly two numbers: [min, max]")
    lo, hi = float(values[0]), float(values[1])
    if lo > hi:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    return lo, hi


def _build_i_range_dict(
    irefs: list[float],
    i_ranges: list[list[float]] | list[tuple[float, float]],
) -> dict[float, tuple[float, float]]:
    """Zip Iref values with their respective tolerance windows.

    Args:
        irefs: List of target current-density values [A/cm^2].
        i_ranges: List of (lo, hi) pairs, one per entry in *irefs*.

    Returns:
        Dict mapping each Iref float to its validated ``(lo, hi)`` tuple.

    Raises:
        ValueError: If the lengths of *irefs* and *i_ranges* differ, or
            if any individual range is invalid.
    """
    if len(irefs) != len(i_ranges):
        raise ValueError("irefs and i_ranges must have the same length")

    out: dict[float, tuple[float, float]] = {}
    for iref, i_range in zip(irefs, i_ranges):
        out[float(iref)] = _validate_range_pair(i_range, f"i_ranges for Iref={iref}")
    return out


def _build_gt_figure(
    data_with_gt: pd.DataFrame,
    dataset_name: str,
    iref: float,
    i_range: tuple[float, float],
    t_range: tuple[float, float],
    oh_range: tuple[float, float],
) -> go.Figure:
    """Build an interactive two-panel Plotly figure for GT inspection.

    The top panel overlays all preprocessed voltage readings (grey dots)
    with identified raw GT points (red) and the interpolated GT series
    (green line).  The bottom panel shows a histogram of the raw GT
    voltage values.

    Args:
        data_with_gt: DataFrame produced by
            :func:`degradation_toolbox.utils.ground_truth.extract_ground_truth`;
            must contain columns ``voltage``, ``gt_uref_raw``, and
            ``gt_uref``.
        dataset_name: String label used in the figure title.
        iref: Reference current density [A/cm^2] used in this extraction.
        i_range: Actual (lo, hi) current-density tolerance window applied.
        t_range: Actual (lo, hi) temperature tolerance window applied.
        oh_range: Actual (lo, hi) operating-hours tolerance window applied.

    Returns:
        Interactive Plotly :class:`~plotly.graph_objects.Figure`.
    """
    gt_raw = data_with_gt["gt_uref_raw"].dropna()
    gt_interp = data_with_gt["gt_uref"].dropna()

    title_ref = (
        f"I in [{i_range[0]:.4f}, {i_range[1]:.4f}], "
        f"T in [{t_range[0]:.4f}, {t_range[1]:.4f}], "
        f"OH in [{oh_range[0]:.4f}, {oh_range[1]:.4f}]"
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.12,
        subplot_titles=(
            f"{dataset_name} GT @ Iref={iref} A/cm^2",
            "GT Voltage Distribution",
        ),
    )

    fig.add_trace(
        go.Scattergl(
            x=data_with_gt.index,
            y=data_with_gt["voltage"],
            mode="markers",
            marker=dict(size=2, color="rgba(160,160,160,0.25)"),
            name="All preprocessed voltage",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    if not gt_raw.empty:
        fig.add_trace(
            go.Scattergl(
                x=gt_raw.index,
                y=gt_raw.values,
                mode="markers",
                marker=dict(size=4, color="#C0392B"),
                name="Raw GT points",
            ),
            row=1,
            col=1,
        )

    if not gt_interp.empty:
        fig.add_trace(
            go.Scattergl(
                x=gt_interp.index,
                y=gt_interp.values,
                mode="lines",
                line=dict(width=2, color="#117A65"),
                name="Interpolated GT",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Histogram(
            x=gt_raw.values if not gt_raw.empty else [],
            nbinsx=80,
            marker=dict(color="#C0392B"),
            name="GT histogram",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{dataset_name} Ground Truth Extraction @ Iref={iref} A/cm^2<br><sup>{title_ref}</sup>",
        height=850,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Voltage [V]", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    return fig


def generate_gt_files(
    dataset_path: str,
    output_dir: str,
    html_dir: str,
    irefs: list[float],
    i_ranges: dict[float, tuple[float, float]],
    t_range: tuple[float, float],
    oh_range: tuple[float, float],
    t_ref: float | None,
    oh_ref: float | None,
    preprocess_config: dict,
    dataset_name_override: str | None = None,
) -> list[dict]:
    """Extract and save GT voltage files for multiple Iref values.

    For each Iref value the function:
    1. Calls :func:`~degradation_toolbox.utils.ground_truth.extract_ground_truth`
       with the specified tolerance windows.
    2. Saves the resulting GT series to ``output_dir`` as CSV + JSON sidecar.
    3. Writes an interactive HTML overlay to ``html_dir``.

    Args:
        dataset_path: Path to the raw parquet dataset file.
        output_dir: Directory where GT CSV and JSON files are saved
            (created if absent).
        html_dir: Directory where HTML inspection files are saved
            (created if absent).
        irefs: List of target current-density reference values [A/cm^2].
        i_ranges: Dict mapping each Iref to its (lo, hi) current tolerance.
        t_range: Global (lo, hi) temperature tolerance window [degC].
        oh_range: Global (lo, hi) operating-hours tolerance window [h].
        t_ref: Exact temperature reference point used for interpolation,
            or ``None`` to use the midpoint of *t_range*.
        oh_ref: Exact OH reference point, or ``None`` to use the
            midpoint of *oh_range*.
        preprocess_config: Dict of GMpreprocess and Urc1 filter parameters;
            must contain key ``"preprocess_output_dir"``.
        dataset_name_override: Optional name for the dataset; if ``None``
            the name is inferred from the file path by GMpreprocess.

    Returns:
        List of per-Iref summary dicts containing: ``dataset_name``,
        ``iref``, ``n_points``, ``csv_file``, ``json_file``,
        ``html_file``, ``i_range``, ``t_range``, ``oh_range``.
    """
    preprocessor = GMpreprocess(file_path=dataset_path, output_dir=preprocess_config["preprocess_output_dir"])
    data = preprocessor.run()
    dataset_name = dataset_name_override or preprocessor.name

    shared_pre = Urc1.preprocess_once(
        data,
        i_off=preprocess_config["i_off"],
        u_off=preprocess_config["u_off"],
        data_filter_i_min=preprocess_config["data_filter_i_min"],
        data_filter_U_min=preprocess_config["data_filter_U_min"],
        data_filter_U_max=preprocess_config["data_filter_U_max"],
        data_filter_T_min=preprocess_config["data_filter_T_min"],
        data_filter_T_max=preprocess_config["data_filter_T_max"],
        data_filter_h_since_last_start_min=preprocess_config["data_filter_h_since_last_start_min"],
    )

    output_root = Path(output_dir)
    html_root = Path(html_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    html_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for iref in irefs:
        i_range = i_ranges[float(iref)]
        ref_t = float(t_ref) if t_ref is not None else (t_range[0] + t_range[1]) / 2.0
        ref_oh = float(oh_ref) if oh_ref is not None else (oh_range[0] + oh_range[1]) / 2.0

        gt_uref, data_with_gt = extract_ground_truth(
            data=shared_pre.copy(),
            ref_I=float(iref),
            tol_I=i_range,
            ref_T=ref_t,
            tol_T=t_range,
            ref_OH=ref_oh,
            tol_OH=oh_range,
            dataset_name=dataset_name,
            plot=False,
        )
        iref_token = _format_iref_token(iref)
        # Keep filenames stable and filesystem-safe; ranges are stored in sidecar JSON metadata.
        gt_csv_path = output_root / f"{dataset_name}__gt__iref_{iref_token}.csv"
        html_path = html_root / f"{dataset_name}__gt__iref_{iref_token}.html"

        metadata = save_ground_truth_series(
            gt_uref=gt_uref,
            output_path=str(gt_csv_path),
            dataset_name=dataset_name,
            iref=float(iref),
            ref_T=ref_t,
            ref_OH=ref_oh,
            i_range=i_range,
            t_range=t_range,
            oh_range=oh_range,
            source="build_ground_truth_files.py",
        )

        fig = _build_gt_figure(
            data_with_gt=data_with_gt,
            dataset_name=dataset_name,
            iref=float(iref),
            i_range=i_range,
            t_range=t_range,
            oh_range=oh_range,
        )
        fig.write_html(str(html_path))

        summary.append(
            {
                "dataset_name": dataset_name,
                "iref": float(iref),
                "n_points": metadata["n_points"],
                "csv_file": str(gt_csv_path),
                "json_file": str(gt_csv_path.with_suffix('.json')),
                "html_file": str(html_path),
                "i_range": list(i_range),
                "t_range": list(t_range),
                "oh_range": list(oh_range),
            }
        )

    summary_path = output_root / f"{dataset_name}__gt__summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"\nSaved GT summary: {summary_path}")
    return summary


def main() -> None:
    """Standalone entry-point with an editable configuration block.

    Edit the block labelled ``# Edit this block`` at the top of the
    function body and then run this file directly::

        python build_ground_truth_files.py

    All paths are resolved relative to the project root so the script
    can be run from any working directory.

    Raises:
        FileNotFoundError: If the configured dataset file is not found.
    """
    # =====================================================================
    # Edit this block directly before running the script
    # =====================================================================
    dataset_path = r"explore_data/G1M1_new.parquet"
    dataset_name = "G1M1_new"

    irefs = [0.6, 1.0, 1.5]
    t_ref = 60.0
    oh_ref = 72.0
    # Empirical adjustment: a slightly wider lower-temperature bound improved GT stability.
    t_range = [58, 60]
    oh_range = [10.4575, 495.7186]
    # Empirical adjustment for high-load reference to increase usable GT points.
    i_ranges = [(0.58, 0.62), (0.98, 1.02), (1.46, 1.50)]

    gt_output_dir = r"master_arbeit_Di/ground_truth/output_backup/generated_gt"
    html_output_dir = r"master_arbeit_Di/ground_truth/output_backup/graph_html"
    preprocess_output_dir = r"explore_data/output"

    # Resolve paths relative to project root so running from any cwd works.
    project_root = Path(__file__).resolve().parents[2]

    dataset_path_resolved = Path(dataset_path)
    if not dataset_path_resolved.is_absolute():
        dataset_path_resolved = (project_root / dataset_path_resolved).resolve()

    gt_output_dir_resolved = Path(gt_output_dir)
    if not gt_output_dir_resolved.is_absolute():
        gt_output_dir_resolved = (project_root / gt_output_dir_resolved).resolve()

    html_output_dir_resolved = Path(html_output_dir)
    if not html_output_dir_resolved.is_absolute():
        html_output_dir_resolved = (project_root / html_output_dir_resolved).resolve()

    preprocess_output_dir_resolved = Path(preprocess_output_dir)
    if not preprocess_output_dir_resolved.is_absolute():
        preprocess_output_dir_resolved = (project_root / preprocess_output_dir_resolved).resolve()

    if not dataset_path_resolved.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path_resolved}\n"
            f"Current configured dataset_path: {dataset_path}"
        )

    print(f"[Path] project_root: {project_root}")
    print(f"[Path] dataset: {dataset_path_resolved}")
    print(f"[Path] gt_output_dir: {gt_output_dir_resolved}")
    print(f"[Path] html_output_dir: {html_output_dir_resolved}")
    print(f"[Path] preprocess_output_dir: {preprocess_output_dir_resolved}")

    preprocess_config = {
        "preprocess_output_dir": str(preprocess_output_dir_resolved),
        "i_off": 0.1,
        "u_off": 1.3,
        "data_filter_i_min": 0.1,
        "data_filter_U_min": 1.4,
        "data_filter_U_max": 2.3,
        "data_filter_T_min": 50.0,
        "data_filter_T_max": 65.0,
        "data_filter_h_since_last_start_min": 0.5,
    }

    # =====================================================================
    # Validation and execution
    # =====================================================================
    t_range = _validate_range_pair(t_range, "t_range")
    oh_range = _validate_range_pair(oh_range, "oh_range")
    if t_ref is not None and not (t_range[0] <= float(t_ref) <= t_range[1]):
        print(f"[Warn] t_ref={t_ref} is outside t_range={t_range}")
    if oh_ref is not None and not (oh_range[0] <= float(oh_ref) <= oh_range[1]):
        print(f"[Warn] oh_ref={oh_ref} is outside oh_range={oh_range}")
    i_range_dict = _build_i_range_dict(irefs, i_ranges)

    summary = generate_gt_files(
        dataset_path=str(dataset_path_resolved),
        output_dir=str(gt_output_dir_resolved),
        html_dir=str(html_output_dir_resolved),
        irefs=irefs,
        i_ranges=i_range_dict,
        t_range=t_range,
        oh_range=oh_range,
        t_ref=t_ref,
        oh_ref=oh_ref,
        preprocess_config=preprocess_config,
        dataset_name_override=dataset_name,
    )

    print("\nGround truth generation completed:")
    for item in summary:
        print(
            f"  Iref={item['iref']}: {item['n_points']} pts | "
            f"csv={item['csv_file']} | html={item['html_file']}"
        )


if __name__ == "__main__":
    main()
