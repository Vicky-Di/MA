"""Reference-condition distribution analysis for GT candidate selection.

This module analyses the distribution of candidate (Iref, Tref, OHref)
observations in a preprocessed fuel-cell dataset and identifies the
highest-density regions — the most promising reference-condition windows
for subsequent ground-truth extraction.

Typical workflow
----------------
1. Run ``analyze_reference_condition_distribution()`` from the notebook
    ``a_1-dataset_distribution.ipynb``.
2. Inspect the 1-D histograms (HTML) and 3-D scatter (HTML) that are
    saved to ``output_backup/output_histrogram/``.
3. Use the printed hotspot table to choose concrete (Iref, Tref, OHref)
   targets and tolerance windows for the next pipeline step.

Human intervention point
^^^^^^^^^^^^^^^^^^^^^^^^
After this step a human reviews the distribution plots and picks the
target reference conditions.  Those chosen values are then configured
in ``b_dataset_gt_distribution.ipynb``.

Main public API:
    PreprocessConfig             -- dataclass: GMpreprocess + Urc1 filter params.
    HistogramConfig              -- dataclass: histogram binning and display params.
    load_preprocessed_reference_data()  -- preprocess raw parquet and filter data.
    build_reference_dataframe()         -- rename columns to Iref/Tref/OHref.
    summarize_hotspot_reference_conditions() -- rank highest-density 3-D bins.
    plot_1d_distributions()             -- 1-D histograms per reference dimension.
    plot_3d_histogram()                 -- 3-D scatter-bubble histogram.
    analyze_reference_condition_distribution() -- end-to-end entry-point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from degradation_toolbox.Urc.Urc1 import Urc1

try:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess
except ImportError:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess


@dataclass
class PreprocessConfig:
    """Configuration for the GMpreprocess + Urc1 preprocessing step.

    Attributes:
        i_off: Minimum current density offset for the initial
            current filter [A/cm²].
        u_off: Minimum voltage offset for the initial voltage filter [V].
        resample: Resampling factor (1 = no resampling).
        data_filter_i_min: Hard lower limit on current density [A/cm²].
        data_filter_U_min: Hard lower voltage bound [V].
        data_filter_U_max: Hard upper voltage bound [V].
        data_filter_T_min: Hard lower temperature bound [°C].
        data_filter_T_max: Hard upper temperature bound [°C].
        data_filter_h_since_last_start_min: Discard the early warm-up
            transient shorter than this many hours after stack start [h].
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
class HistogramConfig:
    """Configuration for the histogram plots and hotspot analysis.

    Attributes:
        bins_i: Number of bins for the 1-D current-density histogram.
        bins_t: Number of bins for the 1-D temperature histogram.
        bins_oh: Number of bins for the 1-D operating-hours histogram.
        bins_3d_i: Number of bins along the Iref axis of the 3-D histogram.
        bins_3d_t: Number of bins along the Tref axis of the 3-D histogram.
        bins_3d_oh: Number of bins along the OHref axis of the 3-D histogram.
        use_log_oh_for_3d: Bin the OHref axis in log-scale in the 3-D
            histogram.  Recommended because OH spans several orders of
            magnitude.
        min_count_for_3d: Minimum sample count to display a 3-D bin
            (lower-count bins are hidden to reduce visual clutter).
        top_k_hotspots: Number of highest-density 3-D bins to report in
            the hotspot summary table.
    """

    bins_i: int = 60
    bins_t: int = 60
    bins_oh: int = 60
    bins_3d_i: int = 20
    bins_3d_t: int = 20
    bins_3d_oh: int = 20
    use_log_oh_for_3d: bool = True
    min_count_for_3d: int = 10
    top_k_hotspots: int = 15


def _ensure_dir(path: str | Path) -> Path:
    """Create *path* (and any missing parents) and return a ``Path`` object.

    Args:
        path: Target directory as a string or :class:`~pathlib.Path`.

    Returns:
        Resolved ``Path`` object for the created directory.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _normalize_index_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert the DataFrame index to ``DatetimeIndex``.

    Tries direct :func:`pandas.to_datetime` first, then a millisecond
    epoch conversion.  Returns the original frame unmodified if both
    conversions fail.

    Args:
        df: Input DataFrame whose index may be numeric or string timestamps.

    Returns:
        Copy of ``df`` with a ``DatetimeIndex`` where possible.
    """
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out

    try:
        out.index = pd.to_datetime(out.index)
        return out
    except Exception:
        pass

    try:
        out.index = pd.to_datetime(out.index, unit="ms")
        return out
    except Exception:
        return out


def load_preprocessed_reference_data(
    dataset_path: str,
    preprocess_output_dir: str,
    preprocess_config: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load dataset, run GMpreprocess, then apply Urc1.preprocess_once filtering.

    Returns:
        preprocessed dataframe containing currentDensity, temperature, voltage,
        h_since_last_start and a dataset name.
    """
    cfg = preprocess_config or PreprocessConfig()

    pre = GMpreprocess(file_path=dataset_path, output_dir=preprocess_output_dir)
    raw_df = pre.run()
    if raw_df is None or raw_df.empty:
        raise ValueError("GMpreprocess returned empty data.")

    filtered = Urc1.preprocess_once(
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

    filtered = _normalize_index_to_datetime(filtered)
    return filtered, pre.name


def build_reference_dataframe(preprocessed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean dataframe with candidate reference-condition dimensions:
      Iref candidate  -> currentDensity
      Tref candidate  -> temperature
      OHref candidate -> h_since_last_start
    """
    required = ["currentDensity", "temperature", "h_since_last_start"]
    missing = [c for c in required if c not in preprocessed_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for distribution analysis: {missing}")

    ref_df = preprocessed_df[required].copy()
    ref_df = ref_df.rename(
        columns={
            "currentDensity": "Iref",
            "temperature": "Tref",
            "h_since_last_start": "OHref",
        }
    )
    ref_df = ref_df.replace([np.inf, -np.inf], np.nan).dropna()
    return ref_df


def _compute_histogramdd(
    ref_df: pd.DataFrame,
    bins: tuple[int, int, int],
    use_log_oh: bool,
) -> tuple[np.ndarray, list[np.ndarray], pd.DataFrame]:
    """Compute a 3-D histogram over (Iref, Tref, OHref).

    Args:
        ref_df: DataFrame with columns ``Iref``, ``Tref``, ``OHref``.
        bins: Number of bins for (Iref, Tref, OHref) axes.
        use_log_oh: If ``True``, the OHref column is log-transformed
            before binning so that the log-uniform distribution is
            captured more evenly.

    Returns:
        Tuple of:
        - ``hist``: 3-D numpy array of bin counts.
        - ``edges``: List of three edge arrays (one per axis).
        - ``work``: Possibly log-transformed working DataFrame.
    """
    work = ref_df[["Iref", "Tref", "OHref"]].copy()
    if use_log_oh:
        work = work[work["OHref"] > 0].copy()
        work["OHref"] = np.log(work["OHref"])

    hist, edges = np.histogramdd(
        work[["Iref", "Tref", "OHref"]].to_numpy(),
        bins=bins,
    )
    return hist, edges, work


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    """Return the midpoint of each bin defined by *edges*.

    Args:
        edges: 1-D array of bin edges with length N+1 for N bins.

    Returns:
        1-D array of N bin-center values.
    """
    return (edges[:-1] + edges[1:]) / 2.0


def summarize_hotspot_reference_conditions(
    ref_df: pd.DataFrame,
    bins_3d: tuple[int, int, int] = (20, 20, 20),
    top_k: int = 15,
    use_log_oh: bool = True,
) -> pd.DataFrame:
    """Find top high-density 3D bins as candidate reference-condition hotspots."""
    hist, edges, _ = _compute_histogramdd(ref_df, bins=bins_3d, use_log_oh=use_log_oh)
    if hist.sum() <= 0:
        return pd.DataFrame(columns=["rank", "count", "Iref", "Tref", "OHref"])

    flat = hist.ravel()
    non_zero_idx = np.flatnonzero(flat > 0)
    if len(non_zero_idx) == 0:
        return pd.DataFrame(columns=["rank", "count", "Iref", "Tref", "OHref"])

    top_k = min(top_k, len(non_zero_idx))
    candidate_idx = non_zero_idx[np.argpartition(flat[non_zero_idx], -top_k)[-top_k:]]
    candidate_idx = candidate_idx[np.argsort(flat[candidate_idx])[::-1]]

    i_centers = _bin_centers(edges[0])
    t_centers = _bin_centers(edges[1])
    oh_centers = _bin_centers(edges[2])

    rows = []
    for rank, flat_idx in enumerate(candidate_idx, start=1):
        i_idx, t_idx, oh_idx = np.unravel_index(flat_idx, hist.shape)
        oh_val = float(np.exp(oh_centers[oh_idx])) if use_log_oh else float(oh_centers[oh_idx])
        rows.append(
            {
                "rank": rank,
                "count": int(hist[i_idx, t_idx, oh_idx]),
                "Iref": float(i_centers[i_idx]),
                "Tref": float(t_centers[t_idx]),
                "OHref": oh_val,
            }
        )

    return pd.DataFrame(rows)


def plot_1d_distributions(
    ref_df: pd.DataFrame,
    hist_cfg: HistogramConfig,
    dataset_name: str,
) -> go.Figure:
    """Build a 1-row, 3-column Plotly figure with 1-D histograms.

    Each subplot shows the marginal distribution of one reference
    dimension: current density (Iref), temperature (Tref), and
    operating hours (OHref).

    Args:
        ref_df: DataFrame with columns ``Iref``, ``Tref``, ``OHref``.
        hist_cfg: Binning configuration (uses ``bins_i``, ``bins_t``,
            ``bins_oh``).
        dataset_name: String label used in the figure title.

    Returns:
        Interactive Plotly :class:`~plotly.graph_objects.Figure`.
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Iref Distribution", "Tref Distribution", "OHref Distribution"),
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Histogram(x=ref_df["Iref"], nbinsx=hist_cfg.bins_i, marker=dict(color="#1f77b4"), name="Iref"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=ref_df["Tref"], nbinsx=hist_cfg.bins_t, marker=dict(color="#2ca02c"), name="Tref"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=ref_df["OHref"], nbinsx=hist_cfg.bins_oh, marker=dict(color="#d62728"), name="OHref"),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=f"{dataset_name} Reference Condition 1D Distributions",
        template="plotly_white",
        bargap=0.04,
        showlegend=False,
        height=450,
    )
    fig.update_xaxes(title_text="Iref [A/cm2]", row=1, col=1)
    fig.update_xaxes(title_text="Tref [degC]", row=1, col=2)
    fig.update_xaxes(title_text="OHref [h]", row=1, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    return fig


def plot_3d_histogram(
    ref_df: pd.DataFrame,
    hist_cfg: HistogramConfig,
    dataset_name: str,
) -> tuple[go.Figure, pd.DataFrame]:
    """Build an interactive 3-D scatter-bubble histogram.

    Each bubble represents one 3-D bin; bubble size and colour encode
    sample count.  Bins with fewer than ``hist_cfg.min_count_for_3d``
    samples are omitted to avoid clutter.

    Args:
        ref_df: DataFrame with columns ``Iref``, ``Tref``, ``OHref``.
        hist_cfg: Binning and display configuration.
        dataset_name: String label used in the figure title.

    Returns:
        Tuple of:
        - Interactive Plotly :class:`~plotly.graph_objects.Figure`.
        - DataFrame of visible bins with columns
          ``Iref``, ``Tref``, ``OHref``, ``count``.
    """
    hist, edges, _ = _compute_histogramdd(
        ref_df,
        bins=(hist_cfg.bins_3d_i, hist_cfg.bins_3d_t, hist_cfg.bins_3d_oh),
        use_log_oh=hist_cfg.use_log_oh_for_3d,
    )

    i_centers = _bin_centers(edges[0])
    t_centers = _bin_centers(edges[1])
    oh_centers = _bin_centers(edges[2])

    coords = np.array(np.unravel_index(np.arange(hist.size), hist.shape)).T
    counts = hist.ravel().astype(int)
    mask = counts >= hist_cfg.min_count_for_3d
    coords = coords[mask]
    counts = counts[mask]

    if len(counts) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=(
                f"{dataset_name} 3D Histogram (no bins >= {hist_cfg.min_count_for_3d} counts). "
                "Lower min_count_for_3d."
            ),
            template="plotly_white",
        )
        return fig, pd.DataFrame(columns=["Iref", "Tref", "OHref", "count"])

    i_vals = i_centers[coords[:, 0]]
    t_vals = t_centers[coords[:, 1]]
    if hist_cfg.use_log_oh_for_3d:
        oh_vals = np.exp(oh_centers[coords[:, 2]])
    else:
        oh_vals = oh_centers[coords[:, 2]]

    size = 6.0 + 22.0 * (counts - counts.min()) / max(1, (counts.max() - counts.min()))

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=i_vals,
                y=t_vals,
                z=oh_vals,
                mode="markers",
                marker=dict(
                    size=size,
                    color=counts,
                    colorscale="Viridis",
                    opacity=0.85,
                    colorbar=dict(title="Count"),
                ),
                text=[f"count={c}" for c in counts],
                hovertemplate=(
                    "Iref=%{x:.4f}<br>"
                    "Tref=%{y:.2f}<br>"
                    "OHref=%{z:.2f}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        ]
    )

    oh_axis_title = "OHref [h]"
    if hist_cfg.use_log_oh_for_3d:
        oh_axis_title += " (binned in log scale)"

    fig.update_layout(
        title=f"{dataset_name} 3D Histogram of Candidate Reference Conditions",
        template="plotly_white",
        scene=dict(
            xaxis_title="Iref [A/cm2]",
            yaxis_title="Tref [degC]",
            zaxis_title=oh_axis_title,
        ),
        height=720,
    )

    bins_df = pd.DataFrame(
        {
            "Iref": i_vals,
            "Tref": t_vals,
            "OHref": oh_vals,
            "count": counts,
        }
    ).sort_values("count", ascending=False)

    return fig, bins_df


def analyze_reference_condition_distribution(
    dataset_path: str,
    preprocess_output_dir: str,
    analysis_output_dir: str,
    preprocess_config: PreprocessConfig | None = None,
    hist_config: HistogramConfig | None = None,
    save_html: bool = True,
    save_csv: bool = True,
) -> dict[str, Any]:
    """End-to-end reference-condition distribution analysis.

    Preprocesses the dataset, builds 1-D and 3-D histograms, ranks the
    top hotspot bins, and (optionally) saves all outputs to disk.

    Args:
        dataset_path: Path to the raw parquet dataset file.
        preprocess_output_dir: Directory for GMpreprocess intermediate
            outputs.
        analysis_output_dir: Directory where HTML and CSV outputs are
            written (created if absent).
        preprocess_config: Preprocessing parameters; defaults to
            :class:`PreprocessConfig` with no arguments.
        hist_config: Histogram parameters; defaults to
            :class:`HistogramConfig` with no arguments.
        save_html: Write 1-D and 3-D histogram HTML files.
        save_csv: Write filtered reference DataFrame, 3-D bin table,
            hotspot table, and quantile summary as CSV.

    Returns:
        Dict with keys: ``dataset_name``, ``preprocessed_df``,
        ``reference_df``, ``fig_1d``, ``fig_3d``, ``bins_df``,
        ``hotspot_df``, ``quantile_summary``, ``output_dir``.

    Raises:
        ValueError: If GMpreprocess returns empty data.
        KeyError: If required columns are absent after preprocessing.
    """
    hcfg = hist_config or HistogramConfig()

    preprocessed_df, dataset_name = load_preprocessed_reference_data(
        dataset_path=dataset_path,
        preprocess_output_dir=preprocess_output_dir,
        preprocess_config=preprocess_config,
    )
    ref_df = build_reference_dataframe(preprocessed_df)

    fig_1d = plot_1d_distributions(ref_df, hcfg, dataset_name)
    fig_3d, bins_df = plot_3d_histogram(ref_df, hcfg, dataset_name)

    hotspot_df = summarize_hotspot_reference_conditions(
        ref_df,
        bins_3d=(hcfg.bins_3d_i, hcfg.bins_3d_t, hcfg.bins_3d_oh),
        top_k=hcfg.top_k_hotspots,
        use_log_oh=hcfg.use_log_oh_for_3d,
    )

    quantile_summary = ref_df[["Iref", "Tref", "OHref"]].quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    out_dir = _ensure_dir(analysis_output_dir)
    if save_html:
        fig_1d.write_html(str(out_dir / f"{dataset_name}__ref_dist_1d.html"))
        fig_3d.write_html(str(out_dir / f"{dataset_name}__ref_dist_3d.html"))

    if save_csv:
        ref_df.to_csv(out_dir / f"{dataset_name}__ref_candidates_filtered.csv", index=True)
        bins_df.to_csv(out_dir / f"{dataset_name}__ref_bins_3d.csv", index=False)
        hotspot_df.to_csv(out_dir / f"{dataset_name}__top_ref_hotspots.csv", index=False)
        quantile_summary.to_csv(out_dir / f"{dataset_name}__ref_quantiles.csv", index=True)

    print("=" * 80)
    print("Reference Condition Distribution Analysis")
    print("=" * 80)
    print(f"Dataset path              : {dataset_path}")
    print(f"Dataset name              : {dataset_name}")
    print(f"Filtered data size        : {len(ref_df)}")
    print(f"Preprocess output dir     : {preprocess_output_dir}")
    print(f"Analysis output dir       : {out_dir}")
    print("-" * 80)
    print("Top dense reference-condition bins:")
    if hotspot_df.empty:
        print("  No hotspot bins found. Try lower min_count_for_3d or fewer bins.")
    else:
        for _, row in hotspot_df.iterrows():
            print(
                "  "
                f"#{int(row['rank']):02d} | count={int(row['count']):6d} | "
                f"Iref={row['Iref']:.4f} | Tref={row['Tref']:.2f} | OHref={row['OHref']:.2f}"
            )
    print("=" * 80)

    return {
        "dataset_name": dataset_name,
        "preprocessed_df": preprocessed_df,
        "reference_df": ref_df,
        "fig_1d": fig_1d,
        "fig_3d": fig_3d,
        "bins_df": bins_df,
        "hotspot_df": hotspot_df,
        "quantile_summary": quantile_summary,
        "output_dir": str(out_dir),
    }


def _example_main() -> None:
    # Example: edit these paths/configs and run as a script.
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[2]  # project root
    result = analyze_reference_condition_distribution(
        dataset_path=str(_root / "explore_data" / "G6M2.parquet"),
        preprocess_output_dir=str(_root / "explore_data" / "output"),
        analysis_output_dir=str(_Path(__file__).parent / "output_backup" / "output_histrogram"),
        preprocess_config=PreprocessConfig(
            data_filter_i_min=0.1,
            data_filter_U_min=1.4,
            data_filter_U_max=2.3,
            data_filter_T_min=50.0,
            data_filter_T_max=65.0,
            data_filter_h_since_last_start_min=0.5,
        ),
        hist_config=HistogramConfig(
            bins_i=80,
            bins_t=60,
            bins_oh=80,
            bins_3d_i=22,
            bins_3d_t=18,
            bins_3d_oh=18,
            use_log_oh_for_3d=True,
            min_count_for_3d=12,
            top_k_hotspots=20,
        ),
        save_html=True,
        save_csv=True,
    )

    # In script mode, also show the top rows for quick check.
    print("\nTop hotspot rows:")
    print(result["hotspot_df"].head(10).to_string(index=False))


if __name__ == "__main__":
    _example_main()
