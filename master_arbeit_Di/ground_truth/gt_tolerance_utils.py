"""Sensitivity-based tolerance estimation for GT reference conditions.

This module provides tools to derive defensible I/T/OH tolerance windows
around a target reference point, based on the fitted physical model coefficients
from a baseline Urc1 run.

DEPRECATION NOTE
----------------
This file is retained as a backward-compatibility copy.
The canonical implementation is ``a_2_find_gt_tolerance_ranges.py`` and the
canonical notebook is ``a_2_find_gt_tolerance_ranges.ipynb``.  Both modules
expose the same public function ``run_range_estimation()``.

Algorithm summary
-----------------
Model form (physical coefficients)::

    U ≈ c1*I + c2*I*T + c3*ln(OH) + c4*I² + c5

Sensitivities at reference point::

    dU/dI        = c1 + c2*T + 2*c4*I
    dU/dT        = c2*I  (+dOCV/dT for open-circuit temperature correction)
    dU/d(ln OH)  = c3

Given a voltage error budget (default ±10 mV), each sensitivity is inverted
to produce a half-width tolerance in the corresponding physical variable.

Main public API:
    fit_baseline_and_collect_coeffs()   Train baseline Urc1 and back-calculate coefficients.
    select_high_quality_intervals()     Filter to high-R² / low-RMSE intervals.
    estimate_ref_tolerances()           Compute sensitivity-based tolerance ranges.
    run_range_estimation()              End-to-end helper combining all three steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from degradation_toolbox.Urc.Urc1 import Urc1

try:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess
except ImportError:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_div(err_v: float, sensitivity: float) -> float:
    """Divide a voltage error budget by a sensitivity value safely.

    Args:
        err_v: Voltage error half-width [V].
        sensitivity: Physical sensitivity (dU/dX) at the reference point.

    Returns:
        Tolerance half-width in the X variable, or ``np.inf`` if the
        sensitivity is effectively zero (< 1e-12).
    """
    s = abs(float(sensitivity))
    if s < 1e-12:
        return np.inf
    return err_v / s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_baseline_and_collect_coeffs(
    dataset_path: str,
    output_dir: str,
    model_config: dict,
) -> tuple[Urc1, pd.DataFrame, str]:
    """Train a baseline Urc1 model and back-calculate physical coefficients.

    Args:
        dataset_path: Path to the input parquet file.
        output_dir: GMpreprocess intermediate output directory.
        model_config: Keyword-argument dict forwarded to :class:`Urc1`.

    Returns:
        Tuple of (model, fitting_results_with_physical_coeffs, dataset_name).

    Raises:
        ValueError: If GMpreprocess returns empty data.
    """
    pre = GMpreprocess(file_path=dataset_path, output_dir=output_dir)
    data = pre.run()
    if data is None or data.empty:
        raise ValueError("Preprocessing failed or returned empty data.")

    model = Urc1(data=data, name=f"{pre.name}_baseline", **model_config)
    # back_calculate_coefficients converts scaled regression coefficients to
    # physical units (V-based), stored in back_calc_c1 … back_calc_c5 columns.
    fr = model.back_calculate_coefficients().copy()
    return model, fr, pre.name


def select_high_quality_intervals(
    fr: pd.DataFrame,
    r2_min: float = 0.95,
    rmse_max_v: float | None = None,
    cond_max: float | None = None,
) -> pd.DataFrame:
    """Filter fitting results to retain only high-quality intervals.

    Quality criteria applied in sequence:
    1. R² ≥ ``r2_min``.
    2. RMSE ≤ ``rmse_max_v`` if provided.
    3. Condition number ≤ ``cond_max`` if provided.
    4. Drop rows where any back-calculated coefficient is NaN.

    Args:
        fr: Fitting results DataFrame from :func:`fit_baseline_and_collect_coeffs`.
        r2_min: Minimum acceptable R² value.
        rmse_max_v: Optional maximum RMSE [V].
        cond_max: Optional maximum condition number.

    Returns:
        Filtered DataFrame containing only high-quality intervals.

    Raises:
        KeyError: If required columns are missing from ``fr``.
    """
    need_cols = ["R2", "back_calc_c1", "back_calc_c2", "back_calc_c3", "back_calc_c4", "back_calc_c5"]
    missing = [c for c in need_cols if c not in fr.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    mask = fr["R2"].notna() & (fr["R2"] >= r2_min)
    if rmse_max_v is not None and "RMSE" in fr.columns:
        mask &= fr["RMSE"].notna() & (fr["RMSE"] <= rmse_max_v)
    if cond_max is not None and "cond" in fr.columns:
        mask &= fr["cond"].notna() & (fr["cond"] <= cond_max)

    out = fr.loc[mask].copy()
    return out.dropna(subset=["back_calc_c1", "back_calc_c2", "back_calc_c3", "back_calc_c4", "back_calc_c5"])


def estimate_ref_tolerances(
    coeffs_median: dict,
    i_ref: float,
    t_ref: float,
    oh_ref: float,
    voltage_error_mv: float = 10.0,
    include_ocv_temp_slope: bool = True,
) -> dict:
    """Compute suggested I/T/OH tolerance ranges from sensitivity analysis.

    Uses the physical model sensitivities at the reference point together with
    a voltage error budget to infer how much each variable can deviate before
    the voltage error exceeds the budget.

    OH tolerance is solved in log-space first (``dU/d(ln OH) = c3``), then
    mapped back to linear OH with ``exp()``.

    Args:
        coeffs_median: Dict of median physical coefficients (keys:
            back_calc_c1 … back_calc_c4; back_calc_c5 is not used here).
        i_ref: Reference current density [A/cm²].
        t_ref: Reference temperature [°C].
        oh_ref: Reference operating hours since last start [h].
        voltage_error_mv: Acceptable voltage deviation budget [mV].
        include_ocv_temp_slope: Whether to add the open-circuit voltage
            temperature coefficient (−8.30e-4 V/°C) to dU/dT.

    Returns:
        Dict with two keys:
        - ``sensitivities``: dU/dI, dU/dT, dU/dlnOH at reference point.
        - ``suggested_ranges``: (lo, hi) tuples for I, T, and OH.
    """
    c1 = float(coeffs_median["back_calc_c1"])
    c2 = float(coeffs_median["back_calc_c2"])
    c3 = float(coeffs_median["back_calc_c3"])
    c4 = float(coeffs_median["back_calc_c4"])

    # Partial derivatives at the reference point
    dU_dI = c1 + c2 * t_ref + 2.0 * c4 * i_ref
    dU_dT = c2 * i_ref
    if include_ocv_temp_slope:
        dU_dT += -8.2975e-4  # OCV(T) slope [V/°C] from toolbox constants
    dU_dlnOH = c3

    err_v = voltage_error_mv / 1000.0
    delta_I = _safe_div(err_v, dU_dI)
    delta_T = _safe_div(err_v, dU_dT)

    # OH: solve in log-domain, then exponentiate with physical clipping
    delta_ln_oh = _safe_div(err_v, dU_dlnOH)
    if not np.isfinite(delta_ln_oh):
        oh_lo, oh_hi = 1.0, 500.0
    else:
        oh_lo = max(1.0, oh_ref * np.exp(-delta_ln_oh))
        oh_hi = min(500.0, oh_ref * np.exp(delta_ln_oh))

    return {
        "sensitivities": {
            "dU_dI_V_per_Acm2": dU_dI,
            "dU_dT_V_per_degC": dU_dT,
            "dU_dlnOH_V": dU_dlnOH,
        },
        "suggested_ranges": {
            "I_range": (i_ref - delta_I, i_ref + delta_I),
            "T_range": (t_ref - delta_T, t_ref + delta_T),
            "OH_range": (oh_lo, oh_hi),
        },
    }


def run_range_estimation(
    dataset_path: str,
    output_dir: str,
    i_ref: float = 1.0,
    t_ref: float = 60.0,
    oh_ref: float = 72.0,
    voltage_error_mv: float = 10.0,
    r2_min: float = 0.95,
    rmse_max_mv: float | None = None,
    cond_max: float | None = None,
) -> dict:
    """End-to-end tolerance estimation: baseline fit → quality filter → tolerance.

    Convenience wrapper that chains :func:`fit_baseline_and_collect_coeffs`,
    :func:`select_high_quality_intervals`, and :func:`estimate_ref_tolerances`.

    Args:
        dataset_path: Path to the input parquet file.
        output_dir: GMpreprocess intermediate output directory.
        i_ref: Reference current density [A/cm²].
        t_ref: Reference temperature [°C].
        oh_ref: Reference operating hours [h].
        voltage_error_mv: Voltage error budget [mV].
        r2_min: Minimum R² for quality filtering.
        rmse_max_mv: Optional RMSE threshold [mV].
        cond_max: Optional condition-number threshold.

    Returns:
        Dict with keys: model, fitting_results, selected_results,
        median_coefficients, estimation.

    Raises:
        ValueError: If no high-quality intervals remain after filtering.
    """
    cfg = {
        "Iref": [0.6, 1.0, 1.5],
        "Tref": t_ref,
        "OHref": oh_ref,
        "len_interval": 2,
        "slide": 1,
        "min_num_data_required_for_fit": 300,
        "threshold": 1e6,
        "i_off": 0.1,
        "u_off": 1.3,
        "plot_fit": 0,
        "data_filter_i_min": 0.1,
        "data_filter_U_min": 1.4,
        "data_filter_U_max": 2.3,
        "data_filter_T_min": 50,
        "data_filter_T_max": 65,
    }

    model, fr, ds_name = fit_baseline_and_collect_coeffs(dataset_path, output_dir, cfg)
    rmse_max_v = (rmse_max_mv / 1000.0) if rmse_max_mv is not None else None
    fr_good = select_high_quality_intervals(fr, r2_min=r2_min, rmse_max_v=rmse_max_v, cond_max=cond_max)

    if fr_good.empty:
        raise ValueError("No high-quality intervals after filtering. Relax thresholds.")

    med = {
        "back_calc_c1": fr_good["back_calc_c1"].median(),
        "back_calc_c2": fr_good["back_calc_c2"].median(),
        "back_calc_c3": fr_good["back_calc_c3"].median(),
        "back_calc_c4": fr_good["back_calc_c4"].median(),
        "back_calc_c5": fr_good["back_calc_c5"].median(),
    }

    res = estimate_ref_tolerances(med, i_ref=i_ref, t_ref=t_ref, oh_ref=oh_ref, voltage_error_mv=voltage_error_mv)

    print("=" * 72)
    print("GT Narrow Range Estimation")
    print("=" * 72)
    print(f"Dataset             : {ds_name}")
    print(f"Intervals total     : {len(fr)}")
    print(f"Intervals selected  : {len(fr_good)} (R2 >= {r2_min})")
    if rmse_max_mv is not None:
        print(f"RMSE threshold      : <= {rmse_max_mv} mV")
    if cond_max is not None:
        print(f"Cond threshold      : <= {cond_max}")

    print("-" * 72)
    print("Median coefficients (physical):")
    for k, v in med.items():
        print(f"  {k:<14} = {v:.6g}")

    print("-" * 72)
    print("Sensitivities at reference point:")
    for k, v in res["sensitivities"].items():
        print(f"  {k:<22} = {v:.6g}")

    i_lo, i_hi = res["suggested_ranges"]["I_range"]
    t_lo, t_hi = res["suggested_ranges"]["T_range"]
    h_lo, h_hi = res["suggested_ranges"]["OH_range"]
    print(f"\nTarget error budget: ±{voltage_error_mv:.2f} mV")
    print("Suggested narrow ranges:")
    print(f"  I  range           = [{i_lo:.4f}, {i_hi:.4f}]")
    print(f"  T  range           = [{t_lo:.4f}, {t_hi:.4f}]")
    print(f"  OH range           = [{h_lo:.4f}, {h_hi:.4f}]")
    print("=" * 72)

    return {
        "model": model,
        "fitting_results": fr,
        "selected_results": fr_good,
        "median_coefficients": med,
        "estimation": res,
    }
