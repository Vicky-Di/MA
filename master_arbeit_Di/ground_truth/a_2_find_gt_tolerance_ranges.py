"""Sensitivity-based GT tolerance estimation (canonical version).

This is the up-to-date, well-documented version for deriving defensible
I/T/OH tolerance windows around a PEMFC target reference point.

ALGORITHM
----------
Physical model::

    U ≈ c1·I + c2·I·T + c3·ln(OH) + c4·I² + c5

Sensitivities at the reference point::

    dU/dI        = c1 + c2·T + 2·c4·I
    dU/dT        = c2·I  (+ dOCV/dT)
    dU/d(ln OH)  = c3

Given a voltage error budget (default ±10 mV) each sensitivity is
inverted to produce a half-width tolerance in the corresponding
physical variable.

NOTE
----
Canonical names in the current workflow are
``a_2_find_gt_tolerance_ranges.py`` and
``a_2_find_gt_tolerance_ranges.ipynb``.
The legacy compatibility module ``gt_tolerance_utils.py`` still exposes
``run_range_estimation()`` for older references.

Public API:
    fit_baseline_and_collect_coeffs()
    select_high_quality_intervals()
    estimate_ref_tolerances()
    run_range_estimation()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess
except ImportError:
    from master_arbeit_Di.explore.GMpreprocess import GMpreprocess

from degradation_toolbox.Urc.Urc1 import Urc1


def _safe_div(err_v: float, sensitivity: float) -> float:
    """Return range half-width from voltage error and sensitivity."""
    s = abs(float(sensitivity))
    if s < 1e-12:
        return np.inf
    return err_v / s


def fit_baseline_and_collect_coeffs(
    dataset_path: str,
    output_dir: str,
    model_config: dict,
) -> tuple[Urc1, pd.DataFrame]:
    """
    Run baseline Urc1 and return fitting results with back-calculated coefficients.
    """
    pre = GMpreprocess(file_path=dataset_path, output_dir=output_dir)
    data = pre.run()
    if data is None or data.empty:
        raise ValueError("Preprocessing failed or returned empty data.")

    model = Urc1(data=data, name=f"{pre.name}_baseline", **model_config)

    # Convert scaled coefficients to physical units (V based)
    fr = model.back_calculate_coefficients().copy()
    return model, fr


def select_high_quality_intervals(
    fr: pd.DataFrame,
    r2_min: float = 0.95,
    rmse_max_v: float | None = None,
    cond_max: float | None = None,
) -> pd.DataFrame:
    """
    Filter high-quality intervals from fitting results.
    """
    need_cols = ["R2", "back_calc_c1", "back_calc_c2", "back_calc_c3", "back_calc_c4", "back_calc_c5"]
    missing = [c for c in need_cols if c not in fr.columns]
    if missing:
        raise KeyError(f"Missing required columns in fitting_results: {missing}")

    mask = fr["R2"].notna() & (fr["R2"] >= r2_min)

    if rmse_max_v is not None and "RMSE" in fr.columns:
        mask &= fr["RMSE"].notna() & (fr["RMSE"] <= rmse_max_v)

    if cond_max is not None and "cond" in fr.columns:
        mask &= fr["cond"].notna() & (fr["cond"] <= cond_max)

    out = fr.loc[mask].copy()
    out = out.dropna(subset=["back_calc_c1", "back_calc_c2", "back_calc_c3", "back_calc_c4", "back_calc_c5"])
    return out


def estimate_ref_tolerances(
    coeffs_median: dict,
    i_ref: float,
    t_ref: float,
    oh_ref: float,
    voltage_error_mv: float = 10.0,
    include_ocv_temp_slope: bool = True,
    i_bounds: tuple[float, float] = (0.0, 2.5),
    t_bounds: tuple[float, float] = (40.0, 90.0),
    oh_bounds: tuple[float, float] = (1.0, 500.0),
) -> dict:
    """
    Compute suggested narrow ranges from sensitivity and target voltage error.

    Model form (physical coefficients):
        U ≈ c1*I + c2*I*T + c3*ln(OH) + c4*I^2 + c5

    Sensitivities at reference point:
        dU/dI  = c1 + c2*T + 2*c4*I
        dU/dT  = c2*I (+ dOCV/dT optionally)
        dU/dln(OH) = c3
    """
    c1 = float(coeffs_median["back_calc_c1"])
    c2 = float(coeffs_median["back_calc_c2"])
    c3 = float(coeffs_median["back_calc_c3"])
    c4 = float(coeffs_median["back_calc_c4"])

    dU_dI = c1 + c2 * t_ref + 2.0 * c4 * i_ref

    dU_dT = c2 * i_ref
    if include_ocv_temp_slope:
        # OCV(T) approx slope in toolbox: -8.2975e-4 V/°C
        dU_dT += -8.2975e-4

    dU_dlnOH = c3

    err_v = voltage_error_mv / 1000.0

    delta_I = _safe_div(err_v, dU_dI)
    delta_T = _safe_div(err_v, dU_dT)

    i_lo = max(i_bounds[0], i_ref - delta_I) if np.isfinite(delta_I) else i_bounds[0]
    i_hi = min(i_bounds[1], i_ref + delta_I) if np.isfinite(delta_I) else i_bounds[1]

    t_lo = max(t_bounds[0], t_ref - delta_T) if np.isfinite(delta_T) else t_bounds[0]
    t_hi = min(t_bounds[1], t_ref + delta_T) if np.isfinite(delta_T) else t_bounds[1]

    # For OH, solve in log-domain first and then map back with exp().
    delta_ln_oh = _safe_div(err_v, dU_dlnOH)
    if not np.isfinite(delta_ln_oh):
        oh_lo_unconstrained = oh_bounds[0]
        oh_hi_unconstrained = oh_bounds[1]
    else:
        oh_lo_unconstrained = oh_ref * np.exp(-delta_ln_oh)
        oh_hi_unconstrained = oh_ref * np.exp(delta_ln_oh)
    
    oh_lo = max(oh_bounds[0], oh_lo_unconstrained)
    oh_hi = min(oh_bounds[1], oh_hi_unconstrained)

    return {
        "sensitivities": {
            "dU_dI_V_per_Acm2": dU_dI,
            "dU_dT_V_per_degC": dU_dT,
            "dU_dlnOH_V": dU_dlnOH,
        },
        "suggested_ranges": {
            "I_range": (i_lo, i_hi),
            "T_range": (t_lo, t_hi),
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
):
    """End-to-end helper: baseline -> quality filter -> median coeff -> suggested ranges."""

    common_config = {
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

    model, fr = fit_baseline_and_collect_coeffs(dataset_path, output_dir, common_config)

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

    res = estimate_ref_tolerances(
        coeffs_median=med,
        i_ref=i_ref,
        t_ref=t_ref,
        oh_ref=oh_ref,
        voltage_error_mv=voltage_error_mv,
    )

    print("=" * 72)
    print("GT Narrow Range Estimation")
    print("=" * 72)
    print(f"Dataset             : {dataset_path}")
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

    print(f"\nTarget error budget: ±{voltage_error_mv:.2f} mV")
    print("Suggested narrow ranges:")
    i_lo, i_hi = res["suggested_ranges"]["I_range"]
    t_lo, t_hi = res["suggested_ranges"]["T_range"]
    h_lo, h_hi = res["suggested_ranges"]["OH_range"]
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


if __name__ == "__main__":
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]   # project root
    run_range_estimation(
        dataset_path=str(_root / "explore_data" / "G1M1_new.parquet"),
        output_dir=str(_root / "explore_data" / "output"),
        i_ref=1.0,
        t_ref=60.0,
        oh_ref=72.0,
        voltage_error_mv=10.0,
        r2_min=0.95,
        rmse_max_mv=None,
        cond_max=None,
    )
