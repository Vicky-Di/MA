import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, Union, Optional, List
try:
    from .cal_monotonicity import calculate_rank_monotonicity, calculate_sign_monotonicity
except ImportError:
    from master_arbeit_Di.utils.cal_monotonicity import calculate_rank_monotonicity, calculate_sign_monotonicity

# ==================== NEW: Condition Number Metrics ====================

def extract_all_condition_numbers(urc_instance: Any) -> np.ndarray:
    """
    Extract all condition numbers from model_results, including filtered-out intervals.
    
    Args:
        urc_instance: Urc1 or similar instance with model_results attribute
        
    Returns:
        np.ndarray: Array of valid (non-nan, non-inf) condition numbers
    """
    cond_source = None

    # Preferred: direct model_results (custom wrappers/proxies may provide this)
    if hasattr(urc_instance, 'model_results') and isinstance(getattr(urc_instance, 'model_results'), pd.DataFrame):
        cond_source = urc_instance.model_results
    # Urc wrapper: underlying Urc1 full results
    elif hasattr(urc_instance, 'urc_instance') and hasattr(urc_instance.urc_instance, 'fitting_results'):
        cond_source = urc_instance.urc_instance.fitting_results
    # Urc wrapper/proxy fallback: full results alias
    elif hasattr(urc_instance, 'Urc1_results_incl_unreliable'):
        cond_source = urc_instance.Urc1_results_incl_unreliable
    # Last fallback: reliable table (may be biased but better than empty)
    elif hasattr(urc_instance, 'fitting_results_reliable'):
        cond_source = urc_instance.fitting_results_reliable

    if cond_source is None or 'cond' not in cond_source.columns:
        return np.array([])

    all_conds = cond_source['cond'].dropna()
    # Filter out inf values
    all_conds = all_conds[~np.isinf(all_conds)]
    return all_conds.values


def calculate_condition_number_metrics(
    urc_instance: Any,
    baseline_instance: Optional[Any] = None
) -> Dict[str, Union[float, int]]:
    """
    Calculate condition number statistics across all intervals.
    
    Args:
        urc_instance: Urc1 instance with model_results
        baseline_instance: Optional baseline Urc1 instance for win-rate comparison
        
    Returns:
        Dict with keys: 
            - "Cond Median log10" (main indicator)
            - "Cond P95 log10" (tail risk)
            - "Cond Mean", "Cond Min", "Cond Max" (supplementary)
            - "Cond WinRate vs Baseline (%)" (if baseline provided)
    """
    all_conds = extract_all_condition_numbers(urc_instance)
    
    if len(all_conds) < 2:
        return {
            "Cond Median log10": np.nan,
            "Cond P95 log10": np.nan,
            "Cond Count": len(all_conds),
        }
    
    # Use log10 for robust comparison across orders of magnitude
    log10_conds = np.log10(all_conds)
    
    metrics = {
        "Cond Median log10": round(np.median(log10_conds), 3),  # Main: robust to outliers
        "Cond P95 log10": round(np.percentile(log10_conds, 95), 3),  # Tail risk
        "Cond Mean": round(np.mean(all_conds), 1),  # Supplementary
        "Cond Min": round(np.min(all_conds), 1),
        "Cond Max": round(np.max(all_conds), 1),
        "Cond Count": len(all_conds),
    }
    
    # Calculate win-rate if baseline provided
    if baseline_instance is not None:
        baseline_conds = extract_all_condition_numbers(baseline_instance)
        if len(baseline_conds) > 1:
            baseline_log10 = np.log10(baseline_conds)
            current_log10 = np.log10(all_conds)
            
            # Paired comparison on intervals
            min_len = min(len(baseline_log10), len(current_log10))
            wins = (current_log10[:min_len] < baseline_log10[:min_len]).sum()
            win_rate = (wins / min_len * 100) if min_len > 0 else np.nan
            metrics["Cond WinRate vs Baseline (%)"] = round(win_rate, 1)
    
    return metrics

# ======================================================================

def evaluate_urc_stability(
    urc_instance: Any, 
    i_target: float, 
    outlier_threshold_method: Union[str, float] = '3rmse'
) -> Optional[Dict[str, Union[float, int]]]:
    """
    Calculates and evaluates stability metrics (6 key metrics) for the Urc model 
    at a specific target current density.

    Args:
        urc_instance: An instance of the executed Urc or Urc1 class.
        i_target (float): Target current density (e.g., 1.5).
        outlier_threshold_method (Union[str, float]): Criterion for determining outliers.
            - '2rmse': Deviations > 2 * RMSE 
            - '3rmse': Deviations > 3 * RMSE.(default).
            - float: Absolute deviation in mV (e.g., 10.0).

    Returns:
        Optional[Dict]: A dictionary containing the 6 metrics, or None if data is missing.
    """
    
    # --- 1. Retrieve Data and Regression Model ---
    
    # Find the closest reference current
    closest_i = min(urc_instance.Iref, key=lambda x: abs(x - i_target))
    
    # Check for Urc1/Urc1BaseHuber classes (use fitting_results_reliable directly)
    if hasattr(urc_instance, 'fitting_results_reliable'):
        df = urc_instance.fitting_results_reliable.copy()
        if df.empty:
            print(f"⚠️ Warning: No reliable results found for {i_target} A/cm2")
            return None
        
        # Extract Urc data for the target current
        urc_col = f"Urc_{closest_i}"
        se_col = f"Urc_se_{closest_i}"
        
        if urc_col not in df.columns:
            print(f"⚠️ Warning: Column {urc_col} not found in fitting_results_reliable")
            return None
        
        # Create standardized dataframe
        df_standard = pd.DataFrame({
            'calh': df['calh'].values,
            'Urc': df[urc_col].values,
            'Urc_se': df[se_col].values if se_col in df.columns else np.nan
        })
        
        # For Urc1, calculate linear regression from the data itself
        from scipy.stats import linregress
        mask = df_standard['Urc'].notna() & df_standard['calh'].notna()
        if mask.sum() < 2:
            print(f"⚠️ Warning: Not enough data points for {closest_i} A/cm2")
            return None
        
        reg = linregress(df_standard.loc[mask, 'calh'], df_standard.loc[mask, 'Urc'])
        slope_v_per_h = reg.slope
        intercept_v = reg.intercept
        slope_stderr_v_per_h = reg.stderr  # Standard error of slope in V/h
        
    # Check for old Urc class (use get_Urc_results method)
    elif hasattr(urc_instance, 'get_Urc_results'):
        df_standard = urc_instance.get_Urc_results(i_target)
        
        if df_standard is None or df_standard.empty:
            print(f"⚠️ Warning: No Urc results found for {i_target} A/cm2")
            return None
        
        if closest_i not in urc_instance.degradation_results:
            print(f"⚠️ Warning: No regression model found for {closest_i} A/cm2")
            return None

        reg_result = urc_instance.degradation_results[closest_i]
        slope_v_per_h = reg_result.aging_uV_per_h / 1e6
        intercept_v = reg_result.bol_V
        slope_stderr_v_per_h = np.sqrt(reg_result.uncertainty_uV2) / 1e6  # Convert to V/h
    else:
        print(f"⚠️ Error: Unsupported instance type")
        return None
    
    df = df_standard
    
    # --- 2. Calculate Core Vectors ---
    
    y_true = df['Urc']  # Actual values (V)
    y_pred = df['calh'] * slope_v_per_h + intercept_v  # Predicted values (V)
    residuals_v = y_true - y_pred  # Residuals (V)
    residuals_mv = residuals_v * 1000  # Residuals (mV)

    # --- 3. Calculate the 6 Key Metrics ---
    
    # Metric 1: RMSE (Global Fit Quality)
    # Average dispersion of data points from the regression line.
    mse = mean_squared_error(y_true, y_pred)
    rmse_mv = np.sqrt(mse) * 1000 
    n = len(y_true)
    if n > 2:
        correction_factor = np.sqrt(n / (n - 2))
        rmse_adj_mv = rmse_mv * correction_factor
    else:
        rmse_adj_mv = np.inf 
        print(f"Warning: Not enough data points (n={n}) to calculate adjusted RMSE.")
    
    # Metric 2: Slope Uncertainty (Confidence in Degradation Rate)
    # How confident are we in the calculated degradation rate?
    # Convert to uV/h for display
    slope_sigma_uv_h = slope_stderr_v_per_h * 1e6
    
    # Metric 3: Monotonicity (Global Trend Consistency)
    # Rank: Checks global trend consistency (High is good)
    mono_rank = calculate_rank_monotonicity(y_true)

    # Metric 4: Outlier Count
    
    # Number of points significantly deviating from the trend.
    if outlier_threshold_method == '2rmse':
        limit_mv = 2 * rmse_mv
    elif outlier_threshold_method == '3rmse':
        limit_mv = 3 * rmse_mv
    elif isinstance(outlier_threshold_method, (int, float)):
        limit_mv = float(outlier_threshold_method)
    else:
        # Fallback default
        limit_mv = 2 * rmse_mv
        
    outlier_count = (residuals_mv.abs() > limit_mv).sum()
    outlier_percentage = (outlier_count / n * 100) if n > 0 else 0  # Percentage of outliers
    
    # Metric 5: Max Residual (Worst Case Scenario)
    # How far is the furthest point from the regression line?
    max_res_mv = residuals_mv.abs().max()
    
    # Metric 6: Mean SE (Average Individual Point Confidence)
    # How accurate does the algorithm think its own individual points are?
    # Urc_se is in V, convert to mV
    mean_se_mv = df['Urc_se'].mean() * 1000

    # --- 4. Package Results ---
    metrics = {
        "Target Current (A/cm2)": i_target,
        "Data Points (n)": n,
        "RMSE (mV)": round(rmse_mv, 3),
        "Slope Sigma (uV/h)": round(slope_sigma_uv_h, 3),
        "Mono (Rank) [0-1]": round(mono_rank, 3),
        "Outlier Count": outlier_count,
        "Outlier (%)": round(outlier_percentage, 2),
        "Max Residual (mV)": round(max_res_mv, 3),
        "Mean SE (mV)": round(mean_se_mv, 3)
    }
    
    return metrics


# ==================== NEW: Ground Truth RMSE Metrics ====================

def calculate_gt_rmse(
    urc_instance: Any,
    gt_uref: pd.Series,
    i_target: float,
) -> Dict[str, Union[float, int]]:
    """
    Calculate RMSE between model Urc and ground truth (utrue) for each valid interval.

    For each reliable interval, utrue is defined as the **median** of all ground
    truth values whose timestamps fall within that interval's time window
    [interval_start, interval_start + len_interval].

    Args:
        urc_instance: Urc1 (or subclass) instance with fitting_results_reliable.
        gt_uref: pd.Series with DatetimeIndex, values = ground truth voltage [V].
            This should already be at the reference condition corresponding to i_target.
        i_target: Target reference current density [A/cm²].

    Returns:
        Dict with keys:
            - "GT RMSE (mV)": RMSE between Urc and utrue across valid intervals
            - "GT MAE (mV)": MAE between Urc and utrue
            - "GT Valid Intervals": number of intervals with both Urc and utrue
    """
    # ---- Identify Urc column for this Iref ----
    closest_i = min(urc_instance.Iref, key=lambda x: abs(x - i_target))
    urc_col = f"Urc_{closest_i}"

    # ---- Get reliable intervals ----
    if hasattr(urc_instance, 'fitting_results_reliable'):
        df_rel = urc_instance.fitting_results_reliable.copy()
    elif hasattr(urc_instance, 'Urc1_results'):
        # Urc wrapper object
        df_rel = urc_instance.Urc1_results.copy()
    elif hasattr(urc_instance, 'urc_instance') and hasattr(urc_instance.urc_instance, 'fitting_results_reliable'):
        # Wrapper with nested Urc1 instance
        df_rel = urc_instance.urc_instance.fitting_results_reliable.copy()
    else:
        return {"GT RMSE (mV)": np.nan, "GT MAE (mV)": np.nan, "GT Valid Intervals": 0}

    if df_rel.empty or urc_col not in df_rel.columns:
        return {"GT RMSE (mV)": np.nan, "GT MAE (mV)": np.nan, "GT Valid Intervals": 0}

    # ---- Get interval length (in days) ----
    len_interval_days = getattr(urc_instance, 'len_interval', 2)
    interval_td = datetime.timedelta(days=len_interval_days)

    # ---- Ensure gt_uref has DatetimeIndex ----
    if not isinstance(gt_uref.index, pd.DatetimeIndex):
        gt_uref = gt_uref.copy()
        gt_uref.index = pd.to_datetime(gt_uref.index)

    # ---- Iterate over reliable intervals ----
    urc_values = []
    utrue_values = []

    for _, row in df_rel.iterrows():
        # Interval start timestamp
        interval_start = row.get('date', None)
        if interval_start is None:
            # Some result tables use timestamp instead of date
            interval_start = row.get('timestamp', None)
        if interval_start is None:
            continue
        if not isinstance(interval_start, pd.Timestamp):
            interval_start = pd.Timestamp(interval_start)

        interval_end = interval_start + interval_td

        # Get Urc value for this interval
        urc_val = row.get(urc_col, np.nan)
        if pd.isna(urc_val):
            continue

        # Get GT values within the interval window
        mask = (gt_uref.index >= interval_start) & (gt_uref.index <= interval_end)
        gt_in_interval = gt_uref.loc[mask]

        if len(gt_in_interval) == 0:
            continue

        # utrue = median of GT values in this interval
        utrue = gt_in_interval.median()

        urc_values.append(urc_val)
        utrue_values.append(utrue)

    # ---- Compute RMSE ----
    n_valid = len(urc_values)
    if n_valid < 1:
        return {"GT RMSE (mV)": np.nan, "GT MAE (mV)": np.nan, "GT Valid Intervals": 0}

    urc_arr = np.array(urc_values)
    utrue_arr = np.array(utrue_values)
    diff_mv = (urc_arr - utrue_arr) * 1000  # V → mV

    rmse_mv = np.sqrt(np.mean(diff_mv ** 2))
    mae_mv = np.mean(np.abs(diff_mv))

    return {
        "GT RMSE (mV)": round(rmse_mv, 3),
        "GT MAE (mV)": round(mae_mv, 3),
        "GT Valid Intervals": n_valid,
    }


def compare_urc_models(
    models_dict: Dict[str, Any], 
    i_target: float,
    baseline_name: Optional[str] = None,
    include_cond_metrics: bool = True,
    include_gt_metrics: bool = False,
    gt_uref: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Batch compares the performance of multiple Urc models.

    Args:
        models_dict (Dict[str, Any]): Dictionary mapping 'Model Name' to Urc instances.
        i_target (float): Target current density.
        baseline_name (Optional[str]): Name of baseline model for win-rate comparison.
            If provided, will compute "Cond WinRate vs Baseline (%)" for other models.
        include_cond_metrics (bool): Whether to include condition number metrics (default: True).
        include_gt_metrics (bool): Whether to include ground truth comparison metrics (default: False).
            When True, computes RMSE between each model's Urc and the median ground truth
            (utrue) within each valid interval.
        gt_uref (Optional[pd.Series]): Ground truth voltage Series with DatetimeIndex.
            Required when include_gt_metrics=True. Values should be voltage [V] at the
            reference condition corresponding to i_target.

    Returns:
        pd.DataFrame: A comparison table containing metrics for all models.
    """
    if include_gt_metrics and gt_uref is None:
        raise ValueError("gt_uref must be provided when include_gt_metrics=True")

    results = []
    baseline_instance = None
    
    # Extract baseline if specified
    if baseline_name is not None and baseline_name in models_dict:
        baseline_instance = models_dict[baseline_name]
    
    for name, model in models_dict.items():
        metrics = evaluate_urc_stability(model, i_target)
        if metrics:
            metrics['Model Name'] = name
            
            # Add condition number metrics
            if include_cond_metrics:
                baseline_for_comparison = baseline_instance if name != baseline_name else None
                cond_metrics = calculate_condition_number_metrics(model, baseline_for_comparison)
                metrics.update(cond_metrics)

            # Add ground truth RMSE metrics
            if include_gt_metrics:
                gt_metrics = calculate_gt_rmse(model, gt_uref, i_target)
                metrics.update(gt_metrics)
            
            results.append(metrics)
            
    if not results:
        return pd.DataFrame()
        
    # Create DataFrame and reorder columns to put 'Model Name' first
    df_res = pd.DataFrame(results)
    cols = ['Model Name'] + [c for c in df_res.columns if c != 'Model Name']
    
    return df_res[cols]