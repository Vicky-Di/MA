import datetime

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from degradation_toolbox.Urc.Urc1 import Urc1, func, OCV

class Urc1_Adaptive(Urc1):
    def __init__(self, data, 
                 # =================================================================
                 # 1. Structural Parameters
                 # =================================================================
                 initial_window_size=3,      # [Days] Initial window size
                                             # Recommended >= Urc1 Baseline interval length
                 
                 slide=1,                    # [Days] Sliding step size for target interval
                 
                 # =================================================================
                 # 2. [Strategy: Gatekeeper] Veto Parameters
                 # =================================================================
                 min_target_count_ratio=0.1,     # Minimum target interval data sparsity threshold (0.1 = 10%)
                                                 # If target interval data count < min_req * 10%,
                                                 # skip directly without any window expansion attempt.
                                                 
                 fixed_freshness_threshold=0.20, # Fixed freshness threshold (0.2 = 20%)
                                                 # Regardless of window size, target interval data must exceed
                                                 # 20% of total window data. Below this, historical data
                                                 # over-dilutes current features; force stop expansion.
                 
                 # =================================================================
                 # 3. [Strategy: Weighting] Weighting Parameters
                 # =================================================================
                 decay_rate=0.1,             # Temporal decay factor (WLS)
                                             # 0.1: moderate decay; 0.0: equal weights (OLS)
                 
                 # =================================================================
                 # 4. Hard Constraints
                 # =================================================================
                 threshold=1e6,              # Condition number threshold (primary filter)
                                             # High cond -> ill-conditioned matrix -> insufficient data
                                             # or low feature diversity -> triggers window expansion
                 
                 urc_min=1.4,                # [V] Urc physical lower bound
                 urc_max=2.4,                # [V] Urc physical upper bound
                 
                 **kwargs):
        
        # 1. Initialize subclass-specific parameters
        self.initial_window_size = initial_window_size
        self.min_target_count_ratio = min_target_count_ratio
        self.fixed_freshness_threshold = fixed_freshness_threshold
        self.decay_rate = decay_rate
        
        # Physical constraints
        self.urc_min = urc_min
        self.urc_max = urc_max

        # 2. Initialize parent class
        # Note: parent __init__ calls model_fitting, so parameters must be set beforehand
        super().__init__(data, slide=slide, threshold=threshold, **kwargs)
        
        print(f"Initialized Urc1_Adaptive (Simplified): Init_Win={self.initial_window_size}d")
        print(f"  Filters: Target_Min_Ratio={self.min_target_count_ratio}, Freshness={self.fixed_freshness_threshold}")

    
    def model_fitting(self):
        """
        Override parent model_fitting method.
        Implements a greedy search strategy: starts from the minimum window size
        and expands until finding the smallest dataset satisfying the condition
        number requirement.
        """
        # === Preparation ===
        start_timestamp = self.installation_time
        last_timestamp = self.data.index[-1]
        
        total_duration_days = (last_timestamp - start_timestamp).days
        num_intervals = int(total_duration_days // self.slide)

        print(f"Starting Adaptive Fitting for approx {num_intervals} intervals...")

        # === Initialize results DataFrame ===
        dates = [start_timestamp + datetime.timedelta(days=i * self.slide) for i in range(num_intervals + 5)] 
        index = pd.MultiIndex.from_product([dates, [self.name]], names=['date', 'name'])
        self.fitting_results = pd.DataFrame(index=index)

        # 1. Initialize numerical columns
        num_cols = [
            "c1", "c2", "c3", "c4", "c5", 
            "c1_se", "c2_se", "c3_se", "c4_se", "c5_se",
            "cond", "R2", "sigma2", 
            "freshness", "window_size_days", "num_points"
        ]
        for col in num_cols:
            self.fitting_results[col] = np.nan
        
        # 2. Initialize object/string columns
        obj_cols = ["adaptive_status", "quality", "XTX", "status"] 
        for col in obj_cols:
            self.fitting_results[col] = pd.Series([None] * len(index), index=index, dtype=object)

        # === Main loop: iterate over each target interval ===
        for interval in range(0, num_intervals + 2): 
            
            # Define target interval (the time point to solve for)
            target_start_date = start_timestamp + datetime.timedelta(days=interval * self.slide)
            target_end_date = target_start_date + datetime.timedelta(days=self.len_interval)
            
            # Boundary check
            if target_end_date > last_timestamp + datetime.timedelta(days=1): 
                break

            idx = (target_start_date, self.name)
            
            # Record basic time information
            self.fitting_results.loc[idx, "day_since_install"] = (target_start_date - self.installation_time).days
            self.fitting_results.loc[idx, "calh"] = (target_start_date - self.installation_time) / np.timedelta64(1, "h")

            # ==============================================================
            # [Step 0] Pre-Check: Target Data Sparsity
            # ==============================================================
            
            # 1. Extract target data (invariant; no need to wait for while loop)
            df_target = self.data[(self.data.index >= target_start_date) & 
                                  (self.data.index < target_end_date)]
            n_target = len(df_target)
            
            # 2. Compute absolute minimum requirement (e.g., 400 * 10% = 40 points)
            min_req_absolute = self.min_num_data_required_for_fit * self.min_target_count_ratio
            
            # 3. Check sparsity
            if n_target < min_req_absolute:
                # Record failure and skip
                status_msg = f"Skipped:TargetSparse({n_target}<{min_req_absolute:.0f})"
                self.fitting_results.loc[idx, "status"] = status_msg
                self.fitting_results.loc[idx, "adaptive_status"] = status_msg
                self.fitting_results.loc[idx, "quality"] = "dropped_sparse"
                self.fitting_results.loc[idx, "num_points"] = n_target
                
                # Skip to next interval without any window expansion attempt
                continue 

            # ==============================================================
            # Adaptive Expansion Logic
            # ==============================================================
            
            best_popt = None
            best_pcov = None
            best_quality_pass = False 
            
            final_stats = {} 
            start_time_of_window = None 

            w_size = self.initial_window_size
            
            # Compute expansion upper bound: window start cannot precede earliest data
            max_window_days = (target_end_date - start_timestamp).days + 1
            
            while True:
                # Safety valve: if w_size exceeds the days from data start to target_end,
                # the window already covers all data; further expansion adds no new points.
                if w_size > max_window_days:
                    status_msg = f"Exhausted: MaxWindow({max_window_days}d)reached"
                    final_stats = {'status': status_msg, 'quality': 'dropped_exhausted'}
                    break
                
                # 1. Define current lookback window
                window_start_date = target_end_date - datetime.timedelta(days=w_size)
                
                # df_target is already sliced above; only slice df_window here (includes historical data)
                df_window = self.data[(self.data.index >= window_start_date) & 
                                      (self.data.index < target_end_date)]
                
                n_window = len(df_window)
                
                # -------------------------------------------------------------
                # Step A: Gatekeeper (Freshness Check)
                # -------------------------------------------------------------
                
                # Compute freshness ratio
                if n_window == 0: 
                    actual_freshness = 0
                else: 
                    actual_freshness = n_target / n_window
                
                # [Core filter] If freshness is too low, historical data dominates;
                # further expansion is futile. Break and abandon this point.
                if actual_freshness < self.fixed_freshness_threshold:
                    status_msg = f"Diluted(Fr={actual_freshness:.2f}<{self.fixed_freshness_threshold})"
                    final_stats = {'status': status_msg, 'quality': 'dropped_freshness', 'freshness': actual_freshness}
                    break 

                # -------------------------------------------------------------
                # Step A.5: Data Sufficiency Check
                # -------------------------------------------------------------
                # Window data count below min_num_data_required_for_fit;
                # skip fitting and expand (insufficient data is solvable by expansion)
                if n_window < self.min_num_data_required_for_fit:
                    status_msg = f"Expand: InsufficientData({n_window}<{self.min_num_data_required_for_fit})"
                    final_stats = {
                        'status': status_msg, 
                        'quality': 'searching', 
                        'freshness': actual_freshness,
                        'num_points': n_window
                    }
                    w_size += 1
                    continue

                # -------------------------------------------------------------
                # Step B: Fitting & Math Check
                # -------------------------------------------------------------
                try:
                    # Prepare design matrices
                    X, y = self._prepare_matrices(df_window)
                    # Compute weights (WLS)
                    weights = self._calculate_weights(df_window, target_end_date)
                    sigma = 1.0 / (np.sqrt(weights) + 1e-6)

                    # Fit
                    popt, pcov = curve_fit(
                        func, X, y,
                        sigma=sigma, absolute_sigma=False,
                        bounds=((0, -np.inf, 0, -np.inf, -np.inf), (np.inf, 0, np.inf, np.inf, np.inf))
                    )
                    
                    # -------------------------------------------------------------
                    # Step C: Trigger (Cond & Physics)
                    # -------------------------------------------------------------
                    cond_number = np.linalg.cond(pcov)
                    physics_pass, fail_reason = self._check_physics(popt)
                    
                    # [Core decision]
                    # If cond is acceptable (well-conditioned) and physics passes,
                    # we found the minimum viable window. Break immediately (greedy).
                    if (cond_number < self.threshold) and physics_pass:
                        # >>> SUCCESS <<<
                        best_popt = popt
                        best_pcov = pcov
                        best_quality_pass = True
                        
                        U_pred = func(X, *popt)
                        r2 = r2_score(y, U_pred)
                        
                        final_stats = {
                            'r2': r2, 
                            'cond': cond_number, 
                            'freshness': actual_freshness,
                            'status': f'Success_w={w_size}',
                            'quality': 'good'
                        }
                        start_time_of_window = window_start_date
                        break  # Success: stop expanding
                    
                    else:
                        # >>> FAIL: continue expanding <<<
                        # Record failure reason
                        if cond_number >= self.threshold:
                            fail_detail = f"High Cond({cond_number:.1e})" 
                        else:
                            fail_detail = fail_reason
                        
                        status_msg = f"Expand: {fail_detail}"
                        final_stats = {
                            'status': status_msg, 
                            'cond': cond_number, 
                            'quality': 'searching',
                            'freshness': actual_freshness
                        }
                        
                        # [Fine search] Expand by 1 day
                        w_size += 1
                        continue

                except Exception as e:
                    # Fitting error
                    status_msg = f"Expand: FitError"
                    final_stats = {'status': status_msg, 'quality': 'searching', 'freshness': actual_freshness}
                    w_size += 1
                    continue
            
            # === Save results (Success or Last Attempt Fail) ===
            self.fitting_results.loc[idx, "freshness"] = final_stats.get('freshness', np.nan)
            self.fitting_results.loc[idx, "status"] = final_stats.get('status', 'Unknown')
            self.fitting_results.loc[idx, "adaptive_status"] = final_stats.get('status', 'Unknown')
            self.fitting_results.loc[idx, "quality"] = final_stats.get('quality', 'failed_adaptive')

            # If successful, save coefficients and statistics
            if best_quality_pass and best_popt is not None:
                self.fitting_results.loc[idx, "c1"] = best_popt[0]
                self.fitting_results.loc[idx, "c2"] = best_popt[1]
                self.fitting_results.loc[idx, "c3"] = best_popt[2]
                self.fitting_results.loc[idx, "c4"] = best_popt[3]
                self.fitting_results.loc[idx, "c5"] = best_popt[4]
                
                perr = np.sqrt(np.diag(best_pcov))
                self.fitting_results.loc[idx, "c1_se"] = perr[0]
                self.fitting_results.loc[idx, "c2_se"] = perr[1]
                self.fitting_results.loc[idx, "c3_se"] = perr[2]
                self.fitting_results.loc[idx, "c4_se"] = perr[3]
                self.fitting_results.loc[idx, "c5_se"] = perr[4]
                
                self.fitting_results.loc[idx, "cond"] = final_stats['cond']
                self.fitting_results.loc[idx, "R2"] = final_stats['r2']
                
                # Record actual window size in days
                self.fitting_results.loc[idx, "window_size_days"] = w_size
                self.fitting_results.loc[idx, "num_windows_used"] = w_size 
                
                if start_time_of_window:
                    self.fitting_results.loc[idx, "time_span_hours"] = (target_end_date - start_time_of_window).total_seconds() / 3600
                    
                    # XTX for Parent SE Calc
                    # Re-extract best window data for computation
                    df_best = self.data[(self.data.index >= start_time_of_window) & 
                                        (self.data.index < target_end_date)]
                    X_best, y_best = self._prepare_matrices(df_best)
                    X_const = np.c_[X_best, np.ones(len(X_best))]
                    XTX = X_const.T @ X_const
                    sigma2 = np.sum((y_best - func(X_best, *best_popt))**2) / (len(y_best) - X_const.shape[1])
                    
                    self.fitting_results.at[idx, "XTX"] = XTX
                    self.fitting_results.loc[idx, "sigma2"] = sigma2

        return self.fitting_results
    
    # --- Helper Functions ---
    def _calculate_weights(self, df, target_date):
        if self.decay_rate == 0: return np.ones(len(df))
        delta_days = (target_date - df.index).total_seconds() / (3600 * 24)
        delta_days = np.maximum(delta_days, 0)
        weights = np.exp(-self.decay_rate * delta_days)
        return weights

    def _check_physics(self, popt):
        """
        Physics check: verify that Urc at reference conditions is within a
        physically plausible range.
        """
        i_check = self.Iref[0]
        resolved = self._resolve_reference_input(i_check)
        if resolved is None:
            tref = self.Tref
            ohref = self.OHref
        else:
            _, tref, ohref, _ = resolved
        i_s = self.scaler_I.scale(i_check)
        ixt_s = self.scaler_IxT.scale(i_check * tref)
        logh_s = self.scaler_log_h.scale(np.log(ohref))
        i2_s = self.scaler_I2.scale(i_check**2)
        
        u_scaled = (popt[0]*i_s + popt[1]*ixt_s + popt[2]*logh_s + popt[3]*i2_s + popt[4])
        u_rc = self.scaler_U.unscale(u_scaled) + OCV(tref)
        
        if not (self.urc_min <= u_rc <= self.urc_max):
            return False, f"Urc{u_rc:.2f}V_Out"
        
        return True, "Pass"
    
    def Uref_se_calc(self, row, i):
        xtx_val = row.get("XTX")
        if self._coerce_xtx_matrix(xtx_val) is None: return np.nan
        return super().Uref_se_calc(row, i)