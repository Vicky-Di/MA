"""Urc1_GPR: Gaussian Process Regression coefficient smoothing.

Key idea:
- Runs baseline Urc1 (5-parameter per-window fits)
- Applies Gaussian Process Regression to smooth coefficient trajectories over time
- Estimates missing or unreliable coefficients using GPR interpolation
- Improves coefficient stability and generalization

Model remains the same:
    U = c1*I_s + c2*(IT)_s + c3*ln(h)_s + c4*I²_s + c5

Each of the 5 coefficients is smoothed independently via GPR,
using RBF kernel with initial length scale set by gpr_length_scale_init (hours).
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

try:
    from degradation_toolbox.Urc.Urc1 import Urc1, OCV
except ImportError:
    from Urc1 import Urc1, OCV

class Urc1_GPR(Urc1):
    def __init__(self, data, 
                 gpr_r2_threshold=0.8,    # R2 threshold for filtering good intervals
                 gpr_cond_threshold=1e6,  # Condition number threshold for filtering
                 length_scale_init=500*24,   # Length scale (hours), controls smoothing
                 output_mode="all",       # "all" or "filtered"
                 **kwargs):
        """
        Urc1_GPR: Smooths physical coefficients using Gaussian Process Regression.
        """
        # 1. Initialize and run parent Baseline fitting (produces raw fitting_results)
        super().__init__(data, **kwargs)
        
        self.gpr_r2_threshold = gpr_r2_threshold
        self.gpr_cond_threshold = gpr_cond_threshold
        self.l_init = length_scale_init
        self.output_mode = output_mode

        # 2. Run GPR smoothing and recovery
        self.fitting_results_all = self.run_gpr_smoothing()
        
        # 3. Update reliable result set based on output mode for downstream Comparator
        if self.output_mode == "filtered":
            # Keep only originally good intervals, but with smoothed coefficients
            mask = (self.fitting_results_all["quality"] == "good") & \
                   (self.fitting_results_all["R2"] > self.gpr_r2_threshold) & \
                   (self.fitting_results_all["cond"] < self.gpr_cond_threshold)
            self.fitting_results_reliable = self.fitting_results_all[mask].copy()
        else:
            # Keep all intervals (including recovered ones)
            self.fitting_results_reliable = self.fitting_results_all.copy()

        # 4. Sync Urc1_results and degradation_results to GPR-smoothed data.
        #    (Urc1.__init__ set these from the pre-GPR fitting_results_reliable;
        #     they must be refreshed so the Comparator sees GPR values, not base values.)
        self.Urc1_results = self.fitting_results_reliable.copy()
        self.degradation_results = {}
        self.degradation_results_Urc1 = {}
        self.degradation_results_by_ref = {}
        
        for i_ref in self.Iref:
            try:
                deg_result = self.get_degradation_rate(i_ref)
                if deg_result is not None:
                    self.degradation_results[i_ref] = deg_result
                    self.degradation_results_Urc1[i_ref] = deg_result
            except Exception:
                pass

        for ref_name in self.ref_name_to_i:
            try:
                deg_result = self.get_degradation_rate(ref_name)
                if deg_result is not None:
                    self.degradation_results_by_ref[ref_name] = deg_result
            except Exception:
                pass

        # Comparator interface attributes
        self.model_family = "Urc1_GPR"
        self.model_label = "Urc1_GPR"
        self.model_results = self.fitting_results_reliable.copy()
        self.Urc1_results_incl_unreliable = self.fitting_results_all.copy()

    def run_gpr_smoothing(self):
        df = self.fitting_results.copy()
        # Robust calh validation: coerce to numeric, drop ±inf
        calh_numeric = pd.to_numeric(df["calh"], errors='coerce').replace([np.inf, -np.inf], np.nan)
        valid_calh_mask = calh_numeric.notna()
        if not valid_calh_mask.any():
            print("Error: No valid 'calh' found in results. GPR cannot proceed.")
            return df

        X_all = calh_numeric[valid_calh_mask].values.reshape(-1, 1)

        coeffs = ["c1", "c2", "c3", "c4", "c5"]

        # Pre-initialize c_gpr columns with fallback so downstream never sees uninitialized NaN
        for c in coeffs:
            df[f"{c}_gpr"] = df[c]
            df[f"{c}_gpr_se"] = np.nan

        measured_mask = pd.Series(False, index=df.index)  # accumulate trained rows across coefficients

        for c in coeffs:
            # Extract training data
            train_mask = (df["quality"] == "good") & \
                     (df["R2"] > self.gpr_r2_threshold) & \
                     (df["cond"] < self.gpr_cond_threshold) & \
                     valid_calh_mask  # Use cleaned mask to ensure no NaN/inf in training set

            train_df = df[train_mask].dropna(subset=[c])
            if len(train_df) < 5:
                continue

            X_train = calh_numeric.loc[train_df.index].values.reshape(-1, 1)
            y_train = train_df[c].values
            
            # Use Baseline-computed standard errors (SE) as GPR observation noise (alpha)
            # This incorporates Baseline coefficient uncertainty as a prior
            y_se = train_df[f"{c}_se"].fillna(np.nanmean(train_df[f"{c}_se"])).values
            
            # Kernel: RBF (captures trends) + WhiteKernel (tolerates observation noise)
            kernel =  C(1.0, (1e-5, 1e3)) * RBF(length_scale=self.l_init, length_scale_bounds=(100*24, 1e8)) + \
                     WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 0.2))
            
            gpr = GaussianProcessRegressor(
                kernel=kernel, 
                alpha=y_se**2,  # Key: leverage Baseline uncertainty
                n_restarts_optimizer=10,
                normalize_y=True         
            )
            
            gpr.fit(X_train, y_train)
            
            # Predict over entire time horizon (including bad intervals)
            y_pred, y_std = gpr.predict(X_all, return_std=True)
            
            df.loc[valid_calh_mask, f"{c}_gpr"] = y_pred
            df.loc[valid_calh_mask, f"{c}_gpr_se"] = y_std
            measured_mask |= train_mask  # Accumulate training flags across all coefficients

        # Label data source (uses accumulated measured_mask, not just the last coefficient's)
        df["gpr_status"] = "Recovered"
        df.loc[measured_mask, "gpr_status"] = "Measured"

        # Recalculate Urc with GPR-smoothed coefficients
        df = self._recalculate_urc_with_gpr(df)
        return df

    def _recalculate_urc_with_gpr(self, df):
        """Reconstruct reference voltage curves using GPR-corrected coefficients and their uncertainties."""
        
        # Pre-compute U scaler range factor (to convert scaled-space SE back to physical units [V])
        u_scale_factor = self.scaler_U.max - self.scaler_U.min

        for ref_name, ref_cfg in self.ref_config.items():
            i = float(ref_cfg["Iref"])
            tref = float(ref_cfg["Tref"])
            ohref = float(ref_cfg["OHref"])
            # --- Step A: Compute feature weights ---
            # These are the partial derivatives dU/dci
            w1 = self.scaler_I.scale(i)
            w2 = self.scaler_IxT.scale(i * tref)
            w3 = self.scaler_log_h.scale(np.log(ohref))
            w4 = self.scaler_I2.scale(i**2)
            w5 = 1.0  # Intercept term c5 always has weight 1
            
            # --- Step B: Compute predicted voltage ---
            u_ref_scaled = (
                df["c1_gpr"] * w1 +
                df["c2_gpr"] * w2 +
                df["c3_gpr"] * w3 +
                df["c4_gpr"] * w4 +
                df["c5_gpr"] * w5
            )
            
            # Inverse-scale back to physical voltage and add OCV
            urc_series = self.scaler_U.unscale(u_ref_scaled) + OCV(tref)
            df[f"Urc_{i}"] = urc_series
            df[f"Urc_{ref_name}"] = urc_series
            
            # --- Step C: Apply error propagation law to compute SE ---
            # Var(U_scaled) = sum( w_i^2 * Var(c_i) )
            # where Var(c_i) = df["ci_gpr_se"]**2
            
            var_u_scaled = (
                (w1**2 * df["c1_gpr_se"]**2) +
                (w2**2 * df["c2_gpr_se"]**2) +
                (w3**2 * df["c3_gpr_se"]**2) +
                (w4**2 * df["c4_gpr_se"]**2) +
                (w5**2 * df["c5_gpr_se"]**2)
            )
            
            # SE is the square root of variance
            se_u_scaled = np.sqrt(var_u_scaled)
            
            # Convert scaled-space SE back to physical units [V]
            # Note: SE is a delta quantity, so only multiply by scale factor (no offset)
            urc_se_series = se_u_scaled * u_scale_factor
            df[f"Urc_se_{i}"] = urc_se_series
            df[f"Urc_se_{ref_name}"] = urc_se_series
            
        return df