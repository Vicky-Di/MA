"""
Urc1_cbrt_gpr_update.py

GPR-smoothed version of Urc1_c6_sigmoid with cbrt (cube root) nonlinear term.
Inherits from Urc1_c6_sigmoid and applies Gaussian Process Regression smoothing
to the fitted coefficients for improved generalization and uncertainty quantification.

Key differences from Urc1_gpr_update (which uses I2 term):
- Uses cbrt nonlinear term: c4*(I - c6)^(1/3)
- 6 parameters instead of 5
- c6 is optimized as the inflection point of the cbrt function
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

try:
    from degradation_toolbox.Urc.Urc1_c6_sigmoid import Urc1_c6_sigmoid, OCV
except ImportError:
    from Urc1_c6_sigmoid import Urc1_c6_sigmoid, OCV


class Urc1_cbrt_GPR(Urc1_c6_sigmoid):
    """
    Urc1_cbrt_GPR: GPR-smoothed voltage model using cbrt nonlinear term.
    
    Inherits from Urc1_c6_sigmoid with nonlinear_term="cbrt" and applies
    Gaussian Process Regression smoothing to the fitted coefficients.
    
    Attributes
    ----------
    nonlinear_term : str
        Fixed to "cbrt"
    gpr_r2_threshold : float
        R² threshold for filtering intervals used in GPR training
    gpr_cond_threshold : float
        Condition number threshold for filtering intervals
    l_init : float
        Initial length scale for RBF kernel (in hours)
    output_mode : str
        "all": keep all intervals (including recovered by GPR)
        "filtered": keep only originally good intervals with smoothed coefficients
    fitting_results_all : pd.DataFrame
        All intervals after GPR smoothing
    fitting_results_reliable : pd.DataFrame
        Filtered reliable intervals (used by comparator)
    """

    def __init__(self, data,
                 gpr_r2_threshold=0.8,
                 gpr_cond_threshold=1e6,
                 length_scale_init=500 * 24,
                 output_mode="all",
                 **kwargs):
        """
        Initialize Urc1_cbrt_gpr_update with GPR smoothing.

        Parameters
        ----------
        data : pd.DataFrame
            Input time series data with columns ["currentDensity", "temperature", "voltage"]
        gpr_r2_threshold : float, default 0.8
            R² threshold for selecting training intervals for GPR
        gpr_cond_threshold : float, default 1e6
            Condition number threshold for selecting training intervals
        length_scale_init : float, default 500*24
            Initial length scale for RBF kernel (hours)
        output_mode : str, default "all"
            "all": include recovered intervals
            "filtered": keep only originally good intervals
        **kwargs : dict
            Additional parameters passed to Urc1_c6_sigmoid
            - Must NOT include nonlinear_term (set to "cbrt" here)
        """
        # Force nonlinear_term to "cbrt"
        kwargs["nonlinear_term"] = "cbrt"

        self.gpr_r2_threshold = gpr_r2_threshold
        self.gpr_cond_threshold = gpr_cond_threshold
        self.l_init = length_scale_init
        self.output_mode = output_mode

        # 1. Initialize parent Baseline fitting (produces raw fitting_results with cbrt)
        super().__init__(data, **kwargs)

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

        # 4. Sync Urc1_results and degradation_results to GPR-smoothed data
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
        self.model_family = "Urc1_cbrt_gpr"
        self.model_label = "Urc1_cbrt_gpr"
        self.model_results = self.fitting_results_reliable.copy()
        self.Urc1_results_incl_unreliable = self.fitting_results_all.copy()

    def run_gpr_smoothing(self):
        """
        Apply Gaussian Process Regression to smooth the fitted cbrt coefficients.

        For each coefficient [c1, c2, c3, c4, c5, c6], fit a GPR model using
        intervals that pass quality checks, then predict smoothed values over
        the entire time horizon.

        Returns
        -------
        pd.DataFrame
            Updated fitting_results with columns:
            - c1_gpr, c2_gpr, c3_gpr, c4_gpr, c5_gpr, c6_gpr: smoothed coefficients
            - c1_gpr_se, ..., c6_gpr_se: standard errors from GPR
            - gpr_status: "Measured" (from training) or "Recovered" (predicted only)
        """
        df = self.fitting_results.copy()

        # Robust calh validation: coerce to numeric, drop ±inf
        calh_numeric = pd.to_numeric(df["calh"], errors='coerce').replace([np.inf, -np.inf], np.nan)
        valid_calh_mask = calh_numeric.notna()
        if not valid_calh_mask.any():
            print("Error: No valid 'calh' found in results. GPR cannot proceed.")
            return df

        X_all = calh_numeric[valid_calh_mask].values.reshape(-1, 1)

        # c6 is a special parameter: inflection point for cbrt
        # It should be smoother and may need different treatment
        coeffs = ["c1", "c2", "c3", "c4", "c5", "c6"]

        # Pre-initialize c_gpr columns with fallback
        for c in coeffs:
            df[f"{c}_gpr"] = df[c]
            df[f"{c}_gpr_se"] = np.nan

        measured_mask = pd.Series(False, index=df.index)

        for c in coeffs:
            # Extract training data (good intervals only)
            train_mask = (df["quality"] == "good") & \
                         (df["R2"] > self.gpr_r2_threshold) & \
                         (df["cond"] < self.gpr_cond_threshold) & \
                         valid_calh_mask

            train_df = df[train_mask].dropna(subset=[c])
            if len(train_df) < 5:
                # Not enough points to train GPR, skip this coefficient
                continue

            X_train = calh_numeric.loc[train_df.index].values.reshape(-1, 1)
            y_train = train_df[c].values

            # Use baseline-computed standard errors as GPR observation noise
            y_se = train_df[f"{c}_se"].fillna(np.nanmean(train_df[f"{c}_se"])).values

            # Kernel: RBF (captures trends) + WhiteKernel (tolerates observation noise)
            kernel = C(1.0, (1e-5, 1e3)) * RBF(
                length_scale=self.l_init,
                length_scale_bounds=(100 * 24, 1e8)
            ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 0.2))

            gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=y_se ** 2,
                n_restarts_optimizer=10,
                normalize_y=True
            )

            gpr.fit(X_train, y_train)

            # Predict over entire time horizon
            y_pred, y_std = gpr.predict(X_all, return_std=True)

            df.loc[valid_calh_mask, f"{c}_gpr"] = y_pred
            df.loc[valid_calh_mask, f"{c}_gpr_se"] = y_std
            measured_mask |= train_mask

        # Label data source
        df["gpr_status"] = "Recovered"
        df.loc[measured_mask, "gpr_status"] = "Measured"

        # Recalculate Urc with GPR-smoothed coefficients
        df = self._recalculate_urc_with_gpr(df)
        return df

    def _recalculate_urc_with_gpr(self, df):
        """
        Reconstruct reference voltage curves using GPR-corrected cbrt coefficients.

        For cbrt model: U = c1*I_s + c2*(IT)_s + c3*ln(h)_s + c4*(I_raw - c6)^(1/3) + c5

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns c1_gpr, ..., c6_gpr and their SEs

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with columns Urc_{i} and Urc_se_{i} for each Iref
        """
        # Pre-compute U scaler range factor
        u_scale_factor = self.scaler_U.max - self.scaler_U.min

        for ref_name, ref_cfg in self.ref_config.items():
            i = float(ref_cfg["Iref"])
            tref = float(ref_cfg["Tref"])
            ohref = float(ref_cfg["OHref"])
            # --- Step A: Compute feature weights (partial derivatives dU/dci) ---
            w1 = self.scaler_I.scale(i)
            w2 = self.scaler_IxT.scale(i * tref)
            w3 = self.scaler_log_h.scale(np.log(ohref))
            w5 = 1.0  # Intercept term c5 always has weight 1

            # For c4 and c6 (cbrt term), need special handling
            # cbrt(i_raw - c6)^(1/3), so d/dc4 = (i_raw - c6)^(1/3)
            #                            d/dc6 = c4 * (-1/3) * (i_raw - c6)^(-2/3)
            cbrt_term = np.cbrt(i - df["c6_gpr"])
            w4 = cbrt_term  # dU/dc4 = cbrt_term

            # For c6: dU/dc6 = c4 * d[cbrt(i - c6)]/dc6
            #                 = c4 * (-1/3) * (i - c6)^(-2/3)
            # Avoid division by zero
            cbrt_term_squared = np.power(np.abs(i - df["c6_gpr"]), 2.0 / 3.0) + 1e-10
            w6 = df["c4_gpr"] * (-1.0 / 3.0) / cbrt_term_squared

            # --- Step B: Compute predicted voltage ---
            u_ref_scaled = (
                df["c1_gpr"] * w1 +
                df["c2_gpr"] * w2 +
                df["c3_gpr"] * w3 +
                df["c4_gpr"] * cbrt_term +
                df["c5_gpr"] * w5
            )

            urc_series = self.scaler_U.unscale(u_ref_scaled) + OCV(tref)
            df[f"Urc_{i}"] = urc_series
            df[f"Urc_{ref_name}"] = urc_series

            # --- Step C: Apply error propagation law ---
            # Var(U_scaled) = sum( w_i^2 * Var(c_i) )
            var_u_scaled = (
                (w1 ** 2 * df["c1_gpr_se"] ** 2) +
                (w2 ** 2 * df["c2_gpr_se"] ** 2) +
                (w3 ** 2 * df["c3_gpr_se"] ** 2) +
                (w4 ** 2 * df["c4_gpr_se"] ** 2) +
                (w5 ** 2 * df["c5_gpr_se"] ** 2) +
                (w6 ** 2 * df["c6_gpr_se"] ** 2)
            )

            se_u_scaled = np.sqrt(var_u_scaled)
            urc_se_series = se_u_scaled * u_scale_factor
            df[f"Urc_se_{i}"] = urc_se_series
            df[f"Urc_se_{ref_name}"] = urc_se_series

        return df
