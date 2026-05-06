"""
Urc1_Surface: Global Surface Fitting with Time-varying Parameters.

Method 2 — Core idea:
- Treats all preprocessed data over the full experiment as a point cloud
  in multi-dimensional space.
- Constructs a global hypersurface that evolves with time *t*, capturing
  the performance degradation trajectory.
- Model: U = c1(t)*I + c2(t)*I*T + c3(t)*ln(OH) + c4(t)*I^2 + c5(t)
  where ci(t) = ai*t + bi (10 parameters total).

Key distinction from Urc1 Baseline:
- Baseline: each interval fits 5 coefficients independently.
- Surface:  single global fit of 10 parameters (evolutionary view).

Preprocessing is identical to Urc1 to ensure method comparability.
"""

import numpy as np
import pandas as pd
import datetime
import math
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Optional, Tuple, List

from degradation_toolbox.Urc.Urc1 import Urc1, func, OCV, MinMaxScalerCustomize


class Urc1_Surface(Urc1):
    """Global Surface Fitting with Time-varying Parameters (Method 2).

    Model equation:
        U(I, T, OH, t) = c1(t)*I + c2(t)*I*T + c3(t)*ln(OH) + c4(t)*I^2 + c5(t)

    Parameter evolution:
        ci(t) = ai*t + bi
        - bi: initial intercept (system state at t=0)
        - ai: evolution rate (degradation slope)

    Attributes:
        train_ratio (float): Train/test split ratio (default 0.8).
        coeffs (np.ndarray): 10 fitted parameters [a1, b1, a2, b2, ..., a5, b5].
        coeffs_se (np.ndarray): Parameter standard errors.
        test_metrics (dict): Test set evaluation metrics.
    """
    
    def __init__(
        self, 
        data, 
        train_ratio: float = 0.8,
        **kwargs
    ):
        """Initialize Urc1_Surface.

        Args:
            data: Input DataFrame with columns [currentDensity, temperature, voltage].
            train_ratio: Train/test split ratio (default 0.8).
            **kwargs: Forwarded to parent Urc1 (preprocessing params identical).
        """
        self.train_ratio = train_ratio

        # Result containers
        self.coeffs = None          # [a1, b1, a2, b2, ..., a5, b5]
        self.coeffs_se = None       # standard errors of 10 parameters
        self.test_metrics = {}      # test set evaluation metrics
        self.test_raw_data = {}     # point-by-point test data
        self.split_idx = None       # train/test split index
        
        # SE calculation cache (for XTX/sigma2 method)
        self._se_cache = None
        self.global_xtx = None      # Global XTX matrix from training data
        self.global_sigma2 = None   # Global sigma2 from test data

        # Parent __init__ calls data_preprocess() and model_fitting()
        super().__init__(data, **kwargs)

        self._calculate_test_metrics()

        self.degradation_results = {}
        self._calculate_degradation_rates()

        print(f"\nUrc1_Surface Initialized:")
        print(f"   Train Ratio: {self.train_ratio:.0%}")
        if self.test_metrics:
            print(f"   Test RMSE (Point): {self.test_metrics.get('test_rmse_mv', 'N/A'):.2f} mV")

    def _prepare_matrices(self, df):
        """[Override] Prepare feature matrices WITH constant column for global surface fitting.

        Unlike Urc1's per-interval fitting (which adds the constant later in _solve_interval_fit),
        the Surface model needs the constant column explicitly for global XTX/sigma2 SE calculation.

        Returns:
            X: Feature matrix shape (n_samples, 5) with constant column [I_s, IxT_s, log_h_s, I2_s, 1.0]
            y: Scaled voltage residuals
        """
        # 1. Feature Engineering
        df = df.copy()
        df["U_ocv"] = OCV(df.temperature)
        df["IxT"] = df.currentDensity * df.temperature
        df["log_h"] = np.log(df.h_since_last_start)
        df["I2"] = df.currentDensity ** 2

        # 2. Scaling (without constant)
        X_base = np.column_stack((
            self.scaler_I.scale(df.currentDensity),
            self.scaler_IxT.scale(df.IxT),
            self.scaler_log_h.scale(df.log_h),
            self.scaler_I2.scale(df.I2),
        ))

        # 3. Add constant column (required for proper XTX computation in SE calculation)
        X = np.column_stack((X_base, np.ones(len(X_base))))

        # 4. Target variable
        y = self.scaler_U.scale(df.voltage - df["U_ocv"])

        return X, y

    def model_fitting(self):
        """[Override] Global surface fitting, replacing per-interval loop.

        Steps:
            1. Split data into train/test sets.
            2. Build time-varying model and optimize globally.
            3. Generate Baseline-compatible interval results.

        Returns:
            pd.DataFrame: Fitting results.
        """
        print(f"\nStarting Global Surface Fitting (Train: {self.train_ratio:.0%})...")

        # Add cumulative hours (calh) for time-varying fitting
        self.data['calh'] = (self.data.index - self.installation_time) / np.timedelta64(1, 'h')

        # --- Step 1: Train/test split ---
        self.split_idx = int(len(self.data) * self.train_ratio)
        df_train = self.data.iloc[:self.split_idx]
        df_test = self.data.iloc[self.split_idx:]

        print(f"   Train: {len(df_train)} points | Test: {len(df_test)} points")

        # --- Step 2: Prepare feature matrices ---
        X_train, y_train = self._prepare_matrices(df_train)
        t_train = df_train['calh'].values

        # --- Step 3: Define time-varying residual function ---
        def surface_residuals(params, X, t, y):
            """Residual function for the time-varying voltage model.

            Args:
                params: [a1, b1, a2, b2, ..., a5, b5] — 10 parameters.
                X: Feature matrix shape (n_samples, 5) with constant column included.
            """
            a = params[0::2]  # slopes [a1, ..., a5]
            b = params[1::2]  # intercepts [b1, ..., b5]

            # Time-varying coefficients: c_i(t) = a_i * t + b_i
            c = [a[i] * t + b[i] for i in range(5)]

            # Model prediction (scaled space)
            # Note: X[:, 4] is the constant column (all 1.0)
            y_pred = (c[0] * X[:, 0] +   # c1 * I_scaled
                      c[1] * X[:, 1] +   # c2 * IT_scaled
                      c[2] * X[:, 2] +   # c3 * log_h_scaled
                      c[3] * X[:, 3] +   # c4 * I^2_scaled
                      c[4] * X[:, 4])    # c5 * 1.0 (constant column)

            residuals = y_pred - y
            penalty = self._compute_physics_penalty(a, b, t)
            return residuals + penalty

        # --- Step 4: Global optimization ---
        # Initial guess: a_i=0 (neutral degradation), b_i from physical priors
        p0 = [
            0, 0.5,    # a1, b1 (ohmic term, dominant)
            0, -0.1,   # a2, b2 (temperature effect, negative)
            0, 0.1,    # a3, b3 (concentration term)
            0, 0.05,   # a4, b4 (quadratic current term)
            0, 0.2     # a5, b5 (constant offset)
        ]
        
        try:
            result = least_squares(
                surface_residuals, 
                p0, 
                args=(X_train, t_train, y_train),
                method='lm',  # Levenberg-Marquardt
                verbose=0
            )
            self.coeffs = result.x

            # Compute parameter standard errors
            self._compute_coeffs_se(result, len(y_train))

            print(f"   Optimization converged (cost: {result.cost:.4e})")
            print(f"   Message: {result.message}")
            print(f"   status: {result.status}")
        except Exception as e:
            print(f"   Optimization failed: {e}")
            self.coeffs = np.zeros(10)
            self.coeffs_se = np.full(10, np.nan)

        # --- Step 5: Store test set raw data ---
        if len(df_test) > 0:
            X_test, y_test = self._prepare_matrices(df_test)
            t_test = df_test['calh'].values
            y_test_pred = self._predict_scaled(X_test, t_test)
            
            self.test_raw_data = {
                'time': df_test.index,
                'calh': t_test,
                'actual_v': self.scaler_U.unscale(y_test) + OCV(df_test.temperature.values),
                'pred_v': self.scaler_U.unscale(y_test_pred) + OCV(df_test.temperature.values),
                'actual_scaled': y_test,
                'pred_scaled': y_test_pred
            }
        
        # --- Step 6: Compute global XTX and sigma2 for SE calculation ---
        self._compute_global_xtx_sigma2(X_train)
        
        # --- Step 7: Generate Baseline-compatible interval results ---
        return self._build_interval_results()

    def _compute_physics_penalty(self, a: np.ndarray, b: np.ndarray, t: np.ndarray) -> float:
        """Compute soft physics constraint penalty.

        Constraints (based on electrochemical physics):
        - c1(t) > 0: ohmic resistance is always positive
        - c2(t) < 0: temperature rise reduces overpotential
        - c3(t) > 0: concentration/log-hour correction is positive
        """
        penalty = 0.0
        weight = 1e5
        t_check = [0, t.max()]
        
        for tc in t_check:
            c = [a[i] * tc + b[i] for i in range(3)]  # check first 3 coefficients
            # c[0] > 0 (ohmic resistance)
            if c[0] <= 0: penalty += weight * (c[0]**2)
            # c[1] < 0 (temperature effect)
            if c[1] >= 0: penalty += weight * (c[1]**2)
            # c[2] > 0 (concentration/log term)
            if c[2] <= 0: penalty += weight * (c[2]**2)
                
        return penalty

    def _compute_coeffs_se(self, result, n_samples: int):
        """Compute parameter standard errors from the Jacobian matrix."""
        mse = np.mean(result.fun ** 2)
        n_params = len(result.x)

        try:
            # Covariance = (J^T * J)^-1 * MSE
            JTJ = result.jac.T @ result.jac
            cov = np.linalg.inv(JTJ)

            # Adjust for degrees of freedom
            dof = max(n_samples - n_params, 1)
            self.coeffs_se = np.sqrt(np.diag(cov) * mse * n_samples / dof)

        except np.linalg.LinAlgError:
            print("   Warning: Singularity detected, SE might be unreliable")
            self.coeffs_se = np.full(10, np.nan)

    def _predict_scaled(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Predict scaled voltage using fitted parameters.
        
        X should have shape (n_samples, 5) with constant column included.
        """
        a = self.coeffs[0::2]
        b = self.coeffs[1::2]
        
        c = [a[i] * t + b[i] for i in range(5)]
        
        return (c[0] * X[:, 0] + 
                c[1] * X[:, 1] + 
                c[2] * X[:, 2] + 
                c[3] * X[:, 3] + 
                c[4] * X[:, 4])

    def _compute_global_xtx_sigma2(self, X_train: np.ndarray):
        """Compute global XTX matrix and sigma2 for SE calculation (Urc1-style).
        
        Args:
            X_train: Training feature matrix (n_samples, 5).
        """
        # Compute XTX = X^T * X
        self.global_xtx = X_train.T @ X_train
        
        # Compute sigma2 from test residuals (prediction variance)
        if self.test_raw_data and len(self.test_raw_data.get('pred_scaled', [])) > 0:
            actual_scaled = np.asarray(self.test_raw_data['actual_scaled'])
            pred_scaled = np.asarray(self.test_raw_data['pred_scaled'])
            residuals = actual_scaled - pred_scaled
            self.global_sigma2 = np.mean(residuals ** 2)
        else:
            self.global_sigma2 = np.nan

    def _build_interval_results(self) -> pd.DataFrame:
        """Generate Baseline-compatible interval-format results.

        Enables all ModelComparison methods to work with Surface results.
        """
        total_days = (self.data.index[-1] - self.installation_time).days
        num_intervals = int(total_days // self.slide) + 1

        dates = [self.installation_time + datetime.timedelta(days=i * self.slide)
                 for i in range(num_intervals)]
        index = pd.MultiIndex.from_product([dates, [self.name]], names=['date', 'name'])
        results = pd.DataFrame(index=index)

        # Initialize columns
        num_cols = ["day_since_install", "calh", "cond", "R2", "sigma2",
                    "c1", "c2", "c3", "c4", "c5",
                    "c1_se", "c2_se", "c3_se", "c4_se", "c5_se"]
        for col in num_cols:
            results[col] = np.nan

        obj_cols = ["quality", "status", "XTX"]
        for col in obj_cols:
            results[col] = pd.Series([None] * len(results), index=index, dtype=object)

        # Extract parameters
        a = self.coeffs[0::2]
        b = self.coeffs[1::2]
        a_se = self.coeffs_se[0::2] if self.coeffs_se is not None else np.zeros(5)
        b_se = self.coeffs_se[1::2] if self.coeffs_se is not None else np.zeros(5)

        # Fill each interval
        for date in dates:
            idx = (date, self.name)
            t_h = (date - self.installation_time).total_seconds() / 3600

            results.loc[idx, "day_since_install"] = (date - self.installation_time).days
            results.loc[idx, "calh"] = t_h
            results.loc[idx, "cond"] = 1.0  # global fit has high numerical stability
            results.loc[idx, "quality"] = "good"
            results.loc[idx, "status"] = "Surface_Fitted"
            results.loc[idx, "R2"] = self.test_metrics.get('test_r2', np.nan)
            results.loc[idx, "sigma2"] = self.global_sigma2 if self.global_sigma2 is not None else np.nan
            
            # Store XTX as serialized string (replicate Urc1 baseline pattern)
            if self.global_xtx is not None:
                xtx_str = str(self.global_xtx.flatten().tolist())
                results.at[idx, "XTX"] = xtx_str

            # Time-varying coefficients: c_i(t) = a_i * t + b_i
            for j in range(5):
                c_j = a[j] * t_h + b[j]
                results.loc[idx, f"c{j+1}"] = c_j

                # Error propagation: SE_ci = sqrt((t * SE_a)^2 + SE_b^2)
                se_j = np.sqrt((t_h * a_se[j])**2 + b_se[j]**2)
                results.loc[idx, f"c{j+1}_se"] = se_j

        return results

    def Urc_calc(self):
        """[Override] Compute Urc for each reference current.
        
        Uses XTX/sigma2-based SE calculation aligned with Urc1 method.
        """
        results = self.fitting_results.copy()

        for ref_name, ref_cfg in self.ref_config.items():
            iref = float(ref_cfg["Iref"])
            tref = float(ref_cfg["Tref"])
            ohref = float(ref_cfg["OHref"])
            col_urc = f"Urc_{iref}"
            col_se = f"Urc_se_{iref}"
            col_urc_named = f"Urc_{ref_name}"
            col_se_named = f"Urc_se_{ref_name}"

            results[col_urc] = np.nan
            results[col_se] = np.nan
            results[col_urc_named] = np.nan
            results[col_se_named] = np.nan

            # Compute Urc for all intervals
            for idx in results.index:
                row = results.loc[idx]

                # Time-varying coefficients
                c1 = row.get('c1', 0)
                c2 = row.get('c2', 0)
                c3 = row.get('c3', 0)
                c4 = row.get('c4', 0)
                c5 = row.get('c5', 0)

                # Scaled reference features
                i_s = self.scaler_I.scale(iref)
                ixt_s = self.scaler_IxT.scale(iref * tref)
                logh_s = self.scaler_log_h.scale(np.log(ohref))
                i2_s = self.scaler_I2.scale(iref ** 2)

                # Compute Urc
                u_scaled = c1 * i_s + c2 * ixt_s + c3 * logh_s + c4 * i2_s + c5
                urc = self.scaler_U.unscale(u_scaled) + OCV(tref)

                results.loc[idx, col_urc] = urc
                results.loc[idx, col_urc_named] = urc
            
            # Compute SE using XTX/sigma2 method
            se_series = self._calculate_reference_se_series(iref)
            results[col_se] = se_series
            results[col_se_named] = se_series

        return results

    def _calculate_reference_se_series(self, i):
        """Compute Urc SE for each interval using XTX/sigma2 variance propagation.
        
        Replaces constant test_rmse proxy with proper variance propagation method
        aligned with Urc1.py _calculate_reference_se_series.
        
        Args:
            i: Reference current density.
            
        Returns:
            pd.Series: Urc_se values indexed by interval.
        """
        series = pd.Series(np.nan, index=self.fitting_results.index, dtype=float)
        
        # Check if XTX and sigma2 are available
        if "XTX" not in self.fitting_results.columns or "sigma2" not in self.fitting_results.columns:
            return series
        
        if self.global_xtx is None or self.global_sigma2 is None:
            return series
        
        # Get reference vector for this current
        try:
            x0 = self.get_reference_vector(i)
        except:
            return series
        
        # Compute XTX inverse
        try:
            xtx_inv = np.linalg.inv(self.global_xtx)
        except np.linalg.LinAlgError:
            return series
        
        # Compute quadratic form: x0^T * (XTX)^-1 * x0
        quadratic_form = np.dot(x0, np.dot(xtx_inv, x0))
        
        # Compute variance: Var(Urc) = sigma2 * (1 + quadratic_form)
        variance = self.global_sigma2 * (1.0 + quadratic_form)
        
        # Convert to physical units
        scale = self.scaler_U.max - self.scaler_U.min
        se = np.sqrt(variance) * scale if variance > 0 else np.nan
        
        # Fill all valid rows with the same SE (global model)
        valid_rows = self.fitting_results["XTX"].notna() & self.fitting_results["sigma2"].notna()
        series.loc[valid_rows] = se
        
        return series

    def _calc_urc_se(self, row, iref: float) -> float:
        """[Deprecated] Use _calculate_reference_se_series instead.
        
        Kept for backward compatibility.
        """
        test_rmse_v = self.test_metrics.get('test_rmse_mv', 5.0) / 1000.0
        return test_rmse_v

    def _calculate_test_metrics(self):
        """Compute test set evaluation metrics (RMSE, R2, MAE, max error)."""
        if not self.test_raw_data:
            return
        
        actual = self.test_raw_data.get('actual_v')
        pred = self.test_raw_data.get('pred_v')
        
        if actual is None or pred is None:
            return
        
        # Point-by-point RMSE (primary metric)
        rmse = np.sqrt(mean_squared_error(actual, pred)) * 1000  # mV
        
        # R² Score
        r2 = r2_score(actual, pred)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - pred)) * 1000  # mV
        
        # Max Error
        max_err = np.max(np.abs(actual - pred)) * 1000  # mV
        
        self.test_metrics = {
            'test_rmse_mv': rmse,
            'test_r2': r2,
            'test_mae_mv': mae,
            'test_max_error_mv': max_err,
            'n_test_points': len(actual)
        }

    def _calculate_degradation_rates(self):
        """Compute degradation rates directly from global fit parameters.

        Core formula:
            Urc(t) = c1(t)*I_s + c2(t)*IT_s + c3(t)*log_h_s + c4(t)*I2_s + c5(t)
            where ci(t) = ai*t + bi

        Derivative w.r.t. time:
            dUrc/dt = a1*I_s + a2*IT_s + a3*log_h_s + a4*I2_s + a5

        This is the degradation rate (V/h), which needs inverse-scaling.
        """
        from degradation_toolbox.Urc.helpers import degradation_result
        
        if self.coeffs is None:
            return
        
        # Extract slopes (a) and intercepts (b)
        a = self.coeffs[0::2]  # [a1, a2, a3, a4, a5]
        b = self.coeffs[1::2]  # [b1, b2, b3, b4, b5]

        # Standard errors
        a_se = self.coeffs_se[0::2] if self.coeffs_se is not None else np.zeros(5)
        b_se = self.coeffs_se[1::2] if self.coeffs_se is not None else np.zeros(5)
        
        for ref_name, ref_cfg in self.ref_config.items():
            iref = float(ref_cfg["Iref"])
            tref = float(ref_cfg["Tref"])
            ohref = float(ref_cfg["OHref"])
            try:
                # Compute scaled features (constant, independent of t)
                i_s = self.scaler_I.scale(iref)
                ixt_s = self.scaler_IxT.scale(iref * tref)
                logh_s = self.scaler_log_h.scale(np.log(ohref))
                i2_s = self.scaler_I2.scale(iref ** 2)
                
                # Degradation rate (scaled space): dU_scaled/dt
                slope_scaled = (a[0] * i_s +
                               a[1] * ixt_s +
                               a[2] * logh_s +
                               a[3] * i2_s +
                               a[4])

                # Inverse-scale to original voltage
                slope_v_per_h = slope_scaled * (self.max_U - self.min_U)
                slope_uv_per_h = slope_v_per_h * 1e6  # convert to uV/h

                # BOL voltage (Urc at t=0)
                u_scaled_t0 = (b[0] * i_s +
                              b[1] * ixt_s +
                              b[2] * logh_s +
                              b[3] * i2_s +
                              b[4])
                bol_v = self.scaler_U.unscale(u_scaled_t0) + OCV(tref)

                # Error propagation for slope uncertainty
                var_slope_scaled = ((i_s * a_se[0]) ** 2 +
                                   (ixt_s * a_se[1]) ** 2 +
                                   (logh_s * a_se[2]) ** 2 +
                                   (i2_s * a_se[3]) ** 2 +
                                   a_se[4] ** 2)
                var_slope_v = var_slope_scaled * ((self.max_U - self.min_U) ** 2)
                uncertainty_uV2 = var_slope_v * 1e12  # convert to (uV/h)^2
                
                result = degradation_result(
                    aging_uV_per_h=slope_uv_per_h,
                    bol_V=bol_v,
                    uncertainty_uV2=uncertainty_uV2
                )
                self.degradation_results[iref] = result
                self.degradation_results_by_ref[ref_name] = result
                
            except Exception as e:
                print(f"   Failed to calculate degradation for {iref} A/cm2: {e}")

    def get_Urc_results(self, i: float) -> pd.DataFrame:
        """Get Urc results for a specific reference current.

        Compatible with stability_evaluation.py interface.

        Args:
            i: Target reference current density [A/cm2].

        Returns:
            DataFrame with columns: [timestamp, calh, Urc, Urc_se, method].
        """
        resolved = self._resolve_reference_input(i)
        if resolved is None:
            print(f'\033[91mNo Urc result available at {i} A/cm².\033[0m')
            return pd.DataFrame()
        closest_i, _, _, ref_name = resolved
        
        try:
            df = self.fitting_results_reliable.reset_index()
            
            if df.empty:
                return pd.DataFrame()
            
            result = pd.DataFrame({
                'timestamp': df['date'],
                'calh': df['calh'],
                'Urc': df[f'Urc_{closest_i}'],
                'Urc_se': df[f'Urc_se_{closest_i}'],
                'method': 'Surface',
                'reference_name': ref_name,
                'reference_i': closest_i,
            })
            
            return result
            
        except Exception as e:
            print(f"Error getting Urc results: {e}")
            return pd.DataFrame()

    def get_test_rmse_mv(self) -> float:
        """Return point-by-point prediction RMSE on test set [mV]."""
        return self.test_metrics.get('test_rmse_mv', np.nan)

    def analyze_full_dataset_residuals(self, return_data: bool = False) -> dict:
        """Perform residual analysis on the entire preprocessed dataset.

        For each data point: compute the Surface model prediction using actual
        (t, I, T, OH), compare to actual U, and gather residual statistics.

        Args:
            return_data: If True, include full residual data in results.

        Returns:
            Dict of residual statistics.
        """
        from scipy import stats
        
        if self.coeffs is None:
            print("Model not fitted yet.")
            return {}

        # --- Step 1: Prepare full dataset ---
        df = self.data.copy()
        n_total = len(df)

        # Ensure calh exists
        if 'calh' not in df.columns:
            df['calh'] = (df.index - self.installation_time) / np.timedelta64(1, 'h')
        
        # --- Step 2: Compute predicted voltage for each point ---
        a = self.coeffs[0::2]
        b = self.coeffs[1::2]
        
        # Vectorized computation
        I = df['currentDensity'].values
        T = df['temperature'].values
        OH = df['h_since_last_start'].values
        t = df['calh'].values
        U_actual = df['voltage'].values
        
        # Scaled features
        I_s = self.scaler_I.scale(I)
        IxT_s = self.scaler_IxT.scale(I * T)
        log_h_s = self.scaler_log_h.scale(np.log(OH))
        I2_s = self.scaler_I2.scale(I ** 2)
        
        # Time-varying coefficients c_i(t) = a_i * t + b_i
        c1 = a[0] * t + b[0]
        c2 = a[1] * t + b[1]
        c3 = a[2] * t + b[2]
        c4 = a[3] * t + b[3]
        c5 = a[4] * t + b[4]
        
        # Predicted scaled voltage
        U_pred_scaled = c1 * I_s + c2 * IxT_s + c3 * log_h_s + c4 * I2_s + c5

        # Inverse-scale to actual voltage
        U_pred = self.scaler_U.unscale(U_pred_scaled) + OCV(T)

        # --- Step 3: Compute residuals ---
        residuals = U_actual - U_pred  # V
        residuals_mv = residuals * 1000  # mV

        # --- Step 4: Statistical analysis ---
        # Basic statistics
        mean_res = np.mean(residuals_mv)
        std_res = np.std(residuals_mv, ddof=1)
        median_res = np.median(residuals_mv)
        min_res = np.min(residuals_mv)
        max_res = np.max(residuals_mv)
        rmse = np.sqrt(np.mean(residuals_mv ** 2))
        mae = np.mean(np.abs(residuals_mv))
        
        # Normality test (Shapiro-Wilk, sample size limited to 5000)
        sample_for_test = residuals_mv[:5000] if len(residuals_mv) > 5000 else residuals_mv
        try:
            shapiro_stat, shapiro_p = stats.shapiro(sample_for_test)
        except:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # Jarque-Bera test (skewness and kurtosis)
        try:
            jb_stat, jb_p = stats.jarque_bera(residuals_mv)
        except:
            jb_stat, jb_p = np.nan, np.nan
        
        # Skewness and kurtosis
        skewness = stats.skew(residuals_mv)
        kurtosis = stats.kurtosis(residuals_mv)

        # Outlier statistics (beyond 2-sigma and 3-sigma)
        outliers_2sigma = np.sum(np.abs(residuals_mv - mean_res) > 2 * std_res)
        outliers_3sigma = np.sum(np.abs(residuals_mv - mean_res) > 3 * std_res)
        
        # Voltage range statistics (for relative error calculation)
        U_min = np.min(U_actual) * 1000  # mV
        U_max = np.max(U_actual) * 1000  # mV
        U_mean = np.mean(U_actual) * 1000  # mV
        
        # Relative error percentage (relative to mean voltage)
        relative_rmse_pct = (rmse / U_mean) * 100
        relative_mae_pct = (mae / U_mean) * 100

        # Store residual data for visualization
        self._full_residuals = {
            'time': df.index,
            'calh': t,
            'I': I,
            'T': T,
            'OH': OH,
            'U_actual': U_actual,
            'U_pred': U_pred,
            'residuals_mv': residuals_mv
        }
        
        # Results dictionary
        results = {
            'n_points': n_total,
            'mean_mv': round(mean_res, 3),
            'std_mv': round(std_res, 3),
            'median_mv': round(median_res, 3),
            'min_mv': round(min_res, 3),
            'max_mv': round(max_res, 3),
            'rmse_mv': round(rmse, 3),
            'mae_mv': round(mae, 3),
            # Voltage range and relative error
            'U_min_mv': round(U_min, 1),
            'U_max_mv': round(U_max, 1),
            'U_mean_mv': round(U_mean, 1),
            'relative_rmse_pct': round(relative_rmse_pct, 4),
            'relative_mae_pct': round(relative_mae_pct, 4),
            'skewness': round(skewness, 3),
            'kurtosis': round(kurtosis, 3),
            'shapiro_stat': round(shapiro_stat, 4) if not np.isnan(shapiro_stat) else np.nan,
            'shapiro_p': shapiro_p,
            'jarque_bera_stat': round(jb_stat, 2) if not np.isnan(jb_stat) else np.nan,
            'jarque_bera_p': jb_p,
            'outliers_2sigma': outliers_2sigma,
            'outliers_2sigma_pct': round(100 * outliers_2sigma / n_total, 2),
            'outliers_3sigma': outliers_3sigma,
            'outliers_3sigma_pct': round(100 * outliers_3sigma / n_total, 2),
            'is_normal_shapiro': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None,
            'is_normal_jb': jb_p > 0.05 if not np.isnan(jb_p) else None
        }
        
        if return_data:
            results['residuals_data'] = self._full_residuals
        
        return results

    def plot_residual_diagnostics(self, save: bool = False):
        """Plot full-dataset residual diagnostics (4 independent plots).

        Includes:
            1. Residual histogram + normal fit curve
            2. Q-Q plot
            3. Residuals vs time
            4. Residuals vs current density
        """
        import plotly.graph_objects as go
        from scipy import stats
        
        # Run residual analysis first
        if not hasattr(self, '_full_residuals') or self._full_residuals is None:
            self.analyze_full_dataset_residuals()
        
        res_data = self._full_residuals
        residuals = res_data['residuals_mv']
        
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        n = len(residuals)
        
        # ===== Plot 1: Residual Histogram + Normal Fit =====
        fig1 = go.Figure()
        
        # Histogram
        fig1.add_trace(go.Histogram(
            x=residuals,
            nbinsx=80,
            name='Residuals',
            opacity=0.7,
            marker_color='steelblue',
            histnorm='probability density'
        ))
        
        # Normal fit curve
        x_norm = np.linspace(min(residuals), max(residuals), 200)
        y_norm = stats.norm.pdf(x_norm, mean_res, std_res)
        fig1.add_trace(go.Scatter(
            x=x_norm, y=y_norm,
            mode='lines',
            name='Normal Fit',
            line=dict(color='red', width=2)
        ))
        
        fig1.update_layout(
            title=f'① Residual Histogram + Normal Fit (n={n:,})',
            xaxis_title='Residual [mV]',
            yaxis_title='Density',
            template='plotly_white',
            height=450,
            width=700,
            annotations=[dict(
                x=0.98, y=0.95,
                xref='paper', yref='paper',
                text=f"μ={mean_res:.2f} mV<br>σ={std_res:.2f} mV",
                showarrow=False,
                font=dict(size=12),
                align='right',
                bgcolor='rgba(255,255,255,0.8)'
            )]
        )
        fig1.show()
        
        # --- Plot 2: Q-Q Plot ---
        fig2 = go.Figure()
        
        sorted_res = np.sort(residuals)
        theoretical_q = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
        
        # Sample to avoid too many points
        sample_step = max(1, n // 2000)
        
        fig2.add_trace(go.Scatter(
            x=theoretical_q[::sample_step],
            y=sorted_res[::sample_step],
            mode='markers',
            name='Q-Q Points',
            marker=dict(size=3, color='steelblue', opacity=0.5)
        ))
        
        # Reference line
        qq_line_x = np.array([theoretical_q.min(), theoretical_q.max()])
        qq_line_y = mean_res + std_res * qq_line_x
        fig2.add_trace(go.Scatter(
            x=qq_line_x, y=qq_line_y,
            mode='lines',
            name='Reference Line (if Normal)',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig2.update_layout(
            title='② Q-Q Plot (Normality Check)',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles [mV]',
            template='plotly_white',
            height=450,
            width=700,
            showlegend=True
        )
        fig2.show()
        
        # --- Plot 3: Residuals vs Time ---
        fig3 = go.Figure()
        
        calh = res_data['calh']
        sample_step_time = max(1, len(calh) // 3000)
        
        fig3.add_trace(go.Scatter(
            x=calh[::sample_step_time],
            y=residuals[::sample_step_time],
            mode='markers',
            name='Residuals',
            marker=dict(size=2, color='steelblue', opacity=0.3)
        ))
        
        # Reference lines
        fig3.add_hline(y=0, line_dash="dash", line_color="red", 
                       annotation_text="Mean=0", annotation_position="right")
        fig3.add_hline(y=2*std_res, line_dash="dot", line_color="orange",
                       annotation_text=f"+2σ ({2*std_res:.1f}mV)", annotation_position="right")
        fig3.add_hline(y=-2*std_res, line_dash="dot", line_color="orange",
                       annotation_text=f"-2σ ({-2*std_res:.1f}mV)", annotation_position="right")
        
        fig3.update_layout(
            title='③ Residuals vs Time (Check for Drift)',
            xaxis_title='Calendar Hours [h]',
            yaxis_title='Residual [mV]',
            template='plotly_white',
            height=450,
            width=900,
            showlegend=False
        )
        fig3.show()
        
        # --- Plot 4: Residuals vs Current Density ---
        fig4 = go.Figure()
        
        I = res_data['I']
        
        fig4.add_trace(go.Scatter(
            x=I[::sample_step_time],
            y=residuals[::sample_step_time],
            mode='markers',
            name='Residuals',
            marker=dict(size=2, color='steelblue', opacity=0.3)
        ))
        
        fig4.add_hline(y=0, line_dash="dash", line_color="red")
        fig4.add_hline(y=2*std_res, line_dash="dot", line_color="orange")
        fig4.add_hline(y=-2*std_res, line_dash="dot", line_color="orange")
        
        fig4.update_layout(
            title='④ Residuals vs Current Density (Check for Heteroscedasticity)',
            xaxis_title='Current Density [A/cm²]',
            yaxis_title='Residual [mV]',
            template='plotly_white',
            height=450,
            width=700,
            showlegend=False
        )
        fig4.show()
        
        if save:
            fig1.write_html(f"plots/{self.name}_Surface_Residual_Histogram.html")
            fig2.write_html(f"plots/{self.name}_Surface_Residual_QQ.html")
            fig3.write_html(f"plots/{self.name}_Surface_Residual_vs_Time.html")
            fig4.write_html(f"plots/{self.name}_Surface_Residual_vs_Current.html")
            print(f"4 diagnostic plots saved to plots/ folder.")

    def print_residual_summary(self):
        """Print residual analysis summary table."""
        results = self.analyze_full_dataset_residuals()
        
        if not results:
            return
        
        print("\n" + "=" * 60)
        print("Surface Model: Full Dataset Residual Analysis")
        print("=" * 60)

        print(f"\nData Voltage Range:")
        print(f"   Min Voltage:       {results['U_min_mv']:>8.1f} mV ({results['U_min_mv']/1000:.3f} V)")
        print(f"   Max Voltage:       {results['U_max_mv']:>8.1f} mV ({results['U_max_mv']/1000:.3f} V)")
        print(f"   Mean Voltage:      {results['U_mean_mv']:>8.1f} mV ({results['U_mean_mv']/1000:.3f} V)")
        
        print(f"\nBasic Statistics ({results['n_points']:,} data points):")
        print(f"   Mean Residual:     {results['mean_mv']:>8.3f} mV")
        print(f"   Std Deviation:     {results['std_mv']:>8.3f} mV")
        print(f"   Median Residual:   {results['median_mv']:>8.3f} mV")
        print(f"   Min Residual:      {results['min_mv']:>8.3f} mV")
        print(f"   Max Residual:      {results['max_mv']:>8.3f} mV")
        print(f"   RMSE:              {results['rmse_mv']:>8.3f} mV  → Relative: {results['relative_rmse_pct']:.4f}%")
        print(f"   MAE:               {results['mae_mv']:>8.3f} mV  → Relative: {results['relative_mae_pct']:.4f}%")
        
        print(f"\nDistribution Shape:")
        print(f"   Skewness:          {results['skewness']:>8.3f}  (0 = symmetric)")
        print(f"   Kurtosis:          {results['kurtosis']:>8.3f}  (0 = normal)")
        
        # print(f"\nNormality Tests:")
        # if results['shapiro_p'] is not None:
        #     normal_shapiro = "Yes" if results['is_normal_shapiro'] else "No"
        #     print(f"   Shapiro-Wilk:      W={results['shapiro_stat']:.4f}, p={results['shapiro_p']:.2e} → {normal_shapiro}")
        # if results['jarque_bera_p'] is not None:
        #     normal_jb = "Yes" if results['is_normal_jb'] else "No"
        #     print(f"   Jarque-Bera:       JB={results['jarque_bera_stat']:.2f}, p={results['jarque_bera_p']:.2e} → {normal_jb}")
        
        print(f"\nOutlier Analysis:")
        print(f"   Beyond ±2σ:        {results['outliers_2sigma']:>6} ({results['outliers_2sigma_pct']:.2f}%)")
        print(f"   Beyond ±3σ:        {results['outliers_3sigma']:>6} ({results['outliers_3sigma_pct']:.2f}%)")
        print(f"   (Expected for normal: 2σ ≈ 4.55%, 3σ ≈ 0.27%)")
        
        print("\n" + "=" * 60)

    def get_coeffs_summary(self) -> pd.DataFrame:
        """Get summary table of 10 fitted parameters."""
        if self.coeffs is None:
            return pd.DataFrame()
        
        a = self.coeffs[0::2]
        b = self.coeffs[1::2]
        a_se = self.coeffs_se[0::2] if self.coeffs_se is not None else np.zeros(5)
        b_se = self.coeffs_se[1::2] if self.coeffs_se is not None else np.zeros(5)
        
        data = {
            'Coefficient': [f'c{i+1}' for i in range(5)],
            'a (Slope)': a,
            'a_SE': a_se,
            'b (Intercept)': b,
            'b_SE': b_se,
            'Physical Meaning': [
                'Ohmic resistance (I term)',
                'Temperature effect (I×T term)',
                'Concentration effect (ln(OH) term)',
                'Quadratic current (I² term)',
                'Constant offset'
            ]
        }
        
        return pd.DataFrame(data)

    def back_calculate_coefficients(self):
        """[Override] Coefficients already computed in _build_interval_results."""
        return self.fitting_results

    def check_vertex(self, row):
        """[Override] Surface method uses global constraints; skip per-row check."""
        return row.get('quality', 'good')

    def plot_test_prediction(self, save: bool = False):
        """Plot test set prediction comparison (3 plots + evaluation summary).

        Includes:
            1. Time series plot (Actual vs Predicted)
            2. Scatter plot (Actual vs Predicted, with 45-deg reference line)
            3. Residual time series
            4. Printed evaluation metrics summary
        """
        import plotly.graph_objects as go
        from scipy import stats
        
        if not self.test_raw_data:
            print("No test data available.")
            return
        
        time = self.test_raw_data['time']
        calh = self.test_raw_data['calh']
        actual = np.array(self.test_raw_data['actual_v'])
        pred = np.array(self.test_raw_data['pred_v'])
        residual = (actual - pred) * 1000  # mV
        
        n = len(actual)
        
        # --- Compute detailed evaluation metrics ---
        # Basic error metrics
        rmse = np.sqrt(np.mean(residual ** 2))
        mae = np.mean(np.abs(residual))
        max_error = np.max(np.abs(residual))
        
        # R-squared and correlation coefficient
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        pearson_r, _ = stats.pearsonr(actual, pred)
        
        # Residual statistics
        mean_res = np.mean(residual)
        std_res = np.std(residual)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs(residual / (actual * 1000))) * 100  # %
        
        # 95% confidence interval coverage
        ci_95 = 1.96 * std_res
        within_ci = np.sum(np.abs(residual - mean_res) <= ci_95) / n * 100
        
        # --- Plot 1: Time series comparison (sampled) ---
        fig1 = go.Figure()

        sample_step = max(1, n // 2000)  # sample to avoid excessive points

        # Actual values
        fig1.add_trace(go.Scatter(
            x=time[::sample_step],
            y=actual[::sample_step],
            mode='lines',
            name='Actual',
            line=dict(color='#1f77b4', width=1.5),
            opacity=0.8
        ))
        
        # Predicted values
        fig1.add_trace(go.Scatter(
            x=time[::sample_step],
            y=pred[::sample_step],
            mode='lines',
            name='Predicted',
            line=dict(color='#d62728', width=1.5, dash='dash'),
            opacity=0.8
        ))
        
        fig1.update_layout(
            title=f'① Test Set: Voltage Time Series (n={n:,} points, sampled)',
            xaxis_title='Time',
            yaxis_title='Voltage [V]',
            template='plotly_white',
            height=400,
            width=900,
            legend=dict(x=0.02, y=0.98),
            annotations=[dict(
                x=0.98, y=0.02,
                xref='paper', yref='paper',
                text=f"RMSE: {rmse:.2f} mV | R²: {r2:.4f}",
                showarrow=False,
                font=dict(size=12),
                bgcolor='rgba(255,255,255,0.8)'
            )]
        )
        fig1.show()
        
        # --- Plot 2: Actual vs Predicted scatter ---
        fig2 = go.Figure()

        # Scatter points
        fig2.add_trace(go.Scatter(
            x=actual[::sample_step],
            y=pred[::sample_step],
            mode='markers',
            name='Test Points',
            marker=dict(
                size=4,
                color=calh[::sample_step],
                colorscale='Viridis',
                colorbar=dict(title='Cal Hours [h]'),
                opacity=0.5
            ),
            hovertemplate='Actual: %{x:.4f} V<br>Predicted: %{y:.4f} V<extra></extra>'
        ))
        
        # 45-degree perfect prediction reference line
        v_min, v_max = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        margin = (v_max - v_min) * 0.05
        fig2.add_trace(go.Scatter(
            x=[v_min - margin, v_max + margin],
            y=[v_min - margin, v_max + margin],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig2.update_layout(
            title='② Actual vs Predicted (Color = Time)',
            xaxis_title='Actual Voltage [V]',
            yaxis_title='Predicted Voltage [V]',
            template='plotly_white',
            height=500,
            width=550,
            xaxis=dict(scaleanchor='y', scaleratio=1),
            annotations=[dict(
                x=0.05, y=0.95,
                xref='paper', yref='paper',
                text=f"Pearson r: {pearson_r:.4f}<br>R²: {r2:.4f}",
                showarrow=False,
                font=dict(size=11),
                bgcolor='rgba(255,255,255,0.8)',
                align='left'
            )]
        )
        fig2.show()
        
        # --- Plot 3: Residual time series ---
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=calh[::sample_step],
            y=residual[::sample_step],
            mode='markers',
            name='Residuals',
            marker=dict(size=3, color='steelblue', opacity=0.4)
        ))
        
        # Reference lines
        fig3.add_hline(y=0, line_dash="solid", line_color="red", line_width=2,
                       annotation_text="Zero", annotation_position="right")
        fig3.add_hline(y=mean_res, line_dash="dash", line_color="green",
                       annotation_text=f"Mean: {mean_res:.2f} mV", annotation_position="right")
        fig3.add_hline(y=mean_res + 2*std_res, line_dash="dot", line_color="orange",
                       annotation_text=f"+2σ", annotation_position="right")
        fig3.add_hline(y=mean_res - 2*std_res, line_dash="dot", line_color="orange",
                       annotation_text=f"-2σ", annotation_position="right")
        
        fig3.update_layout(
            title='③ Residuals vs Calendar Hours',
            xaxis_title='Calendar Hours [h]',
            yaxis_title='Residual [mV]',
            template='plotly_white',
            height=400,
            width=900,
            showlegend=False
        )
        fig3.show()
        
        # --- Print evaluation summary ---
        print("\n" + "=" * 70)
        print("Surface Model: Test Set Prediction Evaluation")
        print("=" * 70)
        print(f"\nDataset Info:")
        print(f"   Test Points:       {n:,}")
        print(f"   Time Range:        {calh[0]:.0f} h → {calh[-1]:.0f} h")
        
        print(f"\nError Metrics:")
        print(f"   RMSE:              {rmse:>8.3f} mV")
        print(f"   MAE:               {mae:>8.3f} mV")
        print(f"   Max Error:         {max_error:>8.3f} mV")
        print(f"   MAPE:              {mape:>8.3f} %")
        
        print(f"\nCorrelation Metrics:")
        print(f"   R² Score:          {r2:>8.4f}")
        print(f"   Pearson r:         {pearson_r:>8.4f}")
        
        print(f"\nResidual Distribution:")
        print(f"   Mean:              {mean_res:>8.3f} mV  (ideal: 0)")
        print(f"   Std Dev:           {std_res:>8.3f} mV")
        print(f"   Within ±2σ:        {100 - np.sum(np.abs(residual - mean_res) > 2*std_res) / n * 100:>7.2f} %  (expected: ~95%)")
        
        # print(f"\nQuality Assessment:")
        # if rmse < 5:
        #     print(f"   Excellent: RMSE < 5 mV")
        # elif rmse < 10:
        #     print(f"   Good: RMSE < 10 mV")
        # elif rmse < 20:
        #     print(f"   Acceptable: RMSE < 20 mV")
        # else:
        #     print(f"   Poor: RMSE >= 20 mV, model may need improvement")
        
        # if abs(mean_res) < 1:
        #     print(f"   No systematic bias (|mean| < 1 mV)")
        # else:
        #     print(f"   Possible systematic bias (mean = {mean_res:.2f} mV)")
        
        # print("=" * 70)
        
        if save:
            fig1.write_html(f"plots/{self.name}_Surface_Test_TimeSeries.html")
            fig2.write_html(f"plots/{self.name}_Surface_Test_Scatter.html")
            fig3.write_html(f"plots/{self.name}_Surface_Test_Residuals.html")
            print(f"3 plots saved to plots/ folder.")
        
        # Return evaluation dict for further use
        return {
            'n_test': n,
            'rmse_mv': rmse,
            'mae_mv': mae,
            'max_error_mv': max_error,
            'mape_pct': mape,
            'r2': r2,
            'pearson_r': pearson_r,
            'mean_residual_mv': mean_res,
            'std_residual_mv': std_res
        }

    def plot_3d_surface(self, fixed_T: float = None, fixed_OH: float = None, save: bool = False):
        """Plot 3D visualization of the fitted hypersurface.

        Since data is 4D (I, T, OH, t) -> U, we fix T and OH to plot
        the U(I, t) surface with overlaid data scatter.

        Args:
            fixed_T: Fixed temperature [deg C], defaults to Tref.
            fixed_OH: Fixed operating hours [h], defaults to OHref.
            save: Whether to save the plot.
        """
        import plotly.graph_objects as go
        
        if self.coeffs is None:
            print("No coefficients available.")
            return
        
        # Default parameters
        T = fixed_T if fixed_T else self.Tref
        OH = fixed_OH if fixed_OH else self.OHref

        # Create mesh grid
        I_range = np.linspace(0.3, 1.6, 30)
        t_max = (self.data.index[-1] - self.installation_time).total_seconds() / 3600
        t_range = np.linspace(0, t_max, 50)
        I_grid, t_grid = np.meshgrid(I_range, t_range)
        
        # Compute Surface
        a = self.coeffs[0::2]
        b = self.coeffs[1::2]
        
        U_surface = np.zeros_like(I_grid)
        for i in range(I_grid.shape[0]):
            for j in range(I_grid.shape[1]):
                I_val = I_grid[i, j]
                t_val = t_grid[i, j]
                
                # Time-varying coefficients
                c = [a[k] * t_val + b[k] for k in range(5)]

                # Scaled features
                i_s = self.scaler_I.scale(I_val)
                ixt_s = self.scaler_IxT.scale(I_val * T)
                logh_s = self.scaler_log_h.scale(np.log(OH))
                i2_s = self.scaler_I2.scale(I_val ** 2)
                
                # Predicted voltage
                u_scaled = c[0]*i_s + c[1]*ixt_s + c[2]*logh_s + c[3]*i2_s + c[4]
                U_surface[i, j] = self.scaler_U.unscale(u_scaled) + OCV(T)
        
        # Create figure
        fig = go.Figure()
        
        # Add surface
        fig.add_trace(go.Surface(
            x=I_grid,
            y=t_grid,
            z=U_surface,
            colorscale='Viridis',
            opacity=0.8,
            name='Fitted Surface',
            showscale=True,
            colorbar=dict(title='Voltage [V]', x=1.02)
        ))
        
        # Add data point cloud (sampled to avoid excessive points)
        sample_step = max(1, len(self.data) // 1000)
        df_sample = self.data.iloc[::sample_step].copy()
        
        # Filter points with T and OH close to fixed values
        T_tol = 3  # ±3°C
        OH_tol = 12  # ±12h
        mask = (
            (df_sample['temperature'] >= T - T_tol) & 
            (df_sample['temperature'] <= T + T_tol) &
            (df_sample['h_since_last_start'] >= OH - OH_tol) &
            (df_sample['h_since_last_start'] <= OH + OH_tol)
        )
        df_filtered = df_sample[mask]
        
        if len(df_filtered) > 0:
            calh_sample = (df_filtered.index - self.installation_time) / np.timedelta64(1, 'h')
            
            fig.add_trace(go.Scatter3d(
                x=df_filtered['currentDensity'],
                y=calh_sample,
                z=df_filtered['voltage'],
                mode='markers',
                marker=dict(
                    size=2,
                    color=df_filtered['voltage'],
                    colorscale='Plasma',
                    opacity=0.6
                ),
                name=f'Data (T≈{T}°C, OH≈{OH}h)'
            ))
        
        # Add Iref section lines (degradation curves)
        for iref in self.Iref:
            t_line = np.linspace(0, t_max, 100)
            u_line = []
            for t_val in t_line:
                c = [a[k] * t_val + b[k] for k in range(5)]
                i_s = self.scaler_I.scale(iref)
                ixt_s = self.scaler_IxT.scale(iref * T)
                logh_s = self.scaler_log_h.scale(np.log(OH))
                i2_s = self.scaler_I2.scale(iref ** 2)
                u_scaled = c[0]*i_s + c[1]*ixt_s + c[2]*logh_s + c[3]*i2_s + c[4]
                u_line.append(self.scaler_U.unscale(u_scaled) + OCV(T))
            
            fig.add_trace(go.Scatter3d(
                x=np.full_like(t_line, iref),
                y=t_line,
                z=u_line,
                mode='lines',
                line=dict(width=6, color='red'),
                name=f'Urc @ {iref} A/cm²'
            ))
        
        fig.update_layout(
            title=f'Surface Fitting: U(I, t) at T={T}°C, OH={OH}h',
            scene=dict(
                xaxis_title='Current Density [A/cm²]',
                yaxis_title='Calendar Hours [h]',
                zaxis_title='Voltage [V]',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            ),
            width=900,
            height=700,
            template='plotly_white'
        )
        
        fig.show()
        
        if save:
            fig.write_html(f"plots/{self.name}_Surface_3D.html")
    
    def evaluate_surface_quality(self) -> pd.DataFrame:
        """Surface-specific quality evaluation table.

        Returns:
            DataFrame with degradation rate, significance, and test metrics per Iref.
        """
        if self.coeffs is None:
            return pd.DataFrame()
        
        # Extract parameters
        a = self.coeffs[0::2]
        a_se = self.coeffs_se[0::2] if self.coeffs_se is not None else np.zeros(5)
        
        results = []
        for iref in self.Iref:
            if iref not in self.degradation_results:
                continue
            
            deg_res = self.degradation_results[iref]
            slope_uv_h = deg_res.aging_uV_per_h
            slope_se = np.sqrt(deg_res.uncertainty_uV2)
            t_stat = abs(slope_uv_h / slope_se) if slope_se > 0 else np.inf
            
            results.append({
                'Iref (A/cm²)': iref,
                'Degradation Rate (μV/h)': round(slope_uv_h, 3),
                'Slope SE (μV/h)': round(slope_se, 3),
                't-statistic': round(t_stat, 2),
                'Significant (|t|>2)': 'Yes' if t_stat > 2 else 'No',
                'BOL Voltage (V)': round(deg_res.bol_V, 4),
                'Test RMSE (mV)': round(self.test_metrics.get('test_rmse_mv', np.nan), 2),
                'Test R²': round(self.test_metrics.get('test_r2', np.nan), 4),
                'N Test Points': self.test_metrics.get('n_test_points', 0)
            })
        
        return pd.DataFrame(results)

    def plot_coefficients_evolution(self, save: bool = False):
        """Plot the evolution curves of 5 time-varying coefficients."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if self.coeffs is None:
            print("No coefficients available.")
            return
        
        # Time range
        t_max = (self.data.index[-1] - self.installation_time).total_seconds() / 3600
        t = np.linspace(0, t_max, 100)
        
        a = self.coeffs[0::2]
        b = self.coeffs[1::2]
        a_se = self.coeffs_se[0::2] if self.coeffs_se is not None else np.zeros(5)
        b_se = self.coeffs_se[1::2] if self.coeffs_se is not None else np.zeros(5)
        
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[f'c{i+1}(t) = {a[i]:.2e}·t + {b[i]:.3f}' for i in range(5)]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i in range(5):
            c_t = a[i] * t + b[i]
            se_t = np.sqrt((t * a_se[i])**2 + b_se[i]**2)
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([c_t + se_t, (c_t - se_t)[::-1]]),
                fill='toself',
                fillcolor=f'rgba{tuple(list(int(colors[i][j:j+2], 16) for j in (1, 3, 5)) + [0.2])}',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ), row=i+1, col=1)
            
            # Main curve
            fig.add_trace(go.Scatter(
                x=t, y=c_t,
                mode='lines',
                name=f'c{i+1}',
                line=dict(color=colors[i], width=2)
            ), row=i+1, col=1)
            
            fig.update_yaxes(title_text=f'c{i+1}', row=i+1, col=1)
        
        fig.update_xaxes(title_text='Calendar Hours [h]', row=5, col=1)
        fig.update_layout(
            title='Time-varying Coefficients Evolution',
            height=800,
            template='plotly_white',
            showlegend=False
        )
        
        fig.show()
        
        if save:
            fig.write_html(f"plots/{self.name}_Surface_Coefficients.html")

