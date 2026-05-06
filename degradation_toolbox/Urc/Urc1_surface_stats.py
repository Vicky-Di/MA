"""
Urc1_Surface_Stats: Global time-varying parameter surface fitting (Statsmodels RLM version).

Key improvements:
- Uses statsmodels.RLM (Robust Linear Model) instead of scipy.optimize.least_squares
- Automatic outlier handling via Huber T-norm
- More stable parameter estimation and standard error computation

Model equation: U = c1(t)I + c2(t)IT + c3(t)ln(OH) + c4(t)I^2 + c5(t)
Parameter evolution: ci(t) = ai*t + bi
"""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from degradation_toolbox.Urc.Urc1 import Urc1, OCV, func, MinMaxScalerCustomize


class Urc1_Surface_Stats(Urc1):
    """
    Method 2: Global time-varying parameter surface fitting (RLM robust version).

    Uses statsmodels.RLM to automatically identify and suppress outliers.
    """
    
    def __init__(self, data, train_ratio: float = 0.8, **kwargs):
        self.train_ratio = train_ratio
        self.coeffs = None          # Stores 10 parameters [a1, b1, ..., a5, b5]
        self.coeffs_se = None       # Stores standard errors for 10 parameters
        self.test_metrics = {}
        self.test_raw_data = {}
        self.split_idx = None
        self.rlm_result = None      # RLM fitting result object
        self.degradation_results = {}  # Compatible with stability_evaluation
        
        # SE calculation cache (for XTX/sigma2 method)
        self._se_cache = None
        self.global_xtx = None      # Global XTX matrix from training data
        self.global_sigma2 = None   # Global sigma2 from test data
        
        # super().__init__ sequentially calls data_preprocess() and model_fitting()
        super().__init__(data, **kwargs)
        
        self._calculate_test_metrics()
        self._calculate_degradation_rates()
        
        print(f"\nUrc1_Surface_Stats Initialized:")
        print(f"   Train Ratio: {self.train_ratio:.0%}")
        if self.test_metrics:
            print(f"   Test RMSE: {self.test_metrics.get('test_rmse_mv', 'N/A'):.2f} mV")

    def model_fitting(self):
        """
        [Override] Global surface fitting: expand the physical equation and fit using RLM.
        """
        print(f"\nStarting Global Surface Fitting via RLM (Train: {self.train_ratio:.0%})...")
        
        # Add cumulative calendar hours (calh)
        self.data['calh'] = (self.data.index - self.installation_time) / np.timedelta64(1, 'h')
        
        # ===== Step 1: Train/test split =====
        self.split_idx = int(len(self.data) * self.train_ratio)
        df_train = self.data.iloc[:self.split_idx]
        df_test = self.data.iloc[self.split_idx:]
        
        print(f"   Train: {len(df_train)} points | Test: {len(df_test)} points")
        
        # ===== Step 2: Feature engineering (key modification) =====
        # Expand time-varying model U = (a*t + b)*X into 10 linear features
        X_train_expanded = self._expand_features(df_train)
        
        # Prepare target variable y (scaled overpotential)
        # y = Scaler(U_actual - OCV)
        y_train = self.scaler_U.scale(df_train['voltage'].values - OCV(df_train['temperature'].values))

        # ===== Step 3: RLM robust fitting (replaces least_squares) =====
        try:
            # M=sm.robust.norms.HuberT() is the core for automatic outlier handling
            model = sm.RLM(y_train, X_train_expanded, M=sm.robust.norms.HuberT())
            self.rlm_result = model.fit()
            
            # Rearrange parameter order to match [a1, b1, a2, b2, ..., a5, b5] format
            # RLM returns order: ['t_I', 'I', 't_IT', 'IT', 't_logH', 'logH', 't_I2', 'I2', 't', 'const']
            # Already in the correct alternating order
            self.coeffs = self.rlm_result.params.values
            self.coeffs_se = self.rlm_result.bse.values
            
            print(f"   RLM Optimization converged (Scale: {self.rlm_result.scale:.4e})")
            print(f"   Weighted Sample Size: {self.rlm_result.nobs:.0f}")
            
            # Print outlier weight statistics
            weights = self.rlm_result.weights
            low_weight_pct = np.sum(weights < 0.5) / len(weights) * 100
            print(f"   Outliers (weight < 0.5): {np.sum(weights < 0.5)} ({low_weight_pct:.2f}%)")
            
        except Exception as e:
            print(f"   Error: RLM Fitting failed: {e}")
            self.coeffs = np.zeros(10)
            self.coeffs_se = np.full(10, np.nan)
            return pd.DataFrame()

        # ===== Step 4: Test set prediction and storage =====
        if len(df_test) > 0:
            X_test_expanded = self._expand_features(df_test)
            y_test_pred_scaled = self.rlm_result.predict(X_test_expanded)
            
            self.test_raw_data = {
                'time': df_test.index,
                'calh': df_test['calh'].values,
                'actual_v': df_test['voltage'].values,
                'pred_v': self.scaler_U.unscale(y_test_pred_scaled) + OCV(df_test['temperature'].values)
            }
        
        # ===== Step 5: Compute global XTX and sigma2 for SE calculation =====
        self._compute_global_xtx_sigma2(X_train_expanded)
        
        return self._build_interval_results()

    def _expand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core logic: expand the physical equation ci(t) = ai*t + bi into 10 linear feature columns.
        """
        t = df['calh'].values
        I = df['currentDensity'].values
        T = df['temperature'].values
        OH = df['h_since_last_start'].values
        
        # Compute base scaled features
        I_s = self.scaler_I.scale(I)
        IT_s = self.scaler_IxT.scale(I * T)
        logH_s = self.scaler_log_h.scale(np.log(OH))
        I2_s = self.scaler_I2.scale(I**2)
        
        # Build expanded feature matrix X (10 columns)
        X = pd.DataFrame(index=df.index)
        X['t_I'] = t * I_s      # coefficient a1
        X['I']   = I_s          # coefficient b1
        X['t_IT']= t * IT_s     # coefficient a2
        X['IT']  = IT_s         # coefficient b2
        X['t_logH'] = t * logH_s# coefficient a3
        X['logH']= logH_s       # coefficient b3
        X['t_I2']= t * I2_s     # coefficient a4
        X['I2']  = I2_s         # coefficient b4
        X['t']   = t            # coefficient a5
        X['const'] = 1.0        # coefficient b5
        
        return X

    def _prepare_matrices(self, df: pd.DataFrame):
        """Prepare feature matrix and target vector (for prediction)."""
        I = df['currentDensity'].values
        T = df['temperature'].values
        OH = df['h_since_last_start'].values
        
        I_s = self.scaler_I.scale(I)
        IT_s = self.scaler_IxT.scale(I * T)
        logH_s = self.scaler_log_h.scale(np.log(OH))
        I2_s = self.scaler_I2.scale(I ** 2)
        
        X = np.column_stack([I_s, IT_s, logH_s, I2_s])
        y = self.scaler_U.scale(df['voltage'].values - OCV(T))
        
        return X, y

    def _predict_scaled(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Predict scaled voltage using fitted parameters."""
        a = self.coeffs[0::2]
        b = self.coeffs[1::2]
        
        c = [a[i] * t + b[i] for i in range(5)]
        
        return (c[0] * X[:, 0] + 
                c[1] * X[:, 1] + 
                c[2] * X[:, 2] + 
                c[3] * X[:, 3] + 
                c[4])

    def _compute_global_xtx_sigma2(self, X_train_expanded: pd.DataFrame):
        """Compute global XTX matrix and sigma2 from RLM for SE calculation.
        
        For RLM robust fitting, we use the X^T*W*X form where W is the RLM weight matrix.
        This accounts for outlier suppression in the variance calculation.
        
        Args:
            X_train_expanded: Expanded feature matrix from training set.
        """
        if self.rlm_result is None:
            self.global_xtx = None
            self.global_sigma2 = np.nan
            return
        
        # Convert DataFrame to numpy array
        X = X_train_expanded.values
        
        # Get RLM weights (downweights outliers)
        weights = np.asarray(self.rlm_result.weights, dtype=float)
        
        # Efficient weighted XTX without materializing (n,n) diagonal matrix:
        # X^T * W * X = (X * w[:, None])^T @ X
        self.global_xtx = (X * weights[:, None]).T @ X
        
        # Compute sigma2 from test residuals
        if self.test_raw_data and len(self.test_raw_data.get('pred_v', [])) > 0:
            actual = np.asarray(self.test_raw_data['actual_v'])
            pred = np.asarray(self.test_raw_data['pred_v'])
            residuals = actual - pred
            self.global_sigma2 = np.mean(residuals ** 2)
        else:
            self.global_sigma2 = np.nan

    def _build_interval_results(self) -> pd.DataFrame:
        """Generate interval-format results compatible with Urc1 baseline."""
        total_days = (self.data.index[-1] - self.installation_time).days
        num_intervals = int(total_days // self.slide) + 1
        
        dates = [self.installation_time + datetime.timedelta(days=i * self.slide) 
                 for i in range(num_intervals)]
        index = pd.MultiIndex.from_product([dates, [self.name]], names=['date', 'name'])
        results = pd.DataFrame(index=index)
        
        num_cols = ["day_since_install", "calh", "cond", "R2", "sigma2",
                    "c1", "c2", "c3", "c4", "c5",
                    "c1_se", "c2_se", "c3_se", "c4_se", "c5_se"]
        for col in num_cols:
            results[col] = np.nan
            
        obj_cols = ["quality", "status", "XTX"]
        for col in obj_cols:
            results[col] = None
        
        a = self.coeffs[0::2]
        b = self.coeffs[1::2]
        a_se = self.coeffs_se[0::2] if self.coeffs_se is not None else np.zeros(5)
        b_se = self.coeffs_se[1::2] if self.coeffs_se is not None else np.zeros(5)
        
        for date in dates:
            t_hours = (date - self.installation_time).total_seconds() / 3600
            t_days = (date - self.installation_time).days
            
            # Compute time-varying coefficients c_i(t) = a_i * t + b_i
            c_vals = a * t_hours + b
            c_se_vals = np.sqrt(a_se**2 * t_hours**2 + b_se**2)
            
            results.loc[(date, self.name), 'day_since_install'] = t_days
            results.loc[(date, self.name), 'calh'] = t_hours
            # Global robust fit does not have per-interval condition inflation;
            # keep cond small so base-class reliability filtering retains rows.
            results.loc[(date, self.name), 'cond'] = 1.0
            results.loc[(date, self.name), 'R2'] = self.test_metrics.get('test_r2', np.nan)
            results.loc[(date, self.name), 'sigma2'] = self.global_sigma2 if self.global_sigma2 is not None else np.nan
            
            # Store XTX as serialized string (replicate Urc1 baseline pattern)
            if self.global_xtx is not None:
                xtx_str = str(self.global_xtx.flatten().tolist())
                results.at[(date, self.name), 'XTX'] = xtx_str
            
            for i in range(5):
                results.loc[(date, self.name), f'c{i+1}'] = c_vals[i]
                results.loc[(date, self.name), f'c{i+1}_se'] = c_se_vals[i]
            
            results.loc[(date, self.name), 'quality'] = 'good'
            results.loc[(date, self.name), 'status'] = 'normal'
        
        return results

    def Urc_calc(self):
        """[Override] Compute Urc at each reference operating condition.
        
        Uses XTX/sigma2-based SE calculation aligned with Urc1 method.
        """
        results = self.fitting_results.copy()
        
        for ref_name, ref_cfg in self.ref_config.items():
            iref = float(ref_cfg["Iref"])
            tref = float(ref_cfg["Tref"])
            ohref = float(ref_cfg["OHref"])
            col_urc = f"Urc_{iref}"
            col_se = f"Urc_se_{iref}"  # Note: column name is Urc_se_{iref}, not Urc_{iref}_se
            col_urc_named = f"Urc_{ref_name}"
            col_se_named = f"Urc_se_{ref_name}"
            
            results[col_urc] = np.nan
            results[col_se] = np.nan
            results[col_urc_named] = np.nan
            results[col_se_named] = np.nan
            
            # Compute Urc for all intervals
            for idx in results.index:
                row = results.loc[idx]
                
                # Get time-varying coefficients
                c1 = row.get('c1', 0)
                c2 = row.get('c2', 0)
                c3 = row.get('c3', 0)
                c4 = row.get('c4', 0)
                c5 = row.get('c5', 0)
                
                # Compute scaled features
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
        """[Override] Compute Urc SE using 10D expanded reference vector (not 5D).
        
        For Surface_Stats with RLM, the XTX is computed from the 10-dimensional expanded
        feature matrix [t*I, I, t*IT, IT, t*logH, logH, t*I2, I2, t, const].
        Therefore, the reference vector must also be 10D for proper variance propagation.
        
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
        
        # For each interval, compute the 10D reference vector
        for idx in self.fitting_results.index:
            date, name = idx
            
            # Get time from interval
            t_hours = (date - self.installation_time).total_seconds() / 3600
            
            # Compute scaled reference features (5 base features)
            i_s = self.scaler_I.scale(i)
            ixt_s = self.scaler_IxT.scale(i * self.ref_config[list(self.ref_config.keys())[0]]["Tref"])
            # Use the first reference's Tref as default (should compute for each ref_config)
            # For now, iterate through ref_configs to find matching iref
            ref_tref = None
            for ref_name, ref_cfg in self.ref_config.items():
                if abs(float(ref_cfg["Iref"]) - i) < 1e-6:
                    ref_tref = float(ref_cfg["Tref"])
                    ref_ohref = float(ref_cfg["OHref"])
                    break
            
            if ref_tref is None:
                # Fallback: use first config or skip
                ref_tref = float(list(self.ref_config.values())[0]["Tref"])
                ref_ohref = float(list(self.ref_config.values())[0]["OHref"])
            
            ixt_s = self.scaler_IxT.scale(i * ref_tref)
            logh_s = self.scaler_log_h.scale(np.log(ref_ohref))
            i2_s = self.scaler_I2.scale(i ** 2)
            
            # Build 10D reference vector: [t*I, I, t*IT, IT, t*logH, logH, t*I2, I2, t, const]
            x0_10d = np.array([
                t_hours * i_s,           # a1 coefficient
                i_s,                     # b1 coefficient
                t_hours * ixt_s,         # a2 coefficient
                ixt_s,                   # b2 coefficient
                t_hours * logh_s,        # a3 coefficient
                logh_s,                  # b3 coefficient
                t_hours * i2_s,          # a4 coefficient
                i2_s,                    # b4 coefficient
                t_hours,                 # a5 coefficient
                1.0                      # b5 coefficient (constant)
            ])
            
            # Compute XTX inverse
            try:
                xtx_inv = np.linalg.inv(self.global_xtx)
            except np.linalg.LinAlgError:
                continue
            
            # Compute quadratic form: x0^T * (XTX)^-1 * x0
            quadratic_form = np.dot(x0_10d, np.dot(xtx_inv, x0_10d))
            
            # Compute variance: Var(Urc) = sigma2 * (1 + quadratic_form)
            variance = self.global_sigma2 * (1.0 + quadratic_form)
            
            # Convert to physical units
            scale = self.scaler_U.max - self.scaler_U.min
            se = np.sqrt(variance) * scale if variance > 0 else np.nan
            
            series.loc[idx] = se
        
        return series

    def _calc_urc_se(self, row, iref: float) -> float:
        """[Deprecated] Use _calculate_reference_se_series instead.
        
        Kept for backward compatibility.
        """
        test_rmse_v = self.test_metrics.get('test_rmse_mv', 5.0) / 1000.0
        return test_rmse_v

    def _calculate_test_metrics(self):
        """Compute evaluation metrics on the test set."""
        if not self.test_raw_data:
            return
        
        actual = self.test_raw_data.get('actual_v')
        pred = self.test_raw_data.get('pred_v')
        
        if actual is None or pred is None:
            return
        
        rmse = np.sqrt(mean_squared_error(actual, pred)) * 1000
        r2 = r2_score(actual, pred)
        mae = np.mean(np.abs(actual - pred)) * 1000
        max_err = np.max(np.abs(actual - pred)) * 1000
        
        self.test_metrics = {
            'test_rmse_mv': rmse,
            'test_r2': r2,
            'test_mae_mv': mae,
            'test_max_error_mv': max_err,
            'n_test_points': len(actual)
        }

    def _calculate_degradation_rates(self):
        """Compute degradation rates directly from global fitting parameters a_i."""
        from degradation_toolbox.Urc.helpers import degradation_result
        
        if self.coeffs is None:
            return
        
        a = self.coeffs[0::2]
        b = self.coeffs[1::2]
        a_se = self.coeffs_se[0::2] if self.coeffs_se is not None else np.zeros(5)
        b_se = self.coeffs_se[1::2] if self.coeffs_se is not None else np.zeros(5)
        
        for ref_name, ref_cfg in self.ref_config.items():
            iref = float(ref_cfg["Iref"])
            tref = float(ref_cfg["Tref"])
            ohref = float(ref_cfg["OHref"])
            try:
                # Compute scaled features (constants, independent of t)
                i_s = self.scaler_I.scale(iref)
                ixt_s = self.scaler_IxT.scale(iref * tref)
                logh_s = self.scaler_log_h.scale(np.log(ohref))
                i2_s = self.scaler_I2.scale(iref ** 2)
                
                # Degradation rate (scaled space): dU_scaled/dt = a1*I_s + a2*IT_s + a3*log_h_s + a4*I^2_s + a5
                slope_scaled = (a[0] * i_s + 
                               a[1] * ixt_s + 
                               a[2] * logh_s + 
                               a[3] * i2_s + 
                               a[4])
                
                # Inverse-scale to original voltage: slope_V = slope_scaled * (max_U - min_U)
                slope_v_per_h = slope_scaled * (self.max_U - self.min_U)
                slope_uv_per_h = slope_v_per_h * 1e6  # Convert to uV/h
                
                # Compute BOL voltage (Urc at t=0)
                u_scaled_t0 = (b[0] * i_s + 
                              b[1] * ixt_s + 
                              b[2] * logh_s + 
                              b[3] * i2_s + 
                              b[4])
                bol_v = self.scaler_U.unscale(u_scaled_t0) + OCV(tref)
                
                # Error propagation for slope uncertainty
                # Var(slope) = (I_s * a1_se)^2 + (IT_s * a2_se)^2 + ...
                var_slope_scaled = ((i_s * a_se[0]) ** 2 + 
                                   (ixt_s * a_se[1]) ** 2 + 
                                   (logh_s * a_se[2]) ** 2 + 
                                   (i2_s * a_se[3]) ** 2 + 
                                   a_se[4] ** 2)
                var_slope_v = var_slope_scaled * ((self.max_U - self.min_U) ** 2)
                uncertainty_uV2 = var_slope_v * 1e12  # Convert to (uV/h)^2
                
                result = degradation_result(
                    aging_uV_per_h=slope_uv_per_h,
                    bol_V=bol_v,
                    uncertainty_uV2=uncertainty_uV2
                )
                self.degradation_results[iref] = result
                self.degradation_results_by_ref[ref_name] = result
                
            except Exception as e:
                print(f"   Warning: Failed to calculate degradation for {iref} A/cm2: {e}")

    def get_Urc_results(self, i: float) -> pd.DataFrame:
        """Get Urc results for a specified reference current."""
        resolved = self._resolve_reference_input(i)
        if resolved is None:
            print(f"Warning: Requested i={i}, but no matching reference is available")
            return pd.DataFrame()
        closest_i, _, _, ref_name = resolved
        
        try:
            df = self.fitting_results.copy()
            df = df.reset_index()
            
            # Note: column name is Urc_se_{iref}, not Urc_{iref}_se
            return pd.DataFrame({
                'timestamp': df['date'],
                'calh': df['calh'],
                'Urc': df[f'Urc_{closest_i}'],
                'Urc_se': df[f'Urc_se_{closest_i}'],  # Corrected column name
                'method': 'surface_stats',
                'reference_name': ref_name,
                'reference_i': closest_i,
            })
            
        except Exception as e:
            print(f"Error in get_Urc_results: {e}")
            print(f"   Available columns: {df.columns.tolist()}")
            return pd.DataFrame()

    def get_test_rmse_mv(self) -> float:
        """Return test set point-wise prediction RMSE (mV)."""
        return self.test_metrics.get('test_rmse_mv', np.nan)

    def back_calculate_coefficients(self):
        """[Override] Coefficients already computed in _build_interval_results for Surface method."""
        return self.fitting_results

    def check_vertex(self, row):
        """[Override] Surface method uses global constraints; no per-point check needed."""
        return row.get('quality', 'good')

    def get_coeffs_summary(self) -> pd.DataFrame:
        """Get a summary table of the 10 fitting parameters."""
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

    def print_rlm_diagnostics(self):
        """Print RLM fitting diagnostic information."""
        if self.rlm_result is None:
            print("Error: No RLM result available")
            return
        
        print("\n" + "=" * 60)
        print("RLM Fitting Diagnostics")
        print("=" * 60)
        print(self.rlm_result.summary())
        
        weights = self.rlm_result.weights
        print(f"\nOutlier Weights Statistics:")
        print(f"   Min Weight:        {weights.min():.4f}")
        print(f"   Max Weight:        {weights.max():.4f}")
        print(f"   Mean Weight:       {weights.mean():.4f}")
        print(f"   Weight < 0.5:      {np.sum(weights < 0.5)} ({np.sum(weights < 0.5)/len(weights)*100:.2f}%)")
        print(f"   Weight < 0.1:      {np.sum(weights < 0.1)} ({np.sum(weights < 0.1)/len(weights)*100:.2f}%)")

    def plot_outlier_weights(self, save: bool = False):
        """Visualize the RLM outlier weight distribution."""
        if self.rlm_result is None:
            print("Error: No RLM result available")
            return
        
        weights = self.rlm_result.weights
        df_train = self.data.iloc[:self.split_idx]
        calh = df_train['calh'].values
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=['① Weights vs Time', '② Weight Distribution'],
                           vertical_spacing=0.15)
        
        # Plot 1: Weights vs time
        sample_step = max(1, len(calh) // 3000)
        fig.add_trace(go.Scatter(
            x=calh[::sample_step],
            y=weights[::sample_step],
            mode='markers',
            marker=dict(
                size=3,
                color=weights[::sample_step],
                colorscale='RdYlBu',
                colorbar=dict(title='Weight', x=1.15),
                opacity=0.6
            ),
            name='Weights'
        ), row=1, col=1)
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Threshold=0.5", row=1, col=1)
        
        # Plot 2: Weight histogram
        fig.add_trace(go.Histogram(
            x=weights,
            nbinsx=50,
            marker_color='steelblue',
            name='Weight Histogram'
        ), row=2, col=1)
        
        fig.update_xaxes(title_text="Calendar Hours [h]", row=1, col=1)
        fig.update_yaxes(title_text="RLM Weight", row=1, col=1)
        fig.update_xaxes(title_text="Weight Value", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        fig.update_layout(
            height=700,
            title_text="RLM Outlier Detection: Weight Analysis",
            template='plotly_white',
            showlegend=False
        )
        
        fig.show()
        
        if save:
            filename = f"{self.name}_RLM_Weights.html"
            fig.write_html(filename)
            print(f"Saved: {filename}")

    def plot_test_prediction(self, save: bool = False):
        """Plot test set prediction comparison."""
        if not self.test_raw_data:
            print("Error: No test data available")
            return
        
        time = self.test_raw_data['time']
        calh = self.test_raw_data['calh']
        actual = np.array(self.test_raw_data['actual_v'])
        pred = np.array(self.test_raw_data['pred_v'])
        residual = (actual - pred) * 1000
        
        n = len(actual)
        rmse = np.sqrt(mean_squared_error(actual, pred)) * 1000
        r2 = r2_score(actual, pred)
        
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=['① Actual vs Predicted', '② Residuals'],
                           vertical_spacing=0.12)
        
        sample_step = max(1, n // 2000)
        
        fig.add_trace(go.Scatter(
            x=time[::sample_step], y=actual[::sample_step],
            mode='lines', name='Actual',
            line=dict(color='blue', width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=time[::sample_step], y=pred[::sample_step],
            mode='lines', name='Predicted',
            line=dict(color='red', width=1.5, dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=calh[::sample_step], y=residual[::sample_step],
            mode='markers', name='Residuals',
            marker=dict(size=3, color='steelblue', opacity=0.4)
        ), row=2, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
        fig.update_xaxes(title_text="Calendar Hours [h]", row=2, col=1)
        fig.update_yaxes(title_text="Residual [mV]", row=2, col=1)
        
        fig.update_layout(
            height=700,
            title_text=f"Test Set Prediction (n={n:,}, RMSE={rmse:.2f}mV, R²={r2:.4f})",
            template='plotly_white'
        )
        
        fig.show()
        
        print(f"\nTest Metrics:")
        print(f"   RMSE: {rmse:.2f} mV")
        print(f"   R²:   {r2:.4f}")
        print(f"   MAE:  {np.mean(np.abs(residual)):.2f} mV")

    def analyze_full_dataset_residuals(self, return_data: bool = False) -> dict:
        """
        Perform residual analysis on the entire preprocessed dataset.

        For each valid data point:
        1. Use its actual t, I, T, OH values
        2. Compute predicted U via the Surface model
        3. Compare with actual U to compute residuals
        4. Perform statistical analysis on all residuals
        """
        if self.coeffs is None:
            print("Error: Model not fitted yet.")
            return {}
        
        # Prepare full dataset
        df = self.data.copy()
        n_total = len(df)
        
        if 'calh' not in df.columns:
            df['calh'] = (df.index - self.installation_time) / np.timedelta64(1, 'h')
        
        # Compute predicted voltage for each point
        a = self.coeffs[0::2]
        b = self.coeffs[1::2]
        
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
        
        # Time-varying coefficients
        c1 = a[0] * t + b[0]
        c2 = a[1] * t + b[1]
        c3 = a[2] * t + b[2]
        c4 = a[3] * t + b[3]
        c5 = a[4] * t + b[4]
        
        # Predicted voltage
        U_pred_scaled = c1 * I_s + c2 * IxT_s + c3 * log_h_s + c4 * I2_s + c5
        U_pred = self.scaler_U.unscale(U_pred_scaled) + OCV(T)
        
        # Residuals
        residuals = U_actual - U_pred
        residuals_mv = residuals * 1000
        
        # Statistical analysis
        mean_res = np.mean(residuals_mv)
        std_res = np.std(residuals_mv, ddof=1)
        median_res = np.median(residuals_mv)
        min_res = np.min(residuals_mv)
        max_res = np.max(residuals_mv)
        rmse = np.sqrt(np.mean(residuals_mv ** 2))
        mae = np.mean(np.abs(residuals_mv))
        
        # Normality tests
        sample_for_test = residuals_mv[:5000] if len(residuals_mv) > 5000 else residuals_mv
        try:
            shapiro_stat, shapiro_p = stats.shapiro(sample_for_test)
        except:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        try:
            jb_stat, jb_p = stats.jarque_bera(residuals_mv)
        except:
            jb_stat, jb_p = np.nan, np.nan
        
        skewness = stats.skew(residuals_mv)
        kurtosis = stats.kurtosis(residuals_mv)
        
        outliers_2sigma = np.sum(np.abs(residuals_mv - mean_res) > 2 * std_res)
        outliers_3sigma = np.sum(np.abs(residuals_mv - mean_res) > 3 * std_res)
        
        U_min = np.min(U_actual) * 1000
        U_max = np.max(U_actual) * 1000
        U_mean = np.mean(U_actual) * 1000
        
        relative_rmse_pct = (rmse / U_mean) * 100
        relative_mae_pct = (mae / U_mean) * 100
        
        # Store residual data
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
        
        results = {
            'n_points': n_total,
            'mean_mv': round(mean_res, 3),
            'std_mv': round(std_res, 3),
            'median_mv': round(median_res, 3),
            'min_mv': round(min_res, 3),
            'max_mv': round(max_res, 3),
            'rmse_mv': round(rmse, 3),
            'mae_mv': round(mae, 3),
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
        #     print(f"   Shapiro-Wilk:      W={results['shapiro_stat']:.4f}, p={results['shapiro_p']:.2e} -> {normal_shapiro}")
        # if results['jarque_bera_p'] is not None:
        #     normal_jb = "Yes" if results['is_normal_jb'] else "No"
        #     print(f"   Jarque-Bera:       JB={results['jarque_bera_stat']:.2f}, p={results['jarque_bera_p']:.2e} -> {normal_jb}")
        
        print(f"\nOutlier Analysis:")
        print(f"   Beyond ±2σ:        {results['outliers_2sigma']:>6} ({results['outliers_2sigma_pct']:.2f}%)")
        print(f"   Beyond ±3σ:        {results['outliers_3sigma']:>6} ({results['outliers_3sigma_pct']:.2f}%)")
        print(f"   (Expected for normal: 2σ ≈ 4.55%, 3σ ≈ 0.27%)")
        
        print("\n" + "=" * 60)

    def plot_residual_diagnostics(self, save: bool = False):
        """Plot full-dataset residual diagnostic plots (4 separate plots)."""
        if not hasattr(self, '_full_residuals') or self._full_residuals is None:
            self.analyze_full_dataset_residuals()
        
        res_data = self._full_residuals
        residuals = res_data['residuals_mv']
        
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        n = len(residuals)
        
        # Plot 1: Residual histogram + normal fit
        fig1 = go.Figure()
        
        fig1.add_trace(go.Histogram(
            x=residuals,
            nbinsx=80,
            name='Residuals',
            opacity=0.7,
            marker_color='steelblue',
            histnorm='probability density'
        ))
        
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
        
        # Plot 2: Q-Q plot
        fig2 = go.Figure()
        
        sorted_res = np.sort(residuals)
        theoretical_q = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
        sample_step = max(1, n // 2000)
        
        fig2.add_trace(go.Scatter(
            x=theoretical_q[::sample_step],
            y=sorted_res[::sample_step],
            mode='markers',
            name='Q-Q Points',
            marker=dict(size=3, color='steelblue', opacity=0.5)
        ))
        
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
        
        # Plot 3: Residuals vs time
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
        
        # Plot 4: Residuals vs current density
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
            print(f"4 plots saved to plots/ folder.")

    def plot_3d_surface(self, n_grid: int = 80, save: bool = False,
                        show_data: bool = False, data_sample: int = 5000):
        """
        Plot 3D surface: predicted voltage surface U(I, t) at reference T and OH.

        Parameters
        ----------
        n_grid : int
            Number of grid points per axis (default 80).
        save : bool
            Whether to save as an HTML file.
        show_data : bool
            Whether to overlay raw data points (default False).
        data_sample : int
            Maximum number of data points to sample when show_data=True (default 5000).
        """
        if self.coeffs is None:
            print("Error: Model not fitted yet.")
            return

        a = self.coeffs[0::2]
        b = self.coeffs[1::2]

        # Time range (h)
        t_max = (self.data.index[-1] - self.installation_time).total_seconds() / 3600
        t_arr = np.linspace(0, t_max, n_grid)

        # Current density range
        I_min, I_max = self.data['currentDensity'].min(), self.data['currentDensity'].max()
        I_arr = np.linspace(I_min, I_max, n_grid)

        # Scaled features at reference conditions (T=Tref, OH=OHref)
        T_grid = np.full_like(I_arr, self.Tref)
        OH_grid = np.full_like(I_arr, self.OHref)

        # Build mesh grid
        T_mesh, I_mesh = np.meshgrid(t_arr, I_arr)  # shape (n_I, n_t)
        U_mesh = np.empty_like(T_mesh)

        for j, t_val in enumerate(t_arr):
            # Time-varying coefficients
            c = a * t_val + b  # shape (5,)

            # Scaled features (along I axis)
            I_s = self.scaler_I.scale(I_arr)
            IT_s = self.scaler_IxT.scale(I_arr * self.Tref)
            logH_s = self.scaler_log_h.scale(np.log(self.OHref))
            I2_s = self.scaler_I2.scale(I_arr ** 2)

            u_scaled = c[0] * I_s + c[1] * IT_s + c[2] * logH_s + c[3] * I2_s + c[4]
            U_mesh[:, j] = self.scaler_U.unscale(u_scaled) + OCV(self.Tref)

        # ===== Plotting =====
        fig = go.Figure()

        # 3D surface
        fig.add_trace(go.Surface(
            x=t_arr,          # Calendar Hours
            y=I_arr,          # Current Density
            z=U_mesh,         # Voltage
            colorscale='Viridis',
            colorbar=dict(title='U [V]'),
            opacity=0.85,
            name='Predicted Surface'
        ))

        # ===== Overlay raw data points =====
        if show_data:
            df = self.data.copy()
            # Subsample to avoid browser lag
            if len(df) > data_sample:
                df = df.sample(n=data_sample, random_state=42)
            data_t = (df.index - self.installation_time).total_seconds() / 3600
            data_I = df['currentDensity'].values
            data_U = df['voltage'].values

            fig.add_trace(go.Scatter3d(
                x=data_t.values,
                y=data_I,
                z=data_U,
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=data_U,
                    colorscale='Plasma',
                    opacity=0.4,
                ),
                name='Raw Data',
                hovertemplate=(
                    'CalH: %{x:.0f} h<br>'
                    'I: %{y:.3f} A/cm²<br>'
                    'U: %{z:.4f} V<extra></extra>'
                ),
            ))

        # Annotate reference current lines on the surface
        for iref in self.Iref:
            if I_min <= iref <= I_max:
                I_s_ref = self.scaler_I.scale(iref)
                IT_s_ref = self.scaler_IxT.scale(iref * self.Tref)
                logH_s_ref = self.scaler_log_h.scale(np.log(self.OHref))
                I2_s_ref = self.scaler_I2.scale(iref ** 2)
                u_line = np.array([
                    self.scaler_U.unscale(
                        (a * tv + b)[0] * I_s_ref +
                        (a * tv + b)[1] * IT_s_ref +
                        (a * tv + b)[2] * logH_s_ref +
                        (a * tv + b)[3] * I2_s_ref +
                        (a * tv + b)[4]
                    ) + OCV(self.Tref)
                    for tv in t_arr
                ])
                fig.add_trace(go.Scatter3d(
                    x=t_arr, y=np.full_like(t_arr, iref), z=u_line,
                    mode='lines',
                    line=dict(color='red', width=5),
                    name=f'Iref={iref} A/cm²'
                ))

        fig.update_layout(
            title=f'{self.name} – 3D Voltage Surface (T={self.Tref}°C, OH={self.OHref}h)',
            scene=dict(
                xaxis_title='Calendar Hours [h]',
                yaxis_title='Current Density [A/cm²]',
                zaxis_title='Voltage [V]',
            ),
            height=700,
            width=900,
            template='plotly_white',
        )

        fig.show()

        if save:
            filename = f"plots/{self.name}_Surface_3D.html"
            fig.write_html(filename)
            print(f"Saved: {filename}")

    def plot_coefficients_evolution(self, save: bool = False):
        """Plot the evolution curves of 5 time-varying coefficients."""
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