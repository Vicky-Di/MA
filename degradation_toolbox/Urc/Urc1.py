import datetime
import time as _time
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.optimize import lsq_linear
from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_squared_error
from degradation_toolbox.utils.calculation_basics import find_interval, calc_h_since_last_start


OCV_SOURCE = "CoolProp linear approximation at 1 atm"

# --- Main Class ---

class Urc1:
    def __init__(self,
                 data,
                 name="Elyzer",
                 Iref=[0.6, 1.5],
                 Tref=60,
                 OHref=3 * 24,
                 ref_config=None,
                 i_off=0.1,
                 u_off=1.3,
                 resample=1,
                 len_interval=2,
                 slide=1,
                 min_num_data_required_for_fit=0.5 * 1 * 24 * 60,
                 threshold=10 ** 6,
                 data_filter_i_min=0.1,
                 data_filter_U_min=1.4,
                 data_filter_U_max=2.3,
                 data_filter_T_min=50,
                 data_filter_T_max=65,
                 data_filter_h_since_last_start_min=0.5,
                 plot_fit = True,
                 preprocessed_data=None,
                 fit_solver="curve_fit",
                 **kwargs  # [Opt 1] Allow subclasses to pass extra args
                 ):
        """
        Calculate voltage under reference condition (Urc) using operation data.

        Parameters
        ----------
        data : pd.DataFrame
            Input time series data with columns ["currentDensity", "temperature", "voltage"]
            and a datetime index.
        name : str, default "Elyzer"
            Identifier for the electrolyzer cell or dataset.
        Iref : float or list of float, default [0.6, 1.5]
            Reference current density(ies) in A/cm² for Urc calculation.
        Tref : float, default 60
            Reference temperature in °C for Urc calculation.
        OHref : float, default 72
            Reference operating hours [h] for Urc calculation.
        threshold : float, default 1e6
            Condition number threshold for detecting ill-conditioned least-squares systems.
            
            - **Purpose**: Detects numerical instability in the fitted model
            - **Typical values**:
              * 1e6 (default): Strict - rejects unstable fits
              * 1e5: Very strict - rejects marginal fits
              * 1e7: Lenient - allows more data
            - **Physical meaning**: When cond(X'X) > threshold, parameter uncertainties 
              become unreliable (>100% of parameter value itself).
            - **Reference**: Golub & Kahan, "Calculating the Singular Values and 
              Pseudo-Inverse of a Matrix", SIAM 1965
            
            Fitting results with condition number > threshold are marked as "unreliable" 
            and excluded from degradation rate calculation.
        
        preprocessed_data : pd.DataFrame or None, optional
            If provided, skip data_preprocess() and use this DataFrame directly.
            Must contain columns ["currentDensity", "temperature", "voltage", "h_since_last_start"]
            with a datetime index. Use Urc1.preprocess_once() to create it.
        
        **kwargs : dict
            Additional parameters forwarded to subclasses (e.g., epsilon, alpha for Huber models).
        
        Raises
        ------
        ValueError
            If data is empty after preprocessing or if preprocessed_data is invalid.
        """
        
        # --- Parameter Setup ---
        self._preprocessed_data_provided = preprocessed_data is not None
        self.data = data[["currentDensity", "temperature", "voltage"]]
        self.name = name
        self.ref_config = self._normalize_ref_config(ref_config, Iref, Tref, OHref)
        self.ref_name_to_i = {
            ref_name: float(cfg["Iref"]) for ref_name, cfg in self.ref_config.items()
        }
        self.Iref = [float(cfg["Iref"]) for cfg in self.ref_config.values()]
        self.Tref = float(next(iter(self.ref_config.values()))["Tref"])
        self.OHref = float(next(iter(self.ref_config.values()))["OHref"])
        self.i_off = i_off
        self.u_off = u_off
        self.resample = resample
        self.len_interval = len_interval
        self.slide = slide
        self.min_num_data_required_for_fit = min_num_data_required_for_fit
        self.threshold = threshold
        self.data_filter_i_min = data_filter_i_min
        self.data_filter_U_min = data_filter_U_min
        self.data_filter_U_max = data_filter_U_max
        self.data_filter_T_min = data_filter_T_min
        self.data_filter_T_max = data_filter_T_max
        self.data_filter_h_since_last_start_min = data_filter_h_since_last_start_min
        self.plot_fit = plot_fit
        self.fit_solver = fit_solver
        self._se_cache = None

        # Initialize result container
        self.fitting_results = pd.DataFrame() 

        # --- Data Preprocessing (skip if already preprocessed) ---
        if preprocessed_data is not None:
            self.data = preprocessed_data.copy()
            if self.data.empty:
                raise ValueError("preprocessed_data is empty.")
            self.installation_time = self.data.index[0]
            print(f"Using shared preprocessed data ({len(self.data)} points, skipping preprocess).")
        else:
            print("Data preprocessing ...")
            self.data = self.data_preprocess()
            if not self.data.empty:
                self.installation_time = self.data.index[0]
            else:
                raise ValueError("Data is empty after preprocessing.")

        print("Voltage model fitting ...")
        
        # --- Initialize Scalers (Extracted for subclass reuse) ---
        self._init_scalers()

        # --- Model Fitting ---
        _fit_t0 = _time.perf_counter()
        self.fitting_results = self.model_fitting()
        self.fitting_runtime_seconds = float(_time.perf_counter() - _fit_t0)
        self.timing_metrics = {
            "model_fitting_seconds": self.fitting_runtime_seconds,
        }

        # --- Post-Calculation (Urc & Checks) ---
        self.fitting_results = self.Urc_calc()

        # Check vertex physics
        self.fitting_results = self.back_calculate_coefficients()
        self.fitting_results["quality"] = "good" 
        self.fitting_results["quality"] = self.fitting_results.apply(self.check_vertex, axis=1)

        # --- Filter Reliable Results ---
        if "cond" in self.fitting_results.columns:
            self.fitting_results_reliable = self.fitting_results.query(
                f"cond < {self.threshold} and quality=='good'").reset_index()
        else:
            self.fitting_results_reliable = pd.DataFrame()

        # Comparator-friendly aliases/containers
        self.Urc1_results = self.fitting_results_reliable.copy()
        self.Urc_results = pd.DataFrame()
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
            
        print(f"{len(self.fitting_results_reliable)} out of {len(self.fitting_results)} fitting results are reliable.")

    # --- Modular Methods for Subclassing ---

    @staticmethod
    def _extract_ref_value(cfg, canonical_name):
        """Extract a reference-config value allowing flexible key casing."""
        candidate_keys = [
            canonical_name,
            canonical_name.lower(),
            canonical_name.upper(),
            canonical_name.capitalize(),
        ]
        for key in candidate_keys:
            if key in cfg:
                return cfg[key]
        raise ValueError(f"Reference config must contain '{canonical_name}' (case-insensitive).")

    @classmethod
    def _normalize_ref_config(cls, ref_config, Iref, Tref, OHref):
        """Normalize reference settings into a common internal structure."""
        if ref_config is not None:
            if not isinstance(ref_config, dict) or not ref_config:
                raise ValueError("ref_config must be a non-empty dict when provided.")

            normalized = {}
            for ref_name, cfg in ref_config.items():
                if not isinstance(cfg, dict):
                    raise ValueError(f"ref_config['{ref_name}'] must be a dict.")

                normalized[str(ref_name)] = {
                    "Iref": float(cls._extract_ref_value(cfg, "Iref")),
                    "Tref": float(cfg.get("Tref", cfg.get("tref", Tref))),
                    "OHref": float(cfg.get("OHref", cfg.get("ohref", OHref))),
                }
            return normalized

        if isinstance(Iref, list):
            i_values = [float(v) for v in Iref]
        else:
            i_values = [float(Iref)]

        normalized = {}
        for i_val in i_values:
            normalized[f"Iref_{i_val}"] = {
                "Iref": i_val,
                "Tref": float(Tref),
                "OHref": float(OHref),
            }
        return normalized

    def _resolve_reference_input(self, target):
        """Resolve reference identifier (float current or named ref) to config."""
        if isinstance(target, str):
            if target in self.ref_config:
                cfg = self.ref_config[target]
                return float(cfg["Iref"]), float(cfg["Tref"]), float(cfg["OHref"]), target
            return None

        try:
            target_val = float(target)
        except Exception:
            return None

        if not self.Iref:
            return None

        closest_i = min(self.Iref, key=lambda x: abs(x - target_val))
        if abs(target_val - closest_i) > 0.015:
            return None

        for ref_name, cfg in self.ref_config.items():
            if abs(float(cfg["Iref"]) - closest_i) <= 1e-9:
                return closest_i, float(cfg["Tref"]), float(cfg["OHref"]), ref_name

        return closest_i, self.Tref, self.OHref, f"Iref_{closest_i}"

    def _init_scalers(self):
        """Initialize min-max scalers for features."""
        self.min_I = 0.3
        self.max_I = 1.6
        self.min_IxT = 55 * self.min_I
        self.max_IxT = self.max_I * 65
        self.min_log_h = np.log(1)
        self.max_log_h = np.log(200)
        self.min_U = 0
        self.max_U = 2.3 - 1.23
        
        self.scaler_U = MinMaxScalerCustomize(self.min_U, self.max_U)
        self.scaler_I = MinMaxScalerCustomize(self.min_I, self.max_I)
        self.scaler_IxT = MinMaxScalerCustomize(self.min_IxT, self.max_IxT)
        self.scaler_log_h = MinMaxScalerCustomize(self.min_log_h, self.max_log_h)
        self.scaler_I2 = MinMaxScalerCustomize(self.min_I ** 2, self.max_I ** 2)


    def data_preprocess(self):
        len_data_before_preprocess = len(self.data)

        # Reduce data size
        cols = list(self.data.select_dtypes(include=['float64']))
        self.data[cols] = self.data[cols].astype('float32')

        # Remove duplicates and sort
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        self.data = self.data.sort_index()

        # Resample
        self.data = self.data[::self.resample]
        self.data.index.name = 'timestamp'

        # Calculate operational hours
        self.data = calc_h_since_last_start(self.data, f"currentDensity>{self.i_off} or voltage>{self.u_off}", 1)

        # Filter
        self.data = self.data.dropna().query(f"{self.data_filter_i_min}<currentDensity and "
                                             f"{self.data_filter_U_min}<voltage<{self.data_filter_U_max} and "
                                             f"{self.data_filter_T_min}<temperature<{self.data_filter_T_max} and "
                                             f"h_since_last_start>{self.data_filter_h_since_last_start_min}")
        
        if self.data.empty:
            raise ValueError(f"No data is left after filtering:\n "
              f"i>{self.data_filter_i_min}A/cm2, "
              f"{self.data_filter_U_min}V<U<{self.data_filter_U_max}V, "
              f"{self.data_filter_T_min}degC<T<{self.data_filter_T_max}degC, "
              f"h_since_last_start>{self.data_filter_h_since_last_start_min}h.\n")
        

        len_data_after_preprocess = len(self.data)
        print(f"Before preprocess: {len_data_before_preprocess} data points.\n"
              f"After preprocess: {len_data_after_preprocess} data points.\n "
              f"(Filter criteria: i>{self.data_filter_i_min}A/cm2, "
              f"{self.data_filter_U_min}V<U<{self.data_filter_U_max}V, "
              f"{self.data_filter_T_min}degC<T<{self.data_filter_T_max}degC, "
              f"h_since_last_start>{self.data_filter_h_since_last_start_min}h.)\n")
        return self.data


    @classmethod
    def preprocess_once(cls, data, i_off=0.1, u_off=1.3, resample=1,
                        data_filter_i_min=0.1, data_filter_U_min=1.4,
                        data_filter_U_max=2.3, data_filter_T_min=50,
                        data_filter_T_max=65,
                        data_filter_h_since_last_start_min=0.5):
        """
        Run data_preprocess() once and return the result for reuse across models.

        This avoids repeated heavy preprocessing (dedup, resample, calc_h, filter)
        when comparing multiple models on the same dataset.

        Usage
        -----
        >>> from degradation_toolbox.Urc.Urc1 import Urc1
        >>> shared = Urc1.preprocess_once(raw_data)
        >>> m1 = Urc1(raw_data, preprocessed_data=shared)
        >>> m2 = Urc1BaseHuber(raw_data, preprocessed_data=shared)
        >>> m3 = Urc1_Surface_Stats(raw_data, preprocessed_data=shared)
        """
        df = data[["currentDensity", "temperature", "voltage"]].copy()
        len_before = len(df)

        # float64 → float32
        cols = list(df.select_dtypes(include=['float64']))
        df[cols] = df[cols].astype('float32')

        # deduplicate & sort
        df = df[~df.index.duplicated(keep='first')].sort_index()

        # resample
        df = df[::resample]
        df.index.name = 'timestamp'

        # operational hours
        df = calc_h_since_last_start(
            df, f"currentDensity>{i_off} or voltage>{u_off}", 1
        )

        # filter
        df = df.dropna().query(
            f"{data_filter_i_min}<currentDensity and "
            f"{data_filter_U_min}<voltage<{data_filter_U_max} and "
            f"{data_filter_T_min}<temperature<{data_filter_T_max} and "
            f"h_since_last_start>{data_filter_h_since_last_start_min}"
        )

        if df.empty:
            raise ValueError("No data left after preprocessing.")

        print(f"[preprocess_once] {len_before} -> {len(df)} points.")
        return df


    def save_preprocessed_data(self, output_dir="../explore_data/output"):
        if self.data is None: return
        print("\n=== Saving Preprocessed Data ===")
        df_to_save = self.data.copy().reset_index()
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"Preprocessed_data_{self.name}_{current_time}.parquet"
        df_to_save.to_parquet(os.path.join(output_dir, file_name))

    
    def _prepare_matrices(self, df):
        """
        [Opt 3] Centralized Feature Engineering & Scaling.
        This ensures both Baseline and Adaptive use EXACTLY the same physics/math.
        """
        # 1. Feature Engineering (Use copy to avoid SettingWithCopyWarning)
        df = df.copy() 
        df["U_ocv"] = OCV(df.temperature)
        df["IxT"] = df.currentDensity * df.temperature
        df["log_h"] = np.log(df.h_since_last_start)
        df["I2"] = df.currentDensity ** 2

        # 2. Scaling
        X = np.column_stack((
            self.scaler_I.scale(df.currentDensity),
            self.scaler_IxT.scale(df.IxT),
            self.scaler_log_h.scale(df.log_h),
            self.scaler_I2.scale(df.I2),
        ))
        y = self.scaler_U.scale(df.voltage - df["U_ocv"])
        
        return X, y

    def _solve_interval_fit(self, X, y):
        """Solve per-interval fit with a fast bounded linear solver.

        Model is linear in coefficients [c1, c2, c3, c4, c5], so bounded
        linear least squares is equivalent to the curve_fit formulation but
        significantly faster on large sliding-window loops.
        """
        if self.fit_solver == "curve_fit":
            popt, pcov = curve_fit(
                func, X, y,
                bounds=((0, -np.inf, -np.inf, -np.inf, -np.inf), (np.inf, 0, np.inf, np.inf, np.inf))
            )
            return popt, pcov

        A = np.column_stack((X, np.ones(len(X), dtype=float)))

        if self.fit_solver == "auto_fast":
            popt, *_ = np.linalg.lstsq(A, y, rcond=None)
            # Keep original physical constraints: c1 >= 0 and c2 <= 0.
            if popt[0] >= 0 and popt[1] <= 0:
                dof = len(y) - A.shape[1]
                if dof > 0:
                    residual = y - A @ popt
                    sigma2 = float(np.sum(residual ** 2) / dof)
                    ata_inv = np.linalg.pinv(A.T @ A)
                    pcov = sigma2 * ata_inv
                else:
                    pcov = np.full((A.shape[1], A.shape[1]), np.nan)
                return popt, pcov

            # Fallback for the few windows that violate constraints.
            popt, pcov = curve_fit(
                func, X, y,
                bounds=((0, -np.inf, -np.inf, -np.inf, -np.inf), (np.inf, 0, np.inf, np.inf, np.inf))
            )
            return popt, pcov

        if self.fit_solver != "bounded_lstsq":
            raise ValueError(f"Unsupported fit_solver: {self.fit_solver}")

        lower = np.array([0.0, -np.inf, -np.inf, -np.inf, -np.inf], dtype=float)
        upper = np.array([np.inf, 0.0, np.inf, np.inf, np.inf], dtype=float)

        result = lsq_linear(A, y, bounds=(lower, upper), method="trf", lsmr_tol="auto")
        if not result.success:
            raise RuntimeError(result.message)

        popt = result.x

        dof = len(y) - A.shape[1]
        if dof > 0:
            residual = y - A @ popt
            sigma2 = float(np.sum(residual ** 2) / dof)
            ata = A.T @ A
            ata_inv = np.linalg.pinv(ata)
            pcov = sigma2 * ata_inv
        else:
            pcov = np.full((A.shape[1], A.shape[1]), np.nan)

        return popt, pcov
        

    def model_fitting(self):
        """Per-interval OLS fitting loop using sliding windows.

        Iterates over time intervals, prepares scaled feature matrices,
        fits the linear voltage model via curve_fit, and records coefficients,
        R², RMSE, condition number, and covariance matrices.

        Returns:
            pd.DataFrame: Fitting results indexed by (date, name).
        """
        installation_day = self.installation_time.date()
        end_day = max(self.data.index).date()
        num_days = (end_day - installation_day).days
        num_intervals = math.floor(num_days / self.slide)

        # Create output DataFrame structure\n        
        dates = [self.installation_time + datetime.timedelta(days=i * self.slide) for i in range(num_intervals + 5)]
        index = pd.MultiIndex.from_product([dates, [self.name]], names=['date', 'name'])
        results = pd.DataFrame(index=index)
        
        # Initialize columns
        # Initialize numeric columns
        num_cols = ["day_since_install", "calh", "R2", "RMSE", "cond", "sigma2",
                    "c1", "c2", "c3", "c4", "c5", 
                    "c1_se", "c2_se", "c3_se", "c4_se", "c5_se"]
        for c in num_cols: 
            results[c] = np.nan

        # Initialize object columns (e.g. XTX stores numpy matrix objects)
        obj_cols = ["XTX", "quality", "status"]
        for c in obj_cols:
            results[c] = pd.Series([None] * len(results), index=index, dtype=object)

        num_interv_without_enough_data = 0
        num_interv_curve_fit_failed = 0
        num_plot_fit = 1
        ts = self.data.index.to_numpy()
        max_index_time = self.data.index[-1]

        for interval in range(0, num_intervals + 2):
            # Define Interval
            current_day = self.installation_time + datetime.timedelta(days=interval * self.slide)
            interval_end = current_day + datetime.timedelta(days=self.len_interval)
            
            # Brake check
            if interval_end > max_index_time + datetime.timedelta(days=1):
                break
            idx = (current_day, self.name)
            results.loc[idx, "day_since_install"] = (current_day - self.installation_time).days
            results.loc[idx, "calh"] = (current_day - self.installation_time) / np.timedelta64(1, "h")

            # Slice Data
            left = np.searchsorted(ts, np.datetime64(current_day), side="left")
            right = np.searchsorted(ts, np.datetime64(interval_end), side="right")
            data_this_interval = self.data.iloc[left:right]

            if len(data_this_interval) < self.min_num_data_required_for_fit:
                num_interv_without_enough_data += 1
                results.loc[idx, "status"] = f"Insufficient Data ({len(data_this_interval)} < {self.min_num_data_required_for_fit})"
                continue
            
            try:
                # [Core Reuse] Use the shared matrix preparation logic
                X, y = self._prepare_matrices(data_this_interval)

                # Fit (Baseline uses OLS, no weights)
                popt, pcov = self._solve_interval_fit(X, y)
                condition_number = np.linalg.cond(pcov)

                # --- Record Results ---
                idx = (current_day, self.name)
                
                # Predict for stats (Note: y is scaled, so we check R2 on unscaled/true voltage)
                U_true = data_this_interval.voltage
                U_predicted = self.scaler_U.unscale(func(X, *popt)) + OCV(data_this_interval.temperature)
                
                
                results.loc[idx, "R2"] = r2_score(U_true, U_predicted)
                results.loc[idx, "RMSE"] = np.sqrt(mean_squared_error(U_true, U_predicted))
                results.loc[idx, "cond"] = condition_number
                
                # Coefficients & Errors
                for k in range(5):
                    results.loc[idx, f"c{k+1}"] = popt[k]
                    results.loc[idx, f"c{k+1}_se"] = np.sqrt(np.diag(pcov)[k])

                # SE Calculation Params (Saved for Urc_se_calc)
                X_with_constant = np.c_[X, np.ones(len(X))]
                XTX = X_with_constant.T @ X_with_constant
                results.at[idx, "XTX"] = XTX
                results.loc[idx, "sigma2"] = np.sum((y - func(X, *popt)) ** 2) / (len(y) - X_with_constant.shape[1])

                # Optional Plotting (simplified)
                if condition_number < self.threshold and num_plot_fit <= self.plot_fit:
                    num_plot_fit += 1

                if condition_number >= self.threshold:
                    results.loc[idx, "status"] = f"High Cond ({condition_number:.1e})"
                else:
                    results.loc[idx, "status"] = "Fit Success"

            except Exception as e:
                num_interv_curve_fit_failed += 1
                results.loc[idx, "status"] = f"Fit Error: {str(e)}"

        print(f"Fitting Stats: {num_interv_without_enough_data} intervals low data, {num_interv_curve_fit_failed} fit failed.")
        return results

    def Urc_calc(self):
        """Calculate Urc and its SE using fitted coefficients."""
        for ref_name, ref_cfg in self.ref_config.items():
            i = float(ref_cfg["Iref"])
            tref = float(ref_cfg["Tref"])
            ohref = float(ref_cfg["OHref"])
            Uref_scaled = self.fitting_results["c1"] * self.scaler_I.scale(i)
            Uref_scaled += self.fitting_results["c2"] * self.scaler_IxT.scale(tref * i)
            Uref_scaled += self.fitting_results["c3"] * self.scaler_log_h.scale(np.log(ohref))
            Uref_scaled += self.fitting_results["c4"] * self.scaler_I2.scale(i ** 2)
            Uref_scaled += self.fitting_results["c5"]
            
            urc_series = self.scaler_U.unscale(Uref_scaled) + OCV(tref)
            self.fitting_results[f"Urc_{i}"] = urc_series
            self.fitting_results[f"Urc_se_{i}"] = self._calculate_reference_se_series(i)
            self.fitting_results[f"Urc_{ref_name}"] = urc_series
            self.fitting_results[f"Urc_se_{ref_name}"] = self._calculate_reference_se_series(ref_name)
        return self.fitting_results


    def calculate_urc_for_custom_refs(self, ref_list):
        """Compute Urc for arbitrary reference condition sets (post-training).

        After model fitting, this method evaluates Urc at different reference
        combinations without re-training. Useful for Low/Medium/High analysis.

        Args:
            ref_list: List of dicts, each containing:
                - 'iref' (float): Reference current density [A/cm^2]
                - 'tref' (float, optional): Reference temperature [deg C]
                - 'ohref' (float, optional): Reference operating hours [h]
                - 'name' (str, optional): Label for this reference set

        Returns:
            Dict[str, pd.DataFrame]: Mapping from name to DataFrame with
                columns [timestamp, urc, urc_se].
        """
        if not hasattr(self, 'fitting_results_reliable'):
            print("Warning: Model not yet fitted (fitting_results_reliable unavailable)")
            return {}
        
        results = {}
        
        for ref_config in ref_list:
            iref = self._extract_ref_value(ref_config, 'Iref') if any(
                key in ref_config for key in ['Iref', 'iref', 'IREF']
            ) else None
            tref = ref_config.get('Tref', ref_config.get('tref', self.Tref))
            ohref = ref_config.get('OHref', ref_config.get('ohref', self.OHref))
            name = ref_config.get('name', f"Iref_{iref}_Tref_{tref}_OHref_{ohref}")
            
            if iref is None:
                print(f"Warning: Skipping ref config without 'iref': {ref_config}")
                continue
            
            try:
                # Compute scaled reference feature vector
                i_s = self.scaler_I.scale(float(iref))
                ixt_s = self.scaler_IxT.scale(float(tref) * float(iref))
                logh_s = self.scaler_log_h.scale(np.log(float(ohref)))
                i2_s = self.scaler_I2.scale(float(iref) ** 2)
                
                # Compute scaled Urc
                u_scaled = (
                    self.fitting_results_reliable["c1"] * i_s +
                    self.fitting_results_reliable["c2"] * ixt_s +
                    self.fitting_results_reliable["c3"] * logh_s +
                    self.fitting_results_reliable["c4"] * i2_s +
                    self.fitting_results_reliable["c5"]
                )
                
                # Inverse-scale and add OCV
                u_physical = self.scaler_U.unscale(u_scaled) + OCV(float(tref))
                
                # Build a robust timestamp vector for both MultiIndex and single index cases
                fr_rel = self.fitting_results_reliable
                if isinstance(fr_rel.index, pd.MultiIndex):
                    if 'date' in fr_rel.index.names:
                        ts = fr_rel.index.get_level_values('date')
                    else:
                        ts = fr_rel.index.get_level_values(0)
                elif 'date' in fr_rel.columns:
                    ts = pd.to_datetime(fr_rel['date'])
                elif 'timestamp' in fr_rel.columns:
                    ts = pd.to_datetime(fr_rel['timestamp'])
                else:
                    ts = pd.to_datetime(fr_rel.index)

                # Use numpy values to avoid index-alignment errors (MultiIndex vs RangeIndex)
                result_df = pd.DataFrame({
                    'timestamp': np.asarray(ts),
                    'urc': np.asarray(u_physical, dtype=float),
                })
                
                # Compute SE via error propagation (simplified: weighted sum of c_se)
                w = np.array([i_s, ixt_s, logh_s, i2_s, 1.0])
                var_terms = []
                for j, c_name in enumerate(['c1', 'c2', 'c3', 'c4', 'c5']):
                    se_col = f"{c_name}_se"
                    if se_col in self.fitting_results_reliable.columns:
                        var_terms.append((w[j]**2) * (self.fitting_results_reliable[se_col]**2))
                
                if var_terms:
                    var_u_scaled = sum(var_terms)
                    se_u_scaled = np.sqrt(var_u_scaled)
                    # Inverse-scale SE back to physical voltage units
                    scale_factor = self.scaler_U.max - self.scaler_U.min
                    result_df['urc_se'] = np.asarray(se_u_scaled, dtype=float) * scale_factor
                else:
                    result_df['urc_se'] = np.nan
                
                results[name] = result_df
                print(f"Calculated Urc for {name}: {len(result_df)} points")
                
            except Exception as e:
                print(f"Error calculating Urc for {name}: {e}")
        
        return results

    def get_reference_vector(self, i):
        resolved = self._resolve_reference_input(i)
        if resolved is None:
            raise ValueError(f"Unknown reference target: {i}")

        closest_i, tref, ohref, _ = resolved
        return np.array([
            self.scaler_I.scale(closest_i),
            self.scaler_IxT.scale(tref * closest_i),
            self.scaler_log_h.scale(np.log(ohref)),
            self.scaler_I2.scale(closest_i ** 2),
            1.0,
        ], dtype=float)

    def _get_reference_vector(self, i):
        """Backward-compatible alias for reference feature-vector generation.

        Some call sites still use the old private method name
        ``_get_reference_vector``. Keep this thin wrapper to preserve
        behaviour without changing model logic.
        """
        return self.get_reference_vector(i)

    def _coerce_xtx_matrix(self, xtx_value, size=5):
        if xtx_value is None:
            return None

        if isinstance(xtx_value, np.ndarray):
            xtx = np.asarray(xtx_value, dtype=float)
        elif isinstance(xtx_value, (list, tuple)):
            xtx = np.asarray(xtx_value, dtype=float)
        elif isinstance(xtx_value, str):
            xtx = xtx_value.replace('[', '').replace(']', '')
            xtx = np.fromstring(xtx, sep=',')
            if xtx.size != size * size:
                return None
            xtx = xtx.reshape(size, size)
        else:
            return None

        if xtx.shape != (size, size):
            return None
        return xtx

    def _calculate_reference_se_series(self, i):
        series = pd.Series(np.nan, index=self.fitting_results.index, dtype=float)
        if "XTX" not in self.fitting_results.columns or "sigma2" not in self.fitting_results.columns:
            return series

        x0 = self._get_reference_vector(i)
        if self._se_cache is None:
            valid_rows = self.fitting_results["XTX"].notna() & self.fitting_results["sigma2"].notna()
            if not valid_rows.any():
                return series

            matrices = []
            valid_index = []
            for idx, xtx_value in self.fitting_results.loc[valid_rows, "XTX"].items():
                xtx = self._coerce_xtx_matrix(xtx_value, size=len(x0))
                if xtx is None:
                    continue
                matrices.append(xtx)
                valid_index.append(idx)

            if not matrices:
                return series

            xtx_stack = np.stack(matrices)
            try:
                xtx_inv = np.linalg.inv(xtx_stack)
            except np.linalg.LinAlgError:
                inverse_list = []
                inverse_index = []
                for idx, xtx in zip(valid_index, matrices):
                    try:
                        inverse_list.append(np.linalg.inv(xtx))
                        inverse_index.append(idx)
                    except np.linalg.LinAlgError:
                        continue
                if not inverse_list:
                    return series
                xtx_inv = np.stack(inverse_list)
                valid_index = inverse_index

            sigma2 = self.fitting_results.loc[valid_index, "sigma2"].to_numpy(dtype=float)
            self._se_cache = {
                "valid_index": valid_index,
                "xtx_inv": xtx_inv,
                "sigma2": sigma2,
            }

        valid_index = self._se_cache["valid_index"]
        xtx_inv = self._se_cache["xtx_inv"]
        sigma2 = self._se_cache["sigma2"]
        quadratic_form = np.einsum("i,nij,j->n", x0, xtx_inv, x0)
        variance = sigma2 * (1.0 + quadratic_form)
        scale = self.scaler_U.max - self.scaler_U.min
        se = np.where(variance > 0, np.sqrt(variance) * scale, np.nan)
        series.loc[valid_index] = se
        return series

    def Uref_se_calc(self, row, i):
        """Calculate prediction interval SE."""
        x0 = self._get_reference_vector(i)
        XTX = self._coerce_xtx_matrix(row.get("XTX"), size=len(x0))
        if XTX is None:
            return np.nan

        sigma2 = row["sigma2"]
        # Variance of prediction = sigma2 * (1 + x0^T * (X^T X)^-1 * x0)
        variance = sigma2 * (1 + x0.T @ np.linalg.inv(XTX) @ x0)
        
        if variance > 0:
            Uref_scaled_se = np.sqrt(variance)
            return Uref_scaled_se * (self.scaler_U.max - self.scaler_U.min)
        return np.nan

    def back_calculate_coefficients(self):
        """Convert scaled coefficients back to physical units."""
        fr = self.fitting_results
        max_U, min_U = self.max_U, self.min_U
        # Helper to avoid repetitive code
        def unscale_coeff(val, dim_max, dim_min):
            return val * max_U / (dim_max - dim_min)

        fr["back_calc_c1"] = unscale_coeff(fr.c1, self.max_I, self.min_I)
        fr["back_calc_c2"] = unscale_coeff(fr.c2, self.max_IxT, self.min_IxT)
        fr["back_calc_c3"] = unscale_coeff(fr.c3, self.max_log_h, self.min_log_h)
        fr["back_calc_c4"] = unscale_coeff(fr.c4, self.max_I**2, self.min_I**2)
        
        # c5 is more complex
        fr["back_calc_c5"] = max_U * (fr.c5 
            - fr.c1 / (self.max_I - self.min_I) * self.min_I
            - fr.c2 / (self.max_IxT - self.min_IxT) * self.min_IxT
            - fr.c3 / (self.max_log_h - self.min_log_h) * self.min_log_h
            - fr.c4 / (self.max_I**2 - self.min_I**2) * (self.min_I**2))
        return fr
    
    def check_vertex(self, row):
        """Check if the UI curve vertex is within physical range."""
        if pd.isna(row["back_calc_c4"]): return "failed"
        
        # Do not override status if already flagged by condition number check
        if row["status"] and "High Cond" in str(row["status"]):
            return row["quality"]

        c1, c2, c4 = row["back_calc_c1"], row["back_calc_c2"], row["back_calc_c4"]
        if c4 == 0: return "linear"
        vertex_I = - (c1 + c2 * self.Tref) / (2 * c4)
        
        if 0 < vertex_I < 2.5:
            return "Vertex of UI curve is between 0 and 2.5A/cm2."
        return "good"

    def get_Urc_results(self, i):
        """Return reliable Urc results in a comparator-compatible format."""
        resolved = self._resolve_reference_input(i)
        if resolved is None:
            return pd.DataFrame()

        closest_i, _, _, ref_name = resolved

        if not hasattr(self, 'Urc1_results') or self.Urc1_results is None or self.Urc1_results.empty:
            return pd.DataFrame()

        col_urc = f"Urc_{closest_i}"
        col_se = f"Urc_se_{closest_i}"
        if col_urc not in self.Urc1_results.columns or col_se not in self.Urc1_results.columns:
            return pd.DataFrame()

        df = self.Urc1_results.copy()
        if 'date' not in df.columns:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            else:
                df = df.reset_index().rename(columns={df.index.name or 'index': 'date'})

        if 'date' not in df.columns:
            return pd.DataFrame()

        to_save = df[["date", "calh", col_urc, col_se]].copy()
        to_save["method"] = "Urc1"
        to_save["reference_name"] = ref_name
        to_save["reference_i"] = closest_i
        to_save.rename(
            columns={
                "date": "timestamp",
                col_urc: "Urc",
                col_se: "Urc_se",
            },
            inplace=True,
        )

        self.Urc_results = to_save
        return self.Urc_results

    def get_degradation_rate(self, i):
        """Calculate degradation rate from reliable Urc points."""
        resolved = self._resolve_reference_input(i)
        if resolved is None:
            return None

        closest_i, _, _, ref_name = resolved

        urc_results = self.get_Urc_results(closest_i)
        if urc_results is None or urc_results.empty:
            return None

        mask = urc_results["calh"].notna() & urc_results["Urc"].notna()
        if mask.sum() < 2:
            return None

        slope, intercept, _, _, stderr = linregress(
            urc_results.loc[mask, "calh"],
            urc_results.loc[mask, "Urc"],
        )

        from collections import namedtuple
        DegResult = namedtuple('DegResult', ['aging_uV_per_h', 'bol_V', 'uncertainty_uV2'])
        result = DegResult(
            aging_uV_per_h=slope * 1e6,
            bol_V=intercept,
            uncertainty_uV2=stderr ** 2,
        )

        self.degradation_results[closest_i] = result
        self.degradation_results_Urc1[closest_i] = result
        self.degradation_results_by_ref[ref_name] = result
        return result


# --- Global Helper Functions ---

def func(X, c1, c2, c3, c4, c5):
    """
    The empirical voltage function to be fitted:
    U = c1*I + c2*I*T + c3*ln(h) + c4*I^2 + c5
    """
    return c1 * X[:, 0] + c2 * X[:, 1] + c3 * X[:, 2] + c4 * X[:, 3] + c5

def OCV(T):
    """
    Calculate the open circuit voltage (OCV) of the H2O-->H2+O2 reaction under a given temperature 'T'.
    The calculation is an linear approximation to the calculation results given by the CoolProp package (http://www.coolprop.org/coolprop/HighLevelAPI.html).
    Pressure is assumed to be atmospheric pressure in the calculation.

    Args:
        T (float): Temperature in Degree Celsius.

    Returns:
        float: Open circuit voltage under T°C.
    """
    return -8.2975 * (10 ** (-4)) * T + 1.24965


class MinMaxScalerCustomize:
    """
    Custom Scaler to map specific physical ranges to 0-1.
    """
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def scale(self, X):
        return (X - self.min) / (self.max - self.min)

    def unscale(self, X_scaled):
        return (X_scaled) * (self.max - self.min) + self.min


def plot_Urc1(Urc1_multi_cells, plot_name="Urc1 (Multi cells)"):
    """
    Helper function to plot multiple Urc1 instances.
    """
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.figure()
    for cell in Urc1_multi_cells:
        for i in cell.Iref:
            if f"Urc_{i}" in cell.fitting_results_reliable.columns:
                plt.errorbar(cell.fitting_results_reliable.calh, cell.fitting_results_reliable[f"Urc_{i}"],
                            yerr=cell.fitting_results_reliable[f"Urc_se_{i}"],
                            linestyle="", marker=".", label=f"{cell.name}, {i}A/cm2")
    plt.xticks(rotation=45)
    plt.legend()
    plt.xlabel("Calendar hours since test start [h]")
    plt.ylabel("U [V]")
    plt.title(plot_name)
    plt.grid()
    plt.savefig(f"../plots/{plot_name}.png")
    print(f"{plot_name}.png is saved in folder 'plots'.")

