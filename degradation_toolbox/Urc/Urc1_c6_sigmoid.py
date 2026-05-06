"""Urc1_c6_sigmoid: Urc1 variant with nonlinear I term (sigmoid and cube-root).

Key modifications compared to Urc1 (baseline):

1. Sigmoid model (6 parameters):
   - Baseline linear I² term is replaced with a sigmoid nonlinearity
   - Model: U = c1*I_s + c2*(IT)_s + c3*ln(h)_s + c4*sigmoid(I) + c5
   - Captures sharp phase transitions or kinetic phenomena
   - Parameter c6 shifts the sigmoid inflection point

2. Cube-root model (6 parameters):
   - Baseline linear I² term is replaced with cube-root term
   - Model: U = c1*I_s + c2*(IT)_s + c3*ln(h)_s + c4*(I - c6)^(1/3) + c5
   - Describes electrochemical processes with fractional-order kinetics
   - Parameter c6 shifts the reference current

Both use curve_fit with bounds on c4, c6 to ensure physical feasibility.
"""

import datetime
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

from degradation_toolbox.Urc.helpers import calculate_degradation_rate_and_uncertainty, degradation_result
from degradation_toolbox.utils.calculation_basics import calc_h_since_last_start

pd.options.mode.copy_on_write = True

# --- Global Helper Functions ---

def func(X, c1, c2, c3, c4, c5):
    """
    The empirical voltage function to be fitted (original I² version):
    U = c1*I_s + c2*(IT)_s + c3*ln(h)_s + c4*I2_s + c5
    X columns: [I_s, IT_s, logH_s, I2_s]
    """
    return c1 * X[:, 0] + c2 * X[:, 1] + c3 * X[:, 2] + c4 * X[:, 3] + c5


def func_cbrt(X, c1, c2, c3, c4, c5, c6):
    """
    Nonlinear voltage function with cube-root term:
    U = c1*I_s + c2*(IT)_s + c3*ln(h)_s + c4*(I_raw - c6)^(1/3) + c5
    
    X columns: [I_s, IT_s, logH_s, I_raw]
    The 4th column is RAW (unscaled) current density for the cbrt term.
    """
    return (c1 * X[:, 0] + c2 * X[:, 1] + c3 * X[:, 2]
            + c4 * np.cbrt(X[:, 3] - c6) + c5)


def func_sigmoid(X, c1, c2, c3, c4, c5, c6, _steepness=5.0):
    """
    Nonlinear voltage function with sigmoid term:
    U = c1*I_s + c2*(IT)_s + c3*ln(h)_s + c4 * sigmoid(k*(I_raw - c6)) + c5
    
    X columns: [I_s, IT_s, logH_s, I_raw]
    The 4th column is RAW (unscaled) current density for the sigmoid term.
    sigmoid(x) = 1 / (1 + exp(-x))
    
    _steepness controls how sharp the sigmoid transition is.
    Default k=5 ensures meaningful nonlinear variation within I ∈ [0.3, 1.6].
    """
    sig = 1.0 / (1.0 + np.exp(-_steepness * (X[:, 3] - c6)))
    return (c1 * X[:, 0] + c2 * X[:, 1] + c3 * X[:, 2]
            + c4 * sig + c5)


def OCV(T):
    """
    Calculate the open circuit voltage (OCV) of the H2O-->H2+O2 reaction.
    Linear approximation to CoolProp results at atmospheric pressure.
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


# --- Main Class ---

class Urc1_c6_sigmoid:
    def __init__(self,
                 data,
                 name="Elyzer",
                 Iref=[0.6, 1.5],
                 Tref=60,
                 OHref=3 * 24,
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
                 plot_fit=True,
                 nonlinear_term="I2",  # NEW: "I2" (original), "cbrt", or "sigmoid"
                 preprocessed_data=None,
                 ref_config=None,
                 **kwargs
                 ):
        """
        Calculate voltage under reference condition (Urc) using operation data.
        
        Parameters
        ----------
        nonlinear_term : str, default "I2"
            Which nonlinear current term to use in the voltage model:
            - "I2"      : original c4*I^2 (5 parameters, linear in params)
            - "cbrt"    : c4*(I - c6)^(1/3) (6 parameters, nonlinear)
            - "sigmoid" : c4*sigmoid(I - c6) (6 parameters, nonlinear)
        preprocessed_data : pd.DataFrame or None
            If provided, skip data_preprocess() and use this DataFrame directly.
            Use Urc1.preprocess_once() to create it.

        kwargs
        ------
        sigmoid_steepness : float or "auto", default 5.0
            Steepness k in sigmoid(k*(I-c6)). If set to "auto", k is inferred
            from the current-density span of preprocessed data via
            k = 2*ln(9)/(q95-q05).
        """
        
        # 1. Parameter Setup
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
        self.run_in = kwargs.get("run_in", 0)
        self.regression_method = kwargs.get("regression_method", "orth")
        self.fix_point = kwargs.get("fix_point", None)
        
        # NEW: nonlinear term selection
        if nonlinear_term not in ("I2", "cbrt", "sigmoid"):
            raise ValueError(f"nonlinear_term must be 'I2', 'cbrt', or 'sigmoid', got '{nonlinear_term}'")
        self.nonlinear_term = nonlinear_term
        self.n_params = 5 if nonlinear_term == "I2" else 6
        self._sigmoid_steepness_cfg = kwargs.get("sigmoid_steepness", 5.0)
        self.sigmoid_steepness = None
        self._se_cache = None

        # Initialize result container
        self.fitting_results = pd.DataFrame() 

        # 2. Data Preprocessing (skip if already preprocessed)
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

        if self.nonlinear_term == "sigmoid":
            self.sigmoid_steepness = self._resolve_sigmoid_steepness()
            print(f"Using sigmoid_steepness={self.sigmoid_steepness:.3f}")
        else:
            self.sigmoid_steepness = 5.0

        print(f"Voltage model fitting (nonlinear_term='{self.nonlinear_term}') ...")
        
        # Initialize Scalers
        self._init_scalers()

        # 3. Model Fitting
        self.fitting_results = self.model_fitting()

        # 4. Post-Calculation (Urc & Checks)
        self.fitting_results = self.Urc_calc()

        # Check vertex physics
        self.fitting_results = self.back_calculate_coefficients()
        self.fitting_results["quality"] = "good" 
        self.fitting_results["quality"] = self.fitting_results.apply(self.check_vertex, axis=1)

        # 5. Filter Reliable Results
        if "cond" in self.fitting_results.columns:
            self.fitting_results_reliable = self.fitting_results.query(
                f"cond < {self.threshold} and quality=='good'").reset_index()
        else:
            self.fitting_results_reliable = pd.DataFrame()

        self.Urc1_results_incl_unreliable = self.fitting_results.reset_index()
        self.Urc1_results = self.fitting_results_reliable.copy()
        self.Urc_results = pd.DataFrame()
        self.degradation_results = {}
        self.degradation_results_Urc1 = {}
        self.degradation_results_by_ref = {}

        for i_ref in self.Iref:
            deg_result = self.get_degradation_rate(i_ref)
            if deg_result is not None:
                self.degradation_results[i_ref] = deg_result
                self.degradation_results_Urc1[i_ref] = deg_result
        for ref_name in self.ref_name_to_i:
            deg_result = self.get_degradation_rate(ref_name)
            if deg_result is not None:
                self.degradation_results_by_ref[ref_name] = deg_result
            
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

    @staticmethod
    def _normalize_ref_config(ref_config, Iref, Tref, OHref):
        """Normalize reference settings into an ordered dict-like mapping.

        Returns a dict of:
            {
                ref_name: {"Iref": float, "Tref": float, "OHref": float},
                ...
            }
        """
        if ref_config is not None:
            if not isinstance(ref_config, dict) or not ref_config:
                raise ValueError("ref_config must be a non-empty dict when provided.")

            normalized = {}
            for ref_name, cfg in ref_config.items():
                if not isinstance(cfg, dict):
                    raise ValueError(f"ref_config['{ref_name}'] must be a dict.")
                normalized[str(ref_name)] = {
                    "Iref": float(Urc1_c6_sigmoid._extract_ref_value(cfg, "Iref")),
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

        # Pick the first matching named config for this Iref (if any)
        for ref_name, cfg in self.ref_config.items():
            if abs(float(cfg["Iref"]) - closest_i) <= 1e-9:
                return closest_i, float(cfg["Tref"]), float(cfg["OHref"]), ref_name

        return closest_i, self.Tref, self.OHref, f"Iref_{closest_i}"

    def _resolve_sigmoid_steepness(self):
        """
        Resolve sigmoid steepness from config.

        - If float: use directly (must be > 0).
                - If "auto": infer k from robust current span (q95-q05) using
                    logistic 10%-90% width relation, i.e.,
                    k = 2*ln(9)/(q95-q05).
        """
        cfg = self._sigmoid_steepness_cfg

        if isinstance(cfg, str):
            if cfg.lower() != "auto":
                raise ValueError("sigmoid_steepness must be a positive float or 'auto'.")

            i_vals = self.data["currentDensity"].to_numpy(dtype=float)
            if len(i_vals) == 0:
                return 5.0

            q05, q95 = np.quantile(i_vals, [0.05, 0.95])
            i_span = max(float(q95 - q05), 1e-6)
            # Logistic 10%-90% width: w = 2*ln(9)/k  =>  k = 2*ln(9)/w
            k_auto = (2.0 * np.log(9.0)) / i_span
            return k_auto

        try:
            k = float(cfg)
        except Exception as exc:
            raise ValueError("sigmoid_steepness must be a positive float or 'auto'.") from exc

        if k <= 0:
            raise ValueError("sigmoid_steepness must be > 0.")
        return k

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

    def _get_fit_func(self):
        """Return the appropriate fitting function based on nonlinear_term."""
        if self.nonlinear_term == "I2":
            return func
        elif self.nonlinear_term == "cbrt":
            return func_cbrt
        elif self.nonlinear_term == "sigmoid":
            k = self.sigmoid_steepness
            def _sigmoid_with_steepness(X, c1, c2, c3, c4, c5, c6):
                return func_sigmoid(X, c1, c2, c3, c4, c5, c6, _steepness=k)
            return _sigmoid_with_steepness

    def _get_bounds(self):
        """Return parameter bounds for curve_fit based on nonlinear_term."""
        if self.nonlinear_term == "I2":
            # 5 params: c1>0, c2<0, c3/c4/c5 free
            lower = (0, -np.inf, -np.inf, -np.inf, -np.inf)
            upper = (np.inf, 0, np.inf, np.inf, np.inf)
        elif self.nonlinear_term == "cbrt":
            # 6 params: c1>0, c2<0, c3/c4/c5 free, c6 free
            lower = (0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)
            upper = (np.inf, 0, np.inf, np.inf, np.inf, np.inf)
        elif self.nonlinear_term == "sigmoid":
            # 6 params: c1>0, c2<0, c3/c4/c5 free, c6 bounded to data range
            lower = (0, -np.inf, -np.inf, -np.inf, -np.inf, 0.0)
            upper = (np.inf, 0, np.inf, np.inf, np.inf, 2.0)
        return (lower, upper)

    def _get_p0(self):
        """Return initial guess for curve_fit based on nonlinear_term."""
        if self.nonlinear_term == "I2":
            return None
        elif self.nonlinear_term == "cbrt":
            return [1.0, -0.1, 0.1, 0.1, 0.5, 0.5]
        elif self.nonlinear_term == "sigmoid":
            return [1.0, -0.1, 0.1, 0.1, 0.5, 0.8]

    def _prepare_matrices(self, df):
        """
        Centralized Feature Engineering & Scaling.
        
        For "I2" mode:   X columns = [I_s, IT_s, logH_s, I2_s]
        For "cbrt"/"sigmoid" mode: X columns = [I_s, IT_s, logH_s, I_raw]
        """
        df = df.copy() 
        df["U_ocv"] = OCV(df.temperature)
        df["IxT"] = df.currentDensity * df.temperature
        df["log_h"] = np.log(df.h_since_last_start)

        col_I_s = self.scaler_I.scale(df.currentDensity)
        col_IxT_s = self.scaler_IxT.scale(df.IxT)
        col_logH_s = self.scaler_log_h.scale(df.log_h)

        if self.nonlinear_term == "I2":
            df["I2"] = df.currentDensity ** 2
            col_4 = self.scaler_I2.scale(df.I2)
        else:
            # For cbrt and sigmoid: pass RAW current density as the 4th column
            col_4 = df.currentDensity.values

        X = np.column_stack((col_I_s, col_IxT_s, col_logH_s, col_4))
        y = self.scaler_U.scale(df.voltage - df["U_ocv"])
        
        return X, y

    def _calc_nonlinear_term_value(self, i_raw, popt):
        """
        Calculate the value of the nonlinear term for a given raw current density.
        Used in Urc_calc and _check_physics.
        """
        if self.nonlinear_term == "I2":
            return popt[3] * self.scaler_I2.scale(i_raw ** 2)
        elif self.nonlinear_term == "cbrt":
            return popt[3] * np.cbrt(i_raw - popt[5])
        elif self.nonlinear_term == "sigmoid":
            sig = 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (i_raw - popt[5])))
            return popt[3] * sig

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
            raise ValueError(f"No data is left after filtering.")

        len_data_after_preprocess = len(self.data)
        print(f"Before preprocess: {len_data_before_preprocess} -> After: {len_data_after_preprocess} points.")
        return self.data

    @classmethod
    def preprocess_once(cls, data, i_off=0.1, u_off=1.3, resample=1,
                        data_filter_i_min=0.1, data_filter_U_min=1.4,
                        data_filter_U_max=2.3, data_filter_T_min=50,
                        data_filter_T_max=65,
                        data_filter_h_since_last_start_min=0.5):
        """
        Run preprocessing once and reuse the result across multiple models.
        """
        df = data[["currentDensity", "temperature", "voltage"]].copy()
        len_before = len(df)

        cols = list(df.select_dtypes(include=['float64']))
        df[cols] = df[cols].astype('float32')

        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df[::resample]
        df.index.name = 'timestamp'

        df = calc_h_since_last_start(
            df, f"currentDensity>{i_off} or voltage>{u_off}", 1
        )

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

    def model_fitting(self):
        """
        Optimized fitting loop, supports I2/cbrt/sigmoid via _get_fit_func().
        """
        installation_day = self.installation_time.date()
        end_day = max(self.data.index).date()
        num_days = (end_day - installation_day).days
        num_intervals = math.floor(num_days / self.slide)

        # Create output DataFrame structure
        dates = [self.installation_time + datetime.timedelta(days=i * self.slide) for i in range(num_intervals + 5)]
        index = pd.MultiIndex.from_product([dates, [self.name]], names=['date', 'name'])
        results = pd.DataFrame(index=index)
        
        # Initialize columns — dynamically handle c6 for nonlinear models
        coeff_cols = [f"c{k+1}" for k in range(self.n_params)]
        coeff_se_cols = [f"c{k+1}_se" for k in range(self.n_params)]
        num_cols = ["day_since_install", "calh", "R2", "RMSE", "cond", "sigma2"] + coeff_cols + coeff_se_cols
        for c in num_cols: 
            results[c] = np.nan

        obj_cols = ["XTX", "quality", "status"]
        for c in obj_cols:
            results[c] = pd.Series([None] * len(results), index=index, dtype=object)

        num_interv_without_enough_data = 0
        num_interv_curve_fit_failed = 0
        num_plot_fit = 1
        max_index_time = self.data.index[-1]
        ts = self.data.index.to_numpy()

        fit_func = self._get_fit_func()
        bounds = self._get_bounds()
        p0 = self._get_p0()

        for interval in range(0, num_intervals + 2):
            current_day = self.installation_time + datetime.timedelta(days=interval * self.slide)
            interval_end = current_day + datetime.timedelta(days=self.len_interval)
            
            if interval_end > max_index_time + datetime.timedelta(days=1):
                break
            idx = (current_day, self.name)
            results.loc[idx, "day_since_install"] = (current_day - self.installation_time).days
            results.loc[idx, "calh"] = (current_day - self.installation_time) / np.timedelta64(1, "h")

            left = np.searchsorted(ts, np.datetime64(current_day), side="left")
            right = np.searchsorted(ts, np.datetime64(interval_end), side="right")
            data_this_interval = self.data.iloc[left:right]

            if len(data_this_interval) < self.min_num_data_required_for_fit:
                num_interv_without_enough_data += 1
                results.loc[idx, "status"] = f"Insufficient Data ({len(data_this_interval)} < {self.min_num_data_required_for_fit})"
                continue
            
            try:
                X, y = self._prepare_matrices(data_this_interval)

                popt, pcov = curve_fit(
                    fit_func, X, y,
                    p0=p0,
                    bounds=bounds,
                    maxfev=10000
                )
                condition_number = np.linalg.cond(pcov)

                U_predicted = self.scaler_U.unscale(fit_func(X, *popt)) + OCV(data_this_interval.temperature)
                U_true = data_this_interval.voltage
                
                results.loc[idx, "R2"] = r2_score(U_true, U_predicted)
                results.loc[idx, "RMSE"] = np.sqrt(mean_squared_error(U_true, U_predicted))
                results.loc[idx, "cond"] = condition_number
                
                for k in range(self.n_params):
                    results.loc[idx, f"c{k+1}"] = popt[k]
                    results.loc[idx, f"c{k+1}_se"] = np.sqrt(np.diag(pcov)[k])

                # SE Calculation Params
                if self.nonlinear_term == "I2":
                    X_with_constant = np.c_[X, np.ones(len(X))]
                    XTX = X_with_constant.T @ X_with_constant
                    residuals = y - fit_func(X, *popt)
                    sigma2 = np.sum(residuals ** 2) / (len(y) - X_with_constant.shape[1])
                    results.at[idx, "XTX"] = XTX
                    results.loc[idx, "sigma2"] = sigma2
                else:
                    # For nonlinear models, store pcov for SE propagation
                    residuals = y - fit_func(X, *popt)
                    sigma2 = np.sum(residuals ** 2) / (len(y) - self.n_params)
                    results.loc[idx, "sigma2"] = sigma2
                    results.at[idx, "XTX"] = np.asarray(pcov, dtype=float)

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
        self._se_cache = None
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
            
            # Nonlinear term
            if self.nonlinear_term == "I2":
                Uref_scaled += self.fitting_results["c4"] * self.scaler_I2.scale(i ** 2)
            elif self.nonlinear_term == "cbrt":
                Uref_scaled += self.fitting_results["c4"] * np.cbrt(i - self.fitting_results["c6"])
            elif self.nonlinear_term == "sigmoid":
                sig = 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (i - self.fitting_results["c6"])))
                Uref_scaled += self.fitting_results["c4"] * sig
            
            Uref_scaled += self.fitting_results["c5"]
            
            urc_series = self.scaler_U.unscale(Uref_scaled) + OCV(tref)
            urc_se_series = self._calculate_reference_se_series(i)

            # Backward-compatible numeric columns used by comparator.
            self.fitting_results[f"Urc_{i}"] = urc_series
            self.fitting_results[f"Urc_se_{i}"] = urc_se_series

            # Named columns for explicit Low/Medium/High-style references.
            self.fitting_results[f"Urc_{ref_name}"] = urc_series
            self.fitting_results[f"Urc_se_{ref_name}"] = urc_se_series
        return self.fitting_results

    def calculate_urc_for_custom_refs(self, ref_list):
        """Compute Urc for arbitrary reference condition sets (post-training)."""
        if not hasattr(self, 'fitting_results_reliable'):
            print("Warning: Model not yet fitted (fitting_results_reliable unavailable)")
            return {}

        results = {}
        fr_rel = self.fitting_results_reliable

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
                iref = float(iref)
                tref = float(tref)
                ohref = float(ohref)

                i_s = self.scaler_I.scale(iref)
                ixt_s = self.scaler_IxT.scale(tref * iref)
                logh_s = self.scaler_log_h.scale(np.log(ohref))

                # Build nonlinear term according to model type
                if self.nonlinear_term == "I2":
                    nonlin_term = self.scaler_I2.scale(iref ** 2)
                elif self.nonlinear_term == "cbrt":
                    nonlin_term = np.cbrt(iref - fr_rel["c6"].to_numpy(dtype=float))
                else:
                    sig = 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (iref - fr_rel["c6"].to_numpy(dtype=float))))
                    nonlin_term = sig

                u_scaled = (
                    fr_rel["c1"].to_numpy(dtype=float) * i_s +
                    fr_rel["c2"].to_numpy(dtype=float) * ixt_s +
                    fr_rel["c3"].to_numpy(dtype=float) * logh_s +
                    fr_rel["c4"].to_numpy(dtype=float) * nonlin_term +
                    fr_rel["c5"].to_numpy(dtype=float)
                )

                u_physical = self.scaler_U.unscale(u_scaled) + OCV(tref)

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

                result_df = pd.DataFrame({
                    'timestamp': np.asarray(ts),
                    'urc': np.asarray(u_physical, dtype=float),
                })

                # Approximate uncertainty propagation using available coefficient SE columns
                if self.nonlinear_term == "I2":
                    w = np.array([i_s, ixt_s, logh_s, self.scaler_I2.scale(iref ** 2), 1.0], dtype=float)
                    se_cols = ["c1_se", "c2_se", "c3_se", "c4_se", "c5_se"]
                    var_terms = [
                        (w[j] ** 2) * (fr_rel[se_col].to_numpy(dtype=float) ** 2)
                        for j, se_col in enumerate(se_cols)
                        if se_col in fr_rel.columns
                    ]
                else:
                    # Nonlinear variants use derivative-based weights for c4/c6
                    if self.nonlinear_term == "cbrt":
                        diff = iref - fr_rel["c6"].to_numpy(dtype=float)
                        d_c4 = np.cbrt(diff)
                        d_c6 = np.where(np.abs(diff) > 1e-10,
                                        -fr_rel["c4"].to_numpy(dtype=float) / (3.0 * np.cbrt(diff ** 2)),
                                        0.0)
                    else:
                        sig = 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (iref - fr_rel["c6"].to_numpy(dtype=float))))
                        d_c4 = sig
                        d_c6 = -fr_rel["c4"].to_numpy(dtype=float) * self.sigmoid_steepness * sig * (1.0 - sig)

                    weights = [i_s, ixt_s, logh_s, d_c4, 1.0, d_c6]
                    se_cols = ["c1_se", "c2_se", "c3_se", "c4_se", "c5_se", "c6_se"]
                    var_terms = [
                        (weights[j] ** 2) * (fr_rel[se_col].to_numpy(dtype=float) ** 2)
                        for j, se_col in enumerate(se_cols)
                        if se_col in fr_rel.columns
                    ]

                if var_terms:
                    var_u_scaled = np.sum(var_terms, axis=0)
                    se_u_scaled = np.sqrt(np.maximum(var_u_scaled, 0.0))
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
        return self.get_reference_vector(i)

    def _coerce_xtx_matrix(self, xtx_value, size=None):
        if xtx_value is None:
            return None

        if isinstance(xtx_value, np.ndarray):
            xtx = np.asarray(xtx_value, dtype=float)
        elif isinstance(xtx_value, (list, tuple)):
            xtx = np.asarray(xtx_value, dtype=float)
        elif isinstance(xtx_value, str):
            xtx = xtx_value.replace('[', '').replace(']', '')
            xtx = np.fromstring(xtx, sep=',')
            if size is None:
                size = int(np.sqrt(xtx.size))
            if xtx.size != size * size:
                return None
            xtx = xtx.reshape(size, size)
        else:
            return None

        if xtx.ndim != 2 or xtx.shape[0] != xtx.shape[1]:
            return None
        if size is not None and xtx.shape != (size, size):
            return None
        return xtx

    def _calculate_reference_se_series(self, i):
        series = pd.Series(np.nan, index=self.fitting_results.index, dtype=float)
        if "XTX" not in self.fitting_results.columns:
            return series

        linear_mode = self.nonlinear_term == "I2"
        size = 5 if linear_mode else self.n_params

        if self._se_cache is None:
            valid_rows = self.fitting_results["XTX"].notna() & self.fitting_results["sigma2"].notna()
            if not valid_rows.any():
                return series

            matrices = []
            valid_index = []
            for idx, xtx_value in self.fitting_results.loc[valid_rows, "XTX"].items():
                xtx = self._coerce_xtx_matrix(xtx_value, size=size)
                if xtx is None:
                    continue
                matrices.append(xtx)
                valid_index.append(idx)

            if not matrices:
                return series

            if linear_mode:
                matrix_stack = np.stack(matrices)
                try:
                    inv_stack = np.linalg.inv(matrix_stack)
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
                    inv_stack = np.stack(inverse_list)
                    valid_index = inverse_index

                sigma2 = self.fitting_results.loc[valid_index, "sigma2"].to_numpy(dtype=float)
                self._se_cache = {
                    "mode": "linear",
                    "valid_index": valid_index,
                    "inv_stack": inv_stack,
                    "sigma2": sigma2,
                }
            else:
                self._se_cache = {
                    "mode": "nonlinear",
                    "valid_index": valid_index,
                    "pcov_list": matrices,
                }

        if linear_mode:
            if self._se_cache.get("mode") != "linear":
                return series
            x0 = self._get_reference_vector(i)
            valid_index = self._se_cache["valid_index"]
            inv_stack = self._se_cache["inv_stack"]
            sigma2 = self._se_cache["sigma2"]
            quadratic_form = np.einsum("i,nij,j->n", x0, inv_stack, x0)
            variance = sigma2 * (1.0 + quadratic_form)
        else:
            if self._se_cache.get("mode") != "nonlinear":
                return series
            valid_index = self._se_cache["valid_index"]
            matrices = self._se_cache["pcov_list"]
            variance = []
            variance_index = []
            for idx, pcov in zip(valid_index, matrices):
                row = self.fitting_results.loc[idx]
                try:
                    resolved = self._resolve_reference_input(i)
                    if resolved is None:
                        continue
                    target_i, tref, ohref, _ = resolved
                    I_s = self.scaler_I.scale(target_i)
                    IxT_s = self.scaler_IxT.scale(tref * target_i)
                    logH_s = self.scaler_log_h.scale(np.log(ohref))
                    c4_val = row.get("c4", 0)
                    c6_val = row.get("c6", 0)
                    if pd.isna(c4_val):
                        c4_val = 0
                    if pd.isna(c6_val):
                        c6_val = 0

                    if self.nonlinear_term == "cbrt":
                        diff = target_i - c6_val
                        d_c4 = np.cbrt(diff)
                        if abs(diff) > 1e-10:
                            d_c6 = -c4_val / (3.0 * np.cbrt(diff**2))
                        else:
                            d_c6 = 0.0
                    else:
                        sig = 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (target_i - c6_val)))
                        d_c4 = sig
                        d_c6 = -c4_val * self.sigmoid_steepness * sig * (1.0 - sig)

                    grad = np.array([I_s, IxT_s, logH_s, d_c4, 1.0, d_c6], dtype=float)
                    variance.append(float(grad @ pcov @ grad.T))
                    variance_index.append(idx)
                except Exception:
                    continue

            if not variance:
                return series
            variance = np.asarray(variance, dtype=float)
            valid_index = variance_index

        scale = self.scaler_U.max - self.scaler_U.min
        se = np.where(variance > 0, np.sqrt(variance) * scale, np.nan)
        series.loc[valid_index] = se
        return series

    def Uref_se_calc(self, row, i):
        """Calculate prediction interval SE."""
        XTX = self._coerce_xtx_matrix(row.get("XTX"), size=5 if self.nonlinear_term == "I2" else self.n_params)
        if XTX is None:
            return np.nan

        if self.nonlinear_term == "I2":
            # Original linear approach
            x0 = self._get_reference_vector(i)

            sigma2 = row["sigma2"]
            variance = sigma2 * (1 + x0.T @ np.linalg.inv(XTX) @ x0)
            
            if variance > 0:
                Uref_scaled_se = np.sqrt(variance)
                return Uref_scaled_se * (self.scaler_U.max - self.scaler_U.min)
            return np.nan
        
        else:
            # For nonlinear models: use pcov + gradient propagation
            try:
                pcov = XTX
                resolved = self._resolve_reference_input(i)
                if resolved is None:
                    return np.nan
                target_i, tref, ohref, _ = resolved

                I_s = self.scaler_I.scale(target_i)
                IxT_s = self.scaler_IxT.scale(tref * target_i)
                logH_s = self.scaler_log_h.scale(np.log(ohref))
                
                c4_val = row.get("c4", 0)
                c6_val = row.get("c6", 0)
                if pd.isna(c4_val): c4_val = 0
                if pd.isna(c6_val): c6_val = 0
                
                if self.nonlinear_term == "cbrt":
                    diff = target_i - c6_val
                    d_c4 = np.cbrt(diff)
                    if abs(diff) > 1e-10:
                        d_c6 = -c4_val / (3.0 * np.cbrt(diff**2))
                    else:
                        d_c6 = 0.0
                elif self.nonlinear_term == "sigmoid":
                    k = self.sigmoid_steepness
                    sig = 1.0 / (1.0 + np.exp(-k * (target_i - c6_val)))
                    d_c4 = sig
                    d_c6 = -c4_val * k * sig * (1 - sig)
                
                # Gradient: [d/dc1, d/dc2, d/dc3, d/dc4, d/dc5, d/dc6]
                grad = np.array([I_s, IxT_s, logH_s, d_c4, 1.0, d_c6])
                
                variance = grad @ pcov @ grad
                
                if variance > 0:
                    Uref_scaled_se = np.sqrt(variance)
                    return Uref_scaled_se * (self.scaler_U.max - self.scaler_U.min)
                return np.nan
            except Exception:
                return np.nan

    def back_calculate_coefficients(self):
        """Convert scaled coefficients back to physical units."""
        fr = self.fitting_results
        max_U = self.max_U

        def unscale_coeff(val, dim_max, dim_min):
            return val * max_U / (dim_max - dim_min)

        fr["back_calc_c1"] = unscale_coeff(fr.c1, self.max_I, self.min_I)
        fr["back_calc_c2"] = unscale_coeff(fr.c2, self.max_IxT, self.min_IxT)
        fr["back_calc_c3"] = unscale_coeff(fr.c3, self.max_log_h, self.min_log_h)
        
        if self.nonlinear_term == "I2":
            fr["back_calc_c4"] = unscale_coeff(fr.c4, self.max_I**2, self.min_I**2)
            fr["back_calc_c5"] = max_U * (fr.c5 
                - fr.c1 / (self.max_I - self.min_I) * self.min_I
                - fr.c2 / (self.max_IxT - self.min_IxT) * self.min_IxT
                - fr.c3 / (self.max_log_h - self.min_log_h) * self.min_log_h
                - fr.c4 / (self.max_I**2 - self.min_I**2) * (self.min_I**2))
        else:
            # For cbrt/sigmoid: c4 operates on raw I, no I2 scaler
            fr["back_calc_c4"] = fr.c4 * max_U
            fr["back_calc_c6"] = fr.c6  # c6 is already in raw I units
            fr["back_calc_c5"] = max_U * (fr.c5 
                - fr.c1 / (self.max_I - self.min_I) * self.min_I
                - fr.c2 / (self.max_IxT - self.min_IxT) * self.min_IxT
                - fr.c3 / (self.max_log_h - self.min_log_h) * self.min_log_h)
        
        return fr
    
    def check_vertex(self, row):
        """
        Check if the UI curve vertex is within physical range.
        For I2: vertex at I = -(c1 + c2*T) / (2*c4)
        For cbrt/sigmoid: no simple vertex, skip this check.
        """
        if self.nonlinear_term != "I2":
            return row.get("quality", "good")
        
        if pd.isna(row.get("back_calc_c4")): 
            return "failed"
        
        if row.get("status") and "High Cond" in str(row.get("status", "")):
            return row["quality"]

        c1, c2, c4 = row["back_calc_c1"], row["back_calc_c2"], row["back_calc_c4"]
        if c4 == 0: return "linear"
        vertex_I = - (c1 + c2 * self.Tref) / (2 * c4)
        
        if 0 < vertex_I < 2.5:
            return "Vertex of UI curve is between 0 and 2.5A/cm2."
        return "good"

    def get_Urc_results(self, i):
        """Return reliable Urc results in the comparator-compatible format."""
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
            elif df.index.name is not None:
                df = df.reset_index()

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
        if not self.Iref:
            return None

        resolved = self._resolve_reference_input(i)
        if resolved is None:
            return None
        closest_i, _, _, ref_name = resolved

        urc_results = self.get_Urc_results(closest_i)
        if urc_results is None or urc_results.empty:
            return None

        urc_results_excl_run_in = urc_results.query(f"calh > {self.run_in}")
        if len(urc_results_excl_run_in) < 2:
            return None

        try:
            result = calculate_degradation_rate_and_uncertainty(
                urc_results_excl_run_in["calh"],
                urc_results_excl_run_in["Urc"],
                urc_results_excl_run_in["Urc_se"],
                self.fix_point,
                self.regression_method,
            )
        except Exception:
            return None

        self.degradation_results[closest_i] = degradation_result(
            result.aging_uV_per_h,
            result.bol_V,
            result.uncertainty_uV2,
        )
        self.degradation_results_Urc1[closest_i] = self.degradation_results[closest_i]
        self.degradation_results_by_ref[ref_name] = self.degradation_results[closest_i]
        return self.degradation_results[closest_i]

    def plot_Urc1_separately(self, i, save=True, output_dir="plots"):
        """Plot Urc points with uncertainty and optional degradation trend line."""
        closest_i = min(self.Iref, key=lambda x: abs(x - i))
        urc_results = self.get_Urc_results(closest_i)
        if urc_results is None or urc_results.empty:
            print(f"{self.name}: no Urc results to plot at {closest_i}A/cm2.")
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(
            urc_results["calh"],
            urc_results["Urc"],
            yerr=urc_results["Urc_se"],
            linestyle="",
            marker=".",
            markersize=4,
            color="tab:blue",
            ecolor="tab:blue",
            alpha=0.8,
            label=f"Urc @ {closest_i}A/cm2",
        )

        deg_res = self.degradation_results.get(closest_i)
        if deg_res is not None:
            x_line = np.array([urc_results["calh"].min(), urc_results["calh"].max()], dtype=float)
            y_line = deg_res.bol_V + (deg_res.aging_uV_per_h / 1e6) * x_line
            ax.plot(
                x_line,
                y_line,
                linestyle="--",
                color="black",
                label=f"Fit: {deg_res.aging_uV_per_h:.2f} uV/h",
            )

        ax.set_xlabel("Calendar hours since test start [h]")
        ax.set_ylabel("Urc [V]")
        ax.set_title(f"{self.name} - Urc Trend ({self.nonlinear_term}, {closest_i}A/cm2)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        if save:
            os.makedirs(output_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{self.name}_Urc_{self.nonlinear_term}_{closest_i}A_{ts}.png"
            full_path = os.path.join(output_dir, file_name)
            fig.savefig(full_path, dpi=150)
            print(f"Saved: {full_path}")

        return fig

    def plot_residual_diagnostics(self, i, save=True, output_dir="plots"):
        """Plot residual diagnostics against fitted degradation line for a target reference current."""
        closest_i = min(self.Iref, key=lambda x: abs(x - i))
        urc_results = self.get_Urc_results(closest_i)
        deg_res = self.degradation_results.get(closest_i)

        if urc_results is None or urc_results.empty or deg_res is None:
            print(f"{self.name}: insufficient data for residual diagnostics at {closest_i}A/cm2.")
            return None

        y_pred = deg_res.bol_V + (deg_res.aging_uV_per_h / 1e6) * urc_results["calh"].to_numpy(dtype=float)
        residual_mv = (urc_results["Urc"].to_numpy(dtype=float) - y_pred) * 1000.0

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].scatter(urc_results["calh"], residual_mv, s=12, alpha=0.75, color="tab:orange")
        axes[0].axhline(0.0, linestyle="--", color="black", linewidth=1)
        axes[0].set_xlabel("Calendar hours since test start [h]")
        axes[0].set_ylabel("Residual [mV]")
        axes[0].set_title("Residual vs Time")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(residual_mv, bins=20, color="tab:green", alpha=0.8)
        axes[1].axvline(np.mean(residual_mv), linestyle="--", color="black", linewidth=1,
                        label=f"mean={np.mean(residual_mv):.2f} mV")
        axes[1].set_xlabel("Residual [mV]")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Residual Distribution")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.suptitle(f"{self.name} - Residual Diagnostics ({self.nonlinear_term}, {closest_i}A/cm2)")
        fig.tight_layout()

        if save:
            os.makedirs(output_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{self.name}_ResidualDiag_{self.nonlinear_term}_{closest_i}A_{ts}.png"
            full_path = os.path.join(output_dir, file_name)
            fig.savefig(full_path, dpi=150)
            print(f"Saved: {full_path}")

        return fig