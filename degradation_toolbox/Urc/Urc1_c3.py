"""Urc1_c3: Urc1 variant with voltage-drop-based dynamic intervals.

Key modifications compared to Urc1 (baseline):

1. ln(OH) variable change:
   - Urc1:    OH = hours since last operational start
   - Urc1_c3: OH = hours since voltage drops below a configurable
              threshold (u_drop_threshold)

2. Dynamic intervals:
   - Urc1:    fixed-length sliding windows (len_interval, slide)
   - Urc1_c3: intervals auto-detected by voltage-drop events.
              Each interval starts when voltage drops below
              u_drop_threshold and ends at the next such event.

3. Interval diagnostics:
   - Records each interval's start_time, end_time, duration, and
     the number of valid (post-filter) data points (n_valid_points).
"""

import datetime
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

from degradation_toolbox.Urc.Urc1 import MinMaxScalerCustomize, OCV, Urc1, func

pd.options.mode.copy_on_write = True


def plot_Urc1_c3(Urc1_c3_multi_cells, plot_name="Urc1_c3 (Multi cells)"):
    """
    Helper function to plot multiple Urc1_c3 instances.
    """
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.figure()
    for cell in Urc1_c3_multi_cells:
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

class Urc1_c3:
    """
    Calculate voltage under reference condition (Urc) using operation data.

    Compared to Urc1 (baseline), this variant:
      1. Uses "hours since voltage drops below u_drop_threshold" as the log-term
         variable instead of "hours since last operational start".
      2. Automatically detects fitting intervals from voltage-drop events
         (no fixed len_interval / slide).
      3. Records interval_start, interval_end, interval_duration_h, and
         n_valid_points for every interval.

    Parameters
    ----------
    data : pd.DataFrame
        Must have datetime index ("timestamp") and columns:
        "currentDensity" [A/cm2], "temperature" [deg C], "voltage" [V].
    name : str
        Identifier for this cell/stack.
    Iref : float or list of float
        Reference current density [A/cm2] for Urc calculation.
    Tref : float
        Reference temperature [deg C].
    OHref : float
        Reference hours since voltage drop for Urc calculation [h].
    i_off : float
        Current density threshold below which electrolyzer is considered off [A/cm2].
    u_off : float
        Voltage threshold below which electrolyzer is considered off [V].
    u_drop_threshold : float
        **NEW** - Voltage threshold [V] for cycle boundary detection.
        A new interval begins each time voltage transitions from
        >= u_drop_threshold to < u_drop_threshold.  Default 1.6 V.
    resample : int
        Keep every n-th data point (no averaging).
    min_num_data_required_for_fit : int
        Minimum valid data points in an interval to attempt fitting.
    threshold : float
        Maximum acceptable condition number of the covariance matrix.
    data_filter_* : float
        Post-preprocessing filters for I, U, T, and h_since_u_drop.
    data_filter_h_since_u_drop_min : float
        **NEW** - Minimum hours since voltage drop to include a point.
        Avoids log(0) and initial transient.  Default 0.01 h.
    plot_fit : bool / int
        Number of fit-diagnostic plots to generate (0 = none).
    """

    def __init__(self,
                 data,
                 name="Elyzer",
                 Iref=[0.6, 1.5],
                 Tref=60,
                 OHref=3 * 24,
                 ref_config=None,
                 i_off=0.1,
                 u_off=1.3,
                 u_drop_threshold=1.65,
                 resample=1,
                 min_num_data_required_for_fit=100,
                 threshold=10 ** 6,
                 data_filter_i_min=0.1,
                 data_filter_U_min=1.4,
                 data_filter_U_max=2.3,
                 data_filter_T_min=50,
                 data_filter_T_max=65,
                 data_filter_h_since_u_drop_min=0.01,
                 plot_fit=True,
                 preprocessed_data=None,
                 **kwargs):

        # 1. Parameter Setup
        self._preprocessed_data_provided = preprocessed_data is not None
        self.data = data[["currentDensity", "temperature", "voltage"]]
        self.name = name
        self.ref_config = Urc1._normalize_ref_config(ref_config, Iref, Tref, OHref)
        self.ref_name_to_i = {
            ref_name: float(cfg["Iref"]) for ref_name, cfg in self.ref_config.items()
        }
        self.Iref = [float(cfg["Iref"]) for cfg in self.ref_config.values()]
        self.Tref = float(next(iter(self.ref_config.values()))["Tref"])
        self.OHref = float(next(iter(self.ref_config.values()))["OHref"])
        self.i_off = i_off
        self.u_off = u_off
        self.u_drop_threshold = u_drop_threshold
        self.resample = resample
        self.min_num_data_required_for_fit = min_num_data_required_for_fit
        self.threshold = threshold
        self.data_filter_i_min = data_filter_i_min
        self.data_filter_U_min = data_filter_U_min
        self.data_filter_U_max = data_filter_U_max
        self.data_filter_T_min = data_filter_T_min
        self.data_filter_T_max = data_filter_T_max
        self.data_filter_h_since_u_drop_min = data_filter_h_since_u_drop_min
        self.plot_fit = plot_fit
        self._se_cache = None

        # Result containers
        self.fitting_results = pd.DataFrame()
        self.intervals = pd.DataFrame()          # detected intervals

        # 2. Data Preprocessing (includes voltage-drop detection)
        #    NOTE: Urc1_c3 has custom preprocess (voltage-drop interval detection).
        #    preprocessed_data from Urc1.preprocess_once() does NOT include
        #    h_since_u_drop / interval_id, so c3 always runs its own preprocess.
        if preprocessed_data is not None:
            print("[WARNING] Urc1_c3 ignores preprocessed_data "
                  "(needs voltage-drop detection). Running own preprocess.")
        print("Data preprocessing ...")
        self.data = self.data_preprocess()

        if not self.data.empty:
            self.installation_time = self.data.index[0]
        else:
            raise ValueError("Data is empty after preprocessing.")

        print("Voltage model fitting ...")

        # 3. Initialize Scalers
        self._init_scalers()

        # 4. Model Fitting (auto-detected intervals)
        self.fitting_results = self.model_fitting()

        # 5. Post-Calculation (Urc & quality checks)
        self.fitting_results = self.Urc_calc()
        self.fitting_results = self.back_calculate_coefficients()
        self.fitting_results["quality"] = "good"
        self.fitting_results["quality"] = self.fitting_results.apply(self.check_vertex, axis=1)

        # 6. Filter Reliable Results
        if "cond" in self.fitting_results.columns:
            self.fitting_results_reliable = self.fitting_results.query(
                f"cond < {self.threshold} and quality=='good'").reset_index()
        else:
            self.fitting_results_reliable = pd.DataFrame()

        print(f"{len(self.fitting_results_reliable)} out of {len(self.fitting_results)} fitting results are reliable.")

    # ================================================================== #
    #                    Interval Detection (NEW)                         #
    # ================================================================== #

    def _detect_u_drop_events(self, data):
        """
        Find timestamps where voltage transitions from >= threshold to < threshold.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain a "voltage" column; indexed by datetime.

        Returns
        -------
        drop_times : list of pd.Timestamp
        """
        voltage = data["voltage"]
        below = voltage < self.u_drop_threshold

        # Transition: previous point was at-or-above threshold, current is below
        transitions = below & (~below.shift(1, fill_value=False))
        drop_times = data.index[transitions].tolist()
        return drop_times

    def _build_intervals(self, data, drop_times):
        """
        Build an interval table from voltage-drop timestamps.

        Each interval runs from one drop event to the next.
        The last interval extends to the end of the data.

        Returns
        -------
        intervals : pd.DataFrame
            Columns: start_time, end_time, interval_h, n_valid_points (initialised to 0).
        """
        if len(drop_times) == 0:
            raise ValueError(
                f"No voltage drop below {self.u_drop_threshold}V found in data. "
                f"Consider adjusting u_drop_threshold."
            )

        rows = []
        for i in range(len(drop_times)):
            start = drop_times[i]
            end = drop_times[i + 1] if i + 1 < len(drop_times) else data.index[-1]
            rows.append({
                "start_time": start,
                "end_time": end,
                "interval_h": (end - start) / np.timedelta64(1, "h"),
                "n_valid_points": 0,          # filled later after filtering
            })

        return pd.DataFrame(rows)

    def _calc_h_since_u_drop(self, data, intervals):
        """
        For every data point, compute hours since the most recent voltage-drop event
        and tag it with the interval index.

        Adds columns
        -------------
        h_since_u_drop : float   - hours since interval start
        interval_id    : int     - index into self.intervals (-1 = unassigned)
        """
        data["h_since_u_drop"] = np.nan
        data["interval_id"] = -1

        for idx, row in intervals.iterrows():
            start = row["start_time"]
            end = row["end_time"]

            # Last interval is closed on both sides; others are half-open [start, end)
            if idx < len(intervals) - 1:
                mask = (data.index >= start) & (data.index < end)
            else:
                mask = (data.index >= start) & (data.index <= end)

            data.loc[mask, "h_since_u_drop"] = (
                data.index[mask] - start) / np.timedelta64(1, "h")
            data.loc[mask, "interval_id"] = idx

        return data

    # ================================================================== #
    #                        Data Preprocessing                           #
    # ================================================================== #

    def data_preprocess(self):
        len_data_before = len(self.data)

        # --- basic cleaning (same as Urc1) ---
        cols = list(self.data.select_dtypes(include=["float64"]))
        self.data[cols] = self.data[cols].astype("float32")

        self.data = self.data[~self.data.index.duplicated(keep="first")]
        self.data = self.data.sort_index()
        self.data = self.data[::self.resample]
        self.data.index.name = "timestamp"

        # --- KEY CHANGE: detect voltage-drop events BEFORE filtering ---
        drop_times = self._detect_u_drop_events(self.data)
        self.intervals = self._build_intervals(self.data, drop_times)
        print(f"Detected {len(self.intervals)} intervals "
              f"(voltage dropping below {self.u_drop_threshold}V).")

        # Calculate h_since_u_drop for every data point
        self.data = self._calc_h_since_u_drop(self.data, self.intervals)

        # --- Filtering ---
        self.data = self.data.dropna(
            subset=["currentDensity", "temperature", "voltage", "h_since_u_drop"]
        )
        self.data = self.data.query(
            f"{self.data_filter_i_min} < currentDensity and "
            f"{self.data_filter_U_min} < voltage < {self.data_filter_U_max} and "
            f"{self.data_filter_T_min} < temperature < {self.data_filter_T_max} and "
            f"h_since_u_drop > {self.data_filter_h_since_u_drop_min}"
        )

        if self.data.empty:
            raise ValueError("No data is left after filtering.")

        # Update n_valid_points per interval (after filtering)
        valid_counts = self.data.groupby("interval_id").size()
        for interval_id, count in valid_counts.items():
            if 0 <= interval_id < len(self.intervals):
                self.intervals.loc[interval_id, "n_valid_points"] = int(count)

        len_data_after = len(self.data)
        print(f"Before preprocess: {len_data_before} -> After: {len_data_after} points.")
        print(f"  (Filters: i>{self.data_filter_i_min} A/cm2, "
              f"{self.data_filter_U_min} V<U<{self.data_filter_U_max} V, "
              f"{self.data_filter_T_min} C<T<{self.data_filter_T_max} C, "
              f"h_since_u_drop>{self.data_filter_h_since_u_drop_min} h)")
        return self.data

    def save_preprocessed_data(self, output_dir: str = "../explore_data/output"):
        """Save preprocessed data to a Parquet file.

        Args:
            output_dir: Output directory path.
        """
        if self.data is None:
            return
        print("\n=== Saving preprocessed data ===")
        df_to_save = self.data.copy().reset_index()
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"Preprocessed_data_{self.name}_{current_time}.parquet"
        df_to_save.to_parquet(os.path.join(output_dir, file_name))

    # ================================================================== #
    #                    Scalers & Feature Engineering                     #
    # ================================================================== #

    def _init_scalers(self):
        """Initialize min-max scalers for features (same ranges as Urc1)."""
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

    def _prepare_matrices(self, df):
        """
        Feature engineering & scaling.

        KEY CHANGE vs Urc1:
            log_h = ln(h_since_u_drop)   instead of   ln(h_since_last_start)
        """
        df = df.copy()
        df["U_ocv"] = OCV(df.temperature)
        df["IxT"] = df.currentDensity * df.temperature
        df["log_h"] = np.log(df.h_since_u_drop)          # <-- changed variable
        df["I2"] = df.currentDensity ** 2

        X = np.column_stack((
            self.scaler_I.scale(df.currentDensity),
            self.scaler_IxT.scale(df.IxT),
            self.scaler_log_h.scale(df.log_h),
            self.scaler_I2.scale(df.I2),
        ))
        y = self.scaler_U.scale(df.voltage - df["U_ocv"])

        return X, y

    # ================================================================== #
    #                          Model Fitting                              #
    # ================================================================== #

    def model_fitting(self):
        """
        Fit the voltage model for each auto-detected interval.

        KEY CHANGE vs Urc1:
            Loops over dynamically detected intervals (self.intervals)
            instead of fixed-length sliding windows.

        Extra columns in results:
            interval_end, interval_duration_h, n_valid_points
        """
        # --- Build output DataFrame indexed by (interval_start, cell_name) ---
        interval_starts = self.intervals["start_time"].tolist()
        index = pd.MultiIndex.from_product(
            [interval_starts, [self.name]], names=["date", "name"]
        )
        results = pd.DataFrame(index=index)

        # Numerical columns
        num_cols = [
            "day_since_install", "calh",
            "R2", "RMSE", "cond", "sigma2",
            "c1", "c2", "c3", "c4", "c5",
            "c1_se", "c2_se", "c3_se", "c4_se", "c5_se",
            "interval_duration_h", "n_valid_points",
        ]
        for c in num_cols:
            results[c] = np.nan

        # Object columns
        obj_cols = ["XTX", "quality", "status", "interval_end"]
        for c in obj_cols:
            results[c] = pd.Series([None] * len(results), index=index, dtype=object)

        num_interv_without_enough_data = 0
        num_interv_curve_fit_failed = 0
        num_plot_fit = 1
        grouped_data = {int(k): v for k, v in self.data.groupby("interval_id", sort=False)}

        for interval_idx, interval_row in self.intervals.iterrows():
            start_time = interval_row["start_time"]
            end_time = interval_row["end_time"]
            interval_h = interval_row["interval_h"]
            n_valid = int(interval_row["n_valid_points"])

            idx = (start_time, self.name)

            # --- Record interval meta-data ---
            results.loc[idx, "day_since_install"] = (
                start_time - self.installation_time).days
            results.loc[idx, "calh"] = (
                start_time - self.installation_time) / np.timedelta64(1, "h")
            results.loc[idx, "interval_end"] = str(end_time)
            results.loc[idx, "interval_duration_h"] = interval_h
            results.loc[idx, "n_valid_points"] = n_valid

            # --- Slice filtered data for this interval ---
            data_this_interval = grouped_data.get(int(interval_idx), self.data.iloc[0:0])

            if n_valid < self.min_num_data_required_for_fit:
                num_interv_without_enough_data += 1
                results.loc[idx, "status"] = (
                    f"Insufficient Data ({n_valid} < {self.min_num_data_required_for_fit})")
                continue

            try:
                # Prepare matrices (uses h_since_u_drop for log term)
                X, y = self._prepare_matrices(data_this_interval)

                # Fit (OLS with bounds, same as Urc1 baseline)
                popt, pcov = curve_fit(
                    func, X, y,
                    bounds=((0, -np.inf, -np.inf, -np.inf, -np.inf),
                            (np.inf, 0, np.inf, np.inf, np.inf))
                )
                condition_number = np.linalg.cond(pcov)

                # --- Goodness of fit (on unscaled voltage) ---
                U_predicted = (self.scaler_U.unscale(func(X, *popt))
                               + OCV(data_this_interval.temperature))
                U_true = data_this_interval.voltage

                results.loc[idx, "R2"] = r2_score(U_true, U_predicted)
                results.loc[idx, "RMSE"] = np.sqrt(mean_squared_error(U_true, U_predicted))
                results.loc[idx, "cond"] = condition_number

                # Coefficients & standard errors
                for k in range(5):
                    results.loc[idx, f"c{k+1}"] = popt[k]
                    results.loc[idx, f"c{k+1}_se"] = np.sqrt(np.diag(pcov)[k])

                # SE helper quantities (saved for Uref_se_calc)
                X_with_constant = np.c_[X, np.ones(len(X))]
                XTX = X_with_constant.T @ X_with_constant
                results.at[idx, "XTX"] = XTX
                results.loc[idx, "sigma2"] = (
                    np.sum((y - func(X, *popt)) ** 2) /
                    (len(y) - X_with_constant.shape[1])
                )

                # Optional plotting counter
                if condition_number < self.threshold and num_plot_fit <= self.plot_fit:
                    num_plot_fit += 1

                if condition_number >= self.threshold:
                    results.loc[idx, "status"] = f"High Cond ({condition_number:.1e})"
                else:
                    results.loc[idx, "status"] = "Fit Success"

            except Exception as e:
                num_interv_curve_fit_failed += 1
                results.loc[idx, "status"] = f"Fit Error: {str(e)}"

        print(f"Fitting Stats: {num_interv_without_enough_data} intervals low data, "
              f"{num_interv_curve_fit_failed} fit failed.")
        self._se_cache = None
        return results

    # ================================================================== #
    #                       Urc Calculation                               #
    # ================================================================== #

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

    def _resolve_reference_input(self, target):
        return Urc1._resolve_reference_input(self, target)

    def get_reference_vector(self, target):
        resolved = self._resolve_reference_input(target)
        if resolved is None:
            raise ValueError(f"Unknown reference target: {target}")

        closest_i, tref, ohref, _ = resolved
        return np.array([
            self.scaler_I.scale(closest_i),
            self.scaler_IxT.scale(tref * closest_i),
            self.scaler_log_h.scale(np.log(ohref)),
            self.scaler_I2.scale(closest_i ** 2),
            1.0,
        ], dtype=float)

    def _coerce_xtx_matrix(self, xtx_value, size=5):
        if xtx_value is None:
            return None

        if isinstance(xtx_value, np.ndarray):
            xtx = np.asarray(xtx_value, dtype=float)
        elif isinstance(xtx_value, (list, tuple)):
            xtx = np.asarray(xtx_value, dtype=float)
        elif isinstance(xtx_value, str):
            xtx = xtx_value.replace("[", "").replace("]", "")
            xtx = np.fromstring(xtx, sep=",")
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

        x0 = self.get_reference_vector(i)

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
        XTX = self._coerce_xtx_matrix(row.get("XTX"), size=5)
        if XTX is None:
            return np.nan

        x0 = self.get_reference_vector(i)

        sigma2 = row["sigma2"]
        variance = sigma2 * (1 + x0.T @ inv(XTX) @ x0)

        if variance > 0:
            return np.sqrt(variance) * (self.scaler_U.max - self.scaler_U.min)
        return np.nan

    # ================================================================== #
    #              Coefficient Back-Calculation & Quality Check            #
    # ================================================================== #

    def back_calculate_coefficients(self):
        """Convert scaled coefficients back to physical units."""
        fr = self.fitting_results
        max_U = self.max_U

        def unscale_coeff(val, dim_max, dim_min):
            return val * max_U / (dim_max - dim_min)

        fr["back_calc_c1"] = unscale_coeff(fr.c1, self.max_I, self.min_I)
        fr["back_calc_c2"] = unscale_coeff(fr.c2, self.max_IxT, self.min_IxT)
        fr["back_calc_c3"] = unscale_coeff(fr.c3, self.max_log_h, self.min_log_h)
        fr["back_calc_c4"] = unscale_coeff(fr.c4, self.max_I ** 2, self.min_I ** 2)

        fr["back_calc_c5"] = max_U * (
            fr.c5
            - fr.c1 / (self.max_I - self.min_I) * self.min_I
            - fr.c2 / (self.max_IxT - self.min_IxT) * self.min_IxT
            - fr.c3 / (self.max_log_h - self.min_log_h) * self.min_log_h
            - fr.c4 / (self.max_I ** 2 - self.min_I ** 2) * (self.min_I ** 2)
        )
        return fr

    def check_vertex(self, row):
        """Check if the UI curve vertex is within physical range."""
        if pd.isna(row["back_calc_c4"]):
            return "failed"

        if row["status"] and "High Cond" in str(row["status"]):
            return row["quality"]

        c1, c2, c4 = row["back_calc_c1"], row["back_calc_c2"], row["back_calc_c4"]
        if c4 == 0:
            return "linear"
        vertex_I = -(c1 + c2 * self.Tref) / (2 * c4)

        if 0 < vertex_I < 2.5:
            return "Vertex of UI curve is between 0 and 2.5A/cm2."
        return "good"

    # ================================================================== #
    #                       Convenience / Diagnostics                      #
    # ================================================================== #

    def get_intervals_summary(self):
        """
        Return a copy of the intervals table enriched with fitting status.

        Useful for inspecting which intervals were fitted successfully,
        which had insufficient data, etc.
        """
        summary = self.intervals.copy()
        summary["status"] = None
        summary["quality"] = None
        summary["R2"] = np.nan

        for idx, row in summary.iterrows():
            fit_key = (row["start_time"], self.name)
            if fit_key in self.fitting_results.index:
                summary.loc[idx, "status"] = self.fitting_results.loc[fit_key, "status"]
                summary.loc[idx, "quality"] = self.fitting_results.loc[fit_key, "quality"]
                summary.loc[idx, "R2"] = self.fitting_results.loc[fit_key, "R2"]
        return summary
