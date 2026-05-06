import datetime
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
from plotly.subplots import make_subplots

from degradation_toolbox.Urc.Urc1 import Urc1
from degradation_toolbox.Urc.Urc1_adaptive import Urc1_Adaptive
from degradation_toolbox.Urc.helpers import (
    mylinregress,
    calculate_degradation_rate_and_uncertainty,
    degradation_result,
)
from degradation_toolbox.utils.data_helpers import example_data, validate_columns, fill_gaps
from degradation_toolbox.utils.calculation_basics import pressure_correction


logger = logging.getLogger(__name__)


class Urc:
    def __init__(self,
                 data,
                 # --- Basic Parameters ---
                 resample=1,
                 run_in=0,
                 i_off=0.1,
                 u_off=1.3,
                 min_num_data_required_for_fit=400,
                 
                 # --- Regression Parameters ---
                 regression_method="orth",
                 fix_point=None,
                 iteration=1,
                 
                 # --- Feature Switches ---
                 plot_fit=0,
                 method="baseline",  # Core switch: "baseline" or "adaptive"
                 threshold=1e6,      # Explicitly received threshold
                 
                 # --- [Key] Accept all additional parameters (decay_rate, urc_min, etc.) ---
                 **kwargs 
                 ):
        """
        Calculate voltage under reference condition (Urc) using operation data.
        Acts as a wrapper/scheduler for Urc1 (Baseline) and Urc1_Adaptive.
        """
        
        # 1. Pressure correction
        self.need_correct_pressure = kwargs.get('need_correct_pressure', False)
        if self.need_correct_pressure:
            if 'col_pressure' not in kwargs or 'pressure_correction_target_bar' not in kwargs:
                raise KeyError("You set 'need_correct_pressure' to True. Please provide 'col_pressure' and 'pressure_correction_target_bar'.")
            self.col_pressure = kwargs["col_pressure"]
            self.pressure_correction_target_bar = kwargs["pressure_correction_target_bar"]
            data = self._correct_pressure(data)

        # 2. Data validation and gap filling
        required_columns = ['timestamp', 'currentDensity', 'temperature', 'voltage']
        validate_columns(data, required_columns)

        # Keep only essential columns and fill gaps
        data = data[["currentDensity", "temperature", "voltage"]].copy(deep=True)
        data = fill_gaps(data)

        # 3. Parameter assignment
        self.iteration = iteration
        self.warning = ""

        self.resample = resample
        self.i_off = i_off
        self.u_off = u_off
        self.min_num_data_required_for_fit = min_num_data_required_for_fit
        self.plot_fit = plot_fit
        self.threshold = threshold  # Save threshold

        self.regression_method = regression_method
        self.fix_point = fix_point
        self.run_in = run_in
        self.method = method

        # Urc1-related parameters (with defaults)
        self.name = kwargs.get('name', 'Elyzer')
        self.Iref = kwargs.get('Iref', [0.6, 1.5])
        if not isinstance(self.Iref, list):
            self.Iref = [self.Iref]
        self.Tref = kwargs.get('Tref', 60)
        self.OHref = kwargs.get('OHref', 3 * 24)
        self.len_interval = kwargs.get('len_interval', 2)
        self.slide = kwargs.get('slide', 1)
        
        # Filter parameters
        self.data_filter_i_min = kwargs.get('data_filter_i_min', 0.1)
        self.data_filter_U_min = kwargs.get('data_filter_U_min', 1.4)
        self.data_filter_U_max = kwargs.get('data_filter_U_max', 2.3)
        self.data_filter_T_min = kwargs.get('data_filter_T_min', 50)
        self.data_filter_T_max = kwargs.get('data_filter_T_max', 65)
        self.data_filter_h_since_last_start_min = kwargs.get('data_filter_h_since_last_start_min', 0.5)

        # Save extra kwargs (decay_rate, urc_min, etc.) for forwarding to subclasses
        self.extra_kwargs = kwargs

        # Result containers
        self.degradation_results_all_iterations = list()
        self.degradation_results_Urc1_all_iterations = list()
        self.fitting_runtime_seconds = 0.0
        self.fitting_runtime_seconds_per_iteration = []
        self.timing_metrics = {}

        # ==========================================
        # Core Iteration Loop
        # ==========================================
        for iter_idx in range(iteration):
            try:
                logger.info(
                    ">>>>> %s, iteration %s: %s calculation starts <<<<<",
                    self.name,
                    iter_idx + 1,
                    self.method,
                )
                
                # [Key step] Build common parameter dict
                # Map Urc parameter names to Urc1/Adaptive parameter names
                common_params = {
                    "name": self.name,
                    "Iref": self.Iref, "Tref": self.Tref, "OHref": self.OHref,
                    "i_off": self.i_off, "u_off": self.u_off,
                    "resample": self.resample,
                    "len_interval": self.len_interval, "slide": self.slide,
                    
                    # [Mapping]: Urc(for_fit) -> Urc1(per_interval)
                    "min_num_data_required_for_fit": self.min_num_data_required_for_fit, 
                    
                    # Explicitly forward threshold
                    "threshold": self.threshold,
                    
                    # Forward filter parameters
                    "data_filter_i_min": self.data_filter_i_min,
                    "data_filter_U_min": self.data_filter_U_min,
                    "data_filter_U_max": self.data_filter_U_max,
                    "data_filter_T_min": self.data_filter_T_min,
                    "data_filter_T_max": self.data_filter_T_max,
                    "data_filter_h_since_last_start_min": self.data_filter_h_since_last_start_min,
                    "plot_fit": self.plot_fit
                }

                filtered_kwargs = {k: v for k, v in self.extra_kwargs.items() if k not in common_params}

                # [Instantiation logic]
                if self.method == "baseline":
                    # Baseline class receives common_params and filtered_kwargs.
                    # Urc1.__init__ accepts **kwargs, so unknown adaptive params are ignored.
                    Urc1_instance = Urc1(data, **common_params, **filtered_kwargs)
                    
                elif self.method == "adaptive":
                    # Adaptive class extracts its own params (decay_rate, etc.) from kwargs.
                    Urc1_instance = Urc1_Adaptive(data, **common_params, **filtered_kwargs)
                    
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                
                self.urc_instance = Urc1_instance  # Save instance for later access
                iteration_runtime = getattr(Urc1_instance, 'fitting_runtime_seconds', np.nan)
                self.fitting_runtime_seconds_per_iteration.append(iteration_runtime)
                if np.isfinite(iteration_runtime):
                    self.fitting_runtime_seconds += float(iteration_runtime)

                logger.info(
                    ">>>>> %s, iteration %s: Calculation finished <<<<<",
                    self.name,
                    iter_idx + 1,
                )

                # Save preprocessed data on first iteration
                if iter_idx == 0:
                    Urc1_instance.save_preprocessed_data()

                # Extract and save results
                self.Urc1_results = Urc1_instance.fitting_results_reliable
                self.Urc1_results_incl_unreliable = Urc1_instance.fitting_results
                self.installation_time = Urc1_instance.installation_time

            except Exception as e:
                import traceback
                traceback.print_exc()  # Print stack trace for debugging
                logger.exception("Urc calculation failed for %s: %s", self.name, e)
            except SystemExit as e:
                logger.error("Urc failed for %s: %s", self.name, e)

            # --- Calculate Degradation Rate ---
            try: 
                self.degradation_results_this_iteration = dict()
                self.degradation_results_Urc1_this_iteration = dict()
                self.degradation_results_Urc2_this_iteration = dict()
                aging_rate_at_each_load_level = dict()
                
                for i in self.Iref:
                    self.degradation_results_this_iteration[i] = self.get_degradation_rate(i)
                    if self.degradation_results_this_iteration[i]:
                        aging_rate_at_each_load_level[i] = self.degradation_results_this_iteration[i].aging_uV_per_h
                    else:
                        aging_rate_at_each_load_level[i] = 0 # Fallback

                # Voltage correction (remove irreversible degradation for next iteration)
                data['calh'] = (data.index - self.installation_time) / pd.Timedelta(hours=1)
                aging_rate_list = list(aging_rate_at_each_load_level.values())
                
                # Interpolate to compute dU
                if len(aging_rate_list) > 1:
                    agingVSload = scipy.interpolate.interp1d(self.Iref, aging_rate_list, fill_value="extrapolate")
                else:
                    load_add_zero = self.Iref + [0]
                    rate_add_zero = aging_rate_list + [0]
                    agingVSload = scipy.interpolate.interp1d(load_add_zero, rate_add_zero, fill_value="extrapolate")
                
                dU = data['calh'] * agingVSload(data['currentDensity']) / 1e6
                data['voltage'] = data['voltage'] - dU  # Update voltage for next iteration

                # Collect results of this iteration
                self.degradation_results_all_iterations.append(self.degradation_results_this_iteration)
                self.degradation_results_Urc1_all_iterations.append(self.degradation_results_Urc1_this_iteration)

                # Convergence check
                if iteration > 1 and iter_idx == 0 and np.any(np.array(aging_rate_list) < 0):
                    self.warning += f"Iterative process stopped: degradation rate is negative."
                    break
                if iteration > 1 and iter_idx == iteration - 1:
                    for i in self.Iref:
                        deg_res = self.degradation_results_this_iteration.get(i)
                        if deg_res and abs(deg_res.aging_uV_per_h) / np.sqrt(deg_res.uncertainty_uV2) > 1:
                            self.warning += f"Iterative process didn't converge (@{i}A/cm2)."
                            break
            except Exception:
                pass

            self.timing_metrics["model_fitting_seconds"] = self.fitting_runtime_seconds
            self.timing_metrics["model_fitting_seconds_per_iteration"] = list(self.fitting_runtime_seconds_per_iteration)

        # === Final Result Aggregation ===
        self.degradation_results = dict()
        self.degradation_results_Urc1 = dict()
        
        for i in self.Iref:
            try:
                aging_uV_per_h = 0
                # Accumulate aging rate from each iteration
                for deg_results in self.degradation_results_all_iterations:  
                    if i in deg_results:  
                        aging_uV_per_h += deg_results[i].aging_uV_per_h
                        logger.info(
                            "Urc %sA/cm2: %suV/h, sigma: %s",
                            i,
                            round(deg_results[i].aging_uV_per_h, 2),
                            np.sqrt(deg_results[i].uncertainty_uV2),
                        )
                
                # Use intercept and uncertainty from last iteration
                last_res = self.degradation_results_this_iteration.get(i)
                if last_res:
                    self.degradation_results[i] = degradation_result(aging_uV_per_h, last_res.bol_V, last_res.uncertainty_uV2)
            except Exception:
                pass

        return

    # ==========================================
    # Helper Methods
    # ==========================================

    def get_Urc_results(self, i):
        """Get Urc results as a DataFrame."""
        closest_i = min(self.Iref, key=lambda x: abs(x - i))
        if abs(i - closest_i) > 0.015: 
            logger.warning("No Urc result available at %sA/cm2.", i)
            return

        # Accumulate previous aging rates to back-calculate original Urc
        aging_uV_per_h = 0
        if self.iteration > 1:
            for deg_results_each_iteration in self.degradation_results_all_iterations[:-1]:  
                if closest_i in deg_results_each_iteration:
                    aging_uV_per_h += deg_results_each_iteration[closest_i].aging_uV_per_h / 1e6
        
        self.Urc_results = pd.DataFrame() 
        try:
            # Urc1_results from the last iteration
            if hasattr(self, 'Urc1_results'):
                # Key: add back previously subtracted aging to restore true physical values
                self.Urc1_results[f"Urc_{closest_i}_back_calculate"] = self.Urc1_results[f"Urc_{closest_i}"] + aging_uV_per_h * self.Urc1_results["calh"]
                
                to_save = self.Urc1_results[["date", "calh", f"Urc_{closest_i}_back_calculate", f"Urc_se_{closest_i}"]].copy()
                to_save["method"] = "Urc1"
                to_save.rename(columns={"date": "timestamp", f"Urc_{closest_i}_back_calculate": "Urc", f"Urc_se_{closest_i}": "Urc_se"}, inplace=True)
                
                self.Urc_results = pd.concat([self.Urc_results, to_save], axis=0)
        except Exception:
            pass

        return self.Urc_results

    def get_degradation_rate(self, i):
        """Calculate linear regression on Urc points."""
        closest_i = min(self.Iref, key=lambda x: abs(x - i))
        if abs(i - closest_i) > 0.015: return
        
        # Ensure results exist
        if not hasattr(self, 'Urc1_results'): 
            logger.warning("%s: Urc1 provides no valid results.", self.name)
            return

        self.Urc1_exist = False
        try:
            self.Urc1_results.calh = (self.Urc1_results.date - np.datetime64(self.installation_time)) / np.timedelta64(1, "h")
            
            # Exclude run-in period
            Urc1_results_excl_run_in = self.Urc1_results.query(f"calh > {self.run_in}")
            
            x1 = Urc1_results_excl_run_in.calh
            y1 = Urc1_results_excl_run_in[f"Urc_{closest_i}"]
            sy1 = Urc1_results_excl_run_in[f"Urc_se_{closest_i}"]
            
            Urc1_deg = calculate_degradation_rate_and_uncertainty(
                x1, y1, sy1, self.fix_point, self.regression_method)

            self.degradation_results_Urc1_this_iteration[closest_i] = Urc1_deg
            self.Urc1_exist = True
            
            return degradation_result(Urc1_deg.aging_uV_per_h, Urc1_deg.bol_V, Urc1_deg.uncertainty_uV2)
        except Exception:
            return

    def plot_Urc1_separately(self, i):
        self.plot_interactive_Urc1(i)

    def plot_interactive_Urc1(self, i):
        """Generates interactive Plotly chart."""
        if (not getattr(self, 'Urc1_exist', False)):
            logger.warning("%s: Cannot plot interactive chart because no valid Urc results exist.", self.name)
            return

        closest_i = min(self.Iref, key=lambda x: abs(x - i))
        if abs(i - closest_i) > 0.015: return

        urc_results = self.get_Urc_results(closest_i)
        if urc_results is None or urc_results.empty: return

        fig = go.Figure()

        # Plot Data
        if self.Urc1_exist:
            urc1_data = urc_results.query("method == 'Urc1'")
            if not urc1_data.empty:
                fig.add_trace(go.Scatter(
                    x=urc1_data.calh, y=urc1_data.Urc,
                    customdata=urc1_data.timestamp,
                    error_y=dict(type='data', array=urc1_data.Urc_se, visible=True),
                    mode='markers', name=f'Urc1 {closest_i}A/cm2',
                    marker=dict(color='orange', size=6),
                    hovertemplate="<b>%{customdata|%Y-%m-%d}</b><br>U: %{y:.4f}V<br>SE: %{error_y.array:.4f}<extra></extra>"
                ))

        # Plot Fit Line
        if closest_i in self.degradation_results:
            reg = self.degradation_results[closest_i]
            x_range = np.linspace(self.run_in, urc_results.calh.max(), 100)
            y_range = (x_range * reg.aging_uV_per_h / 1e6) + reg.bol_V
            
            fig.add_trace(go.Scatter(
                x=x_range, y=y_range, mode='lines', name='Linear Fit',
                line=dict(color='black', dash='dash'),
                hovertemplate=f"Rate: {reg.aging_uV_per_h:.2f} uV/h<extra></extra>"
            ))

        fig.update_layout(title=f"{self.name} Urc Analysis", template="plotly_white")
        
        if not os.path.exists("plots"): os.makedirs("plots")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./plots/{self.name}_Interactive_{self.method}_{closest_i}A_{ts}.html"
        fig.write_html(filename)
        logger.info("Plot saved to: %s", filename)

    def plot_coefficient_uncertainty_ratios(self, include_rejected=False):
        # Delegated to Urc1_instance; placeholder for future implementation
        pass

    def _correct_pressure(self, data):
        validate_columns(data, [self.col_pressure, 'currentDensity', 'voltage'])
        data_with_corrected_voltage = pressure_correction(data, col_i="currentDensity", col_U="voltage", 
                                                          col_pressure_gauge_bar=self.col_pressure, pressure_correction_target_bar=self.pressure_correction_target_bar)
        data_with_corrected_voltage = data_with_corrected_voltage.drop(columns=["voltage"])
        data_with_corrected_voltage = data_with_corrected_voltage.rename(columns={"voltage_P_corrected": "voltage"})
        return data_with_corrected_voltage
