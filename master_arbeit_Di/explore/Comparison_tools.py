import datetime
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# Lazy imports to avoid circular dependencies
try:
    from ...degradation_toolbox.Urc.Urc1_surface import Urc1_Surface
except ImportError:
    Urc1_Surface = None

try:
    from ...degradation_toolbox.Urc.Urc1_surface_stats import Urc1_Surface_Stats
except ImportError:
    Urc1_Surface_Stats = None

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions for Monotonicity Calculation
# =============================================================================

def calculate_rank_monotonicity(y: Union[pd.Series, np.ndarray]) -> float:
    """
    [Macro Trend] Spearman's Rank Correlation Method.
    Formula: Monotonicity = | corr( rank(X), rank(T) ) |
    
    Returns:
        float: [0, 1]. 1.0 means perfect monotonic trend.
    """
    y_clean = pd.Series(y).dropna()
    n = len(y_clean)
    
    if n < 2:
        return 0.0
    
    t_clean = np.arange(n)
    corr, _ = spearmanr(y_clean, t_clean)
    
    if np.isnan(corr):
        return 0.0
        
    return abs(corr)


def calculate_sign_monotonicity(y: Union[pd.Series, np.ndarray]) -> float:
    """
    [Micro Stability] Signum Function Method.
    Formula: Monotonicity = | sum(sgn(x_{k+1} - x_k)) / (N-1) |
    
    Returns:
        float: [0, 1]. 1.0 means strictly monotonic (no jitter).
    """
    y_arr = np.array(y)
    y_arr = y_arr[~np.isnan(y_arr)]
    
    n = len(y_arr)
    if n < 2:
        return 0.0
    
    diffs = np.diff(y_arr)
    signs = np.sign(diffs)
    monotonicity = abs(np.sum(signs)) / (n - 1)
    
    return monotonicity


# =============================================================================
# Main Model Comparison Class
# =============================================================================

class ModelComparison:
    """Comprehensive model comparison: integrates visualization and numerical evaluation.

    Visualization Methods:
        - plot_interactive_trends(): Urc trend curve comparison
        - plot_residual_diagnostics(): Residual diagnostic plots
        - plot_coverage_gantt(): Data coverage Gantt chart
        - plot_coefficient_diagnostic(): Coefficient diagnostic plots
        - plot_metrics_radar(): Radar chart comparison

    Evaluation Methods:
        - evaluate_stability(): Single-model stability evaluation (6 key metrics)
        - compare_all(): Batch comparison of all models
        - get_summary_table(): Retrieve summary table

    Usage:
        >>> comparator = ModelComparison({"Baseline": urc_base, "Adaptive": urc_adapt})
        >>> comparator.plot_interactive_trends(target_i=1.5)
        >>> metrics_df = comparator.compare_all(i_target=1.5)
    """

    def __init__(self, models_dict: Dict[str, Any]):
        """Initialize the model comparator.

        Args:
            models_dict: Mapping of label names to Urc instances.
                Example: {"Baseline": urc_base, "Adaptive": urc_adapt}
        """
        self.models = models_dict
        self.colors = pc.qualitative.Plotly  # Default color palette
        self._cached_metrics: Dict[float, pd.DataFrame] = {}  # Cached evaluation results


    def _extract_reliable_data(self, urc_instance, target_i):
        """Extract reliable Urc data from a Urc-class instance.

        Logic mapping:
            Urc1.fitting_results_reliable -> passed to -> Urc.Urc1_results
            So we read urc_instance.Urc1_results here.

        For Surface models, the get_Urc_results() method is preferred.
        """
        # 0. Surface model path: use get_Urc_results()
        is_surface = (Urc1_Surface_Stats is not None and isinstance(urc_instance, Urc1_Surface_Stats))
        if not is_surface and Urc1_Surface is not None:
            is_surface = isinstance(urc_instance, Urc1_Surface)
        
        if is_surface and hasattr(urc_instance, 'get_Urc_results'):
            try:
                df = urc_instance.get_Urc_results(target_i)
                if df is not None and not df.empty:
                    # Unify column names
                    if 'timestamp' in df.columns:
                        df['Timestamp'] = df['timestamp']
                    if 'calh' not in df.columns:
                        df['calh'] = (df['Timestamp'] - urc_instance.installation_time) / pd.Timedelta(hours=1)
                    df['Urc'] = df['Urc']
                    df['Urc_se'] = df['Urc_se']
                    df['Windows'] = 1
                    df['Cond'] = np.nan
                    df['Status'] = 'Surface'
                    len_interval_days = getattr(urc_instance, 'len_interval', 2)
                    df['Time_Span_Hours'] = len_interval_days * 24
                    df['End_Time'] = df['Timestamp'] + pd.Timedelta(days=len_interval_days)
                    df['Start_Time'] = df['Timestamp']
                    # Filter NaN values
                    df = df.dropna(subset=['Urc'])
                    return df
            except Exception as e:
                logger.warning("Surface get_Urc_results failed: %s", e)
        
        # 1. Retrieve source data (targeting 'Urc' class instance)
        # In the Urc class, Urc1_results stores the reliable data
        if hasattr(urc_instance, 'Urc1_results') and urc_instance.Urc1_results is not None:
            df = urc_instance.Urc1_results.copy()
            
        # Fallback: if a Urc1 or Urc1_Adaptive instance was passed directly
        elif hasattr(urc_instance, 'fitting_results_reliable'):
            df = urc_instance.fitting_results_reliable.copy()
            
        # Last resort: filter from the full result set
        elif hasattr(urc_instance, 'Urc1_results_incl_unreliable'):
            raw_df = urc_instance.Urc1_results_incl_unreliable.copy()
            threshold = getattr(urc_instance, 'threshold', 1e6)
            
            # Build query
            query_str = f"cond < {threshold}"
            if 'quality' in raw_df.columns:
                query_str += " and quality == 'good'"
            
            try:
                df = raw_df.query(query_str).reset_index()
            except:
                return None
        else:
            return None

        # 2. Verify target current column exists
        col_urc = f"Urc_{target_i}"
        col_se = f"Urc_se_{target_i}"
        
        if col_urc not in df.columns: 
            return None
        
        # 3. Process time columns
        # Urc1_results is typically reset_index'd and has a 'date' column
        if 'date' in df.columns:
            df['Timestamp'] = df['date']
        elif 'timestamp' in df.columns:
            df['Timestamp'] = df['timestamp']
        else:
            # If MultiIndex (should already be reset, but just in case)
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df['Timestamp'] = df['date'] if 'date' in df.columns else df.iloc[:, 0]
            else:
                df['Timestamp'] = df.index

        # 4. Unify field differences between Adaptive and Baseline
        if 'num_windows_used' not in df.columns:
            # === Baseline logic ===
            df['Windows'] = 1
            # Get interval length (default 2 days)
            interval_hours = getattr(urc_instance, 'len_interval', 2) * 24
            df['Time_Span_Hours'] = interval_hours
            
            # If no adaptive_status column, mark as Baseline
            if 'adaptive_status' in df.columns:
                df['Status'] = df['adaptive_status']
            else:
                df['Status'] = 'Baseline'
        else:
            # === Adaptive logic ===
            df['Windows'] = df['num_windows_used']
            df['Time_Span_Hours'] = df['time_span_hours']
            df['Status'] = df['adaptive_status']

        # 5. Compute start and end times
        # End Time = Timestamp + Interval Length
        len_interval_days = getattr(urc_instance, 'len_interval', 2)
        df['End_Time'] = df['Timestamp'] + pd.Timedelta(days=len_interval_days)
        
        # Start Time = End Time - Actual Span
        if 'time_span_hours' in df.columns:
             span = df['Time_Span_Hours'].fillna(len_interval_days * 24)
             df['Start_Time'] = df['End_Time'] - pd.to_timedelta(span, unit='h')
        else:
             df['Start_Time'] = df['Timestamp']
        
        # 6. Extract numerical values
        df['Urc'] = df[col_urc]
        df['Urc_se'] = df[col_se]
        # Some legacy code may store cond in 'cond' or 'condition_number'
        if 'cond' in df.columns:
            df['Cond'] = df['cond']
        else:
            df['Cond'] = np.nan
        
        return df

    def _get_degradation_result(self, model, target_i, df=None):
        """
        Get degradation trend parameters in a backward-compatible way.

        Priority:
        1) Use model.degradation_results[target_i] if available.
        2) Try model.get_degradation_rate(target_i) to populate/return results.
        3) Fallback: fit a simple linear trend from extracted reliable points.
        """
        # 1) Existing cached degradation results
        deg_map = getattr(model, 'degradation_results', None)
        if isinstance(deg_map, dict):
            if target_i in deg_map:
                return deg_map[target_i]

            if hasattr(model, 'get_degradation_rate'):
                try:
                    reg = model.get_degradation_rate(target_i)
                    if reg is not None:
                        return reg
                except Exception:
                    pass

                if target_i in deg_map:
                    return deg_map[target_i]

        # 2) Some implementations return regression result directly
        if hasattr(model, 'get_degradation_rate'):
            try:
                reg = model.get_degradation_rate(target_i)
                if reg is not None and hasattr(reg, 'aging_uV_per_h') and hasattr(reg, 'bol_V'):
                    return reg
            except Exception:
                pass

        # 3) Fallback: weighted linear fit on available Urc points
        if df is None:
            df = self._extract_reliable_data(model, target_i)
        if df is None or df.empty:
            return None

        temp = df.copy()
        if 'calh' not in temp.columns and 'Timestamp' in temp.columns and hasattr(model, 'installation_time'):
            temp['calh'] = (temp['Timestamp'] - model.installation_time) / pd.Timedelta(hours=1)

        if 'calh' not in temp.columns or 'Urc' not in temp.columns:
            return None

        temp = temp.replace([np.inf, -np.inf], np.nan).dropna(subset=['calh', 'Urc'])
        if len(temp) < 2:
            return None

        x = temp['calh'].to_numpy(dtype=float)
        y = temp['Urc'].to_numpy(dtype=float)
        se = None
        if 'Urc_se' in temp.columns:
            se = temp['Urc_se'].to_numpy(dtype=float)

        try:
            if se is not None:
                valid_w = np.isfinite(se) & (se > 0)
                if valid_w.sum() >= 2:
                    coeffs, cov = np.polyfit(x[valid_w], y[valid_w], 1, w=1.0 / se[valid_w], cov=True)
                else:
                    coeffs, cov = np.polyfit(x, y, 1, cov=True)
            else:
                coeffs, cov = np.polyfit(x, y, 1, cov=True)
        except Exception:
            return None

        slope_v_per_h = float(coeffs[0])
        intercept_v = float(coeffs[1])
        slope_var_v2_per_h2 = float(cov[0, 0]) if cov is not None and np.size(cov) >= 1 else np.nan

        return SimpleNamespace(
            aging_uV_per_h=slope_v_per_h * 1e6,
            bol_V=intercept_v,
            uncertainty_uV2=slope_var_v2_per_h2 * 1e12 if np.isfinite(slope_var_v2_per_h2) else np.nan,
        )
        
    


    def plot_interactive_trends(self, target_i=1.5, save=False):
        """Interactive Urc trend comparison: overlay all models' Urc curves and fit lines."""
        fig = go.Figure()
        
        for idx, (label, model) in enumerate(self.models.items()):
            df = self._extract_reliable_data(model, target_i)
            if df is None or df.empty: continue
            
            color = self.colors[idx % len(self.colors)]
            
            # A. Scatter plot (with error bars)
            fig.add_trace(go.Scatter(
                x=df['Timestamp'], y=df['Urc'],
                error_y=dict(type='data', array=df['Urc_se'], visible=True, thickness=1, width=0),
                mode='markers',
                name=f"{label} Data",
                legendgroup=label,
                marker=dict(color=color, size=5),
                # hovertemplate=f"<b>{label}</b><br>Urc: %{{y:.4f}} V<br>SE: ±%{{error_y.array:.4f}} V<extra></extra>"
                hovertemplate=f"<b>{label}</b><br>Urc: %{{y:.4f}} V"
            ))
            
            # B. Fit line
            reg = self._get_degradation_result(model, target_i, df=df)
            if reg is not None:
                # Generate two-point line
                x_range = np.array([df['calh'].min(), df['calh'].max()])
                y_range = (x_range * reg.aging_uV_per_h / 1e6) + reg.bol_V
                
                fig.add_trace(go.Scatter(
                    x=[df['Timestamp'].min(), df['Timestamp'].max()], # Connect first and last timestamps
                    y=y_range,
                    mode='lines',
                    name=f"{label} Fit ({reg.aging_uV_per_h:.1f} uV/h)",
                    legendgroup=label,
                    line=dict(color=color, dash='dash', width=2),
                    hoverinfo='skip'
                ))

        fig.update_layout(title=f"Multi-Method Urc Trend Comparison (@{target_i} A/cm²)", 
                          yaxis_title="Voltage [V]", height=600, template="plotly_white")
        fig.show()
        if save: fig.write_html(f"plots/Comparison_Trends_{target_i}A.html")


    def plot_residual_diagnostics(self, target_i=1.5):
        """Residual diagnostic plot: deviation of each point from the linear degradation trend.

        Y-axis: Residual [mV]
        Color: Windows (Adaptive) or Cond (Baseline)

        Note: Not applicable to Surface models (use urc_surface.plot_residual_diagnostics()).
        """
        from plotly.subplots import make_subplots
        
        # Filter out Surface models
        def is_surface_model(name, model):
            if name.lower().startswith('surface'):
                return True
            if Urc1_Surface is not None and isinstance(model, Urc1_Surface):
                return True
            if Urc1_Surface_Stats is not None and isinstance(model, Urc1_Surface_Stats):
                return True
            return False
        
        filtered_models = {k: v for k, v in self.models.items() if not is_surface_model(k, v)}
        
        if not filtered_models:
            logger.warning("No applicable models for this plot (Surface excluded).")
            return
        
        n_models = len(filtered_models)
        fig = make_subplots(rows=1, cols=n_models, subplot_titles=list(filtered_models.keys()), shared_yaxes=True)
        
        for idx, (label, model) in enumerate(filtered_models.items()):
            df = self._extract_reliable_data(model, target_i)
            if df is None or df.empty: continue
            
            # --- 1. Get degradation trend line parameters ---
            deg_res = self._get_degradation_result(model, target_i, df=df)
            if deg_res is None:
                logger.warning("Skipping %s: cannot derive degradation trend.", label)
                continue
            slope = deg_res.aging_uV_per_h / 1e6  # Convert to V/h
            intercept = deg_res.bol_V
            
            # --- 2. Compute residuals ---
            # Requires calh column; compute on-the-fly if missing
            if 'calh' not in df.columns:
                df['calh'] = (df['Timestamp'] - model.installation_time) / pd.Timedelta(hours=1)

            # Predicted (trend) value
            U_trend = df['calh'] * slope + intercept
            # Residual in mV for readability
            df['Residual_mV'] = (df['Urc'] - U_trend) * 1000

            # --- 3. Set color column (for diagnostics) ---
            if df['Windows'].max() > 1:
                color_col = df['Windows']
                c_title = "Windows"
                c_scale = "Viridis"
            else:
                color_col = np.log10(df['Cond']) 
                c_title = "Log10(Cond)"
                c_scale = "Turbo"

            # --- 4. Plot ---
            fig.add_trace(go.Scatter(
                x=df['Timestamp'],
                y=df['Residual_mV'],
                mode='markers',
                marker=dict(
                    color=color_col,
                    colorscale=c_scale,
                    showscale=True,
                    colorbar=dict(title=c_title, x= (idx+1)/(n_models+0.2)),
                    size=8,
                    line=dict(width=1, color='DarkSlateGrey')  # Add border for clarity
                ),
                name=label,
                customdata=np.stack((df['Windows'], df['Cond'], df['Urc']), axis=-1),
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Res: %{y:.2f} mV<br>" +
                    "Win: %{customdata[0]}<br>" +
                    "Cond: %{customdata[1]:.2e}<br>" +
                    "Urc: %{customdata[2]:.4f} V" +
                    "<extra></extra>"
                )
            ), row=1, col=idx+1)
            
            # Add zero line (perfect trend)
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=idx+1)
            fig.update_xaxes(title_text="Time", row=1, col=idx+1)

        fig.update_yaxes(title_text="Residuals [mV] (Diff from Trend)", row=1, col=1)
        fig.update_layout(title=f"Residual Analysis: Outlier Detection (@{target_i} A/cm²)", 
                          height=500, template="plotly_white", showlegend=False)
        fig.show()

    def plot_coverage_gantt(self, target_i=1.5):
        """Data coverage Gantt chart: shows the actual data interval used for each valid point.

        Note: Not applicable to Surface models (they use global data without segmentation).
        """
        # Filter out Surface models
        def is_surface_model(name, model):
            if name.lower().startswith('surface'):
                return True
            if Urc1_Surface is not None and isinstance(model, Urc1_Surface):
                return True
            if Urc1_Surface_Stats is not None and isinstance(model, Urc1_Surface_Stats):
                return True
            return False
        
        filtered_models = {k: v for k, v in self.models.items() if not is_surface_model(k, v)}
        
        if not filtered_models:
            logger.warning("No applicable models for this plot (Surface excluded).")
            return
        
        all_dfs = []
        for label, model in filtered_models.items():
            df = self._extract_reliable_data(model, target_i)
            if df is None: continue
            df['Method'] = label  # Column used to distinguish Y-axis
            all_dfs.append(df)
            
        if not all_dfs: return
        df_merged = pd.concat(all_dfs)
        
        fig = px.timeline(
            df_merged, 
            x_start="Start_Time", 
            x_end="End_Time", 
            y="Method", 
            color="Windows",  # Color intensity represents number of windows used
            hover_data=["Timestamp", "Urc", "Urc_se", "Cond"],
            title=f"Effective Data Coverage Intervals (Reliable Fits Only, @{target_i} A/cm²)",
            height=400 + 50*len(self.models),  # Dynamic height
            color_continuous_scale="Viridis"
        )
        
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Timeline", template="plotly_white")
        fig.show()


    def plot_coefficient_diagnostic(self):
        """Coefficient diagnostic plot (independent of target_i).

        Displays the 5 core fitted coefficients and their uncertainty.

        Style:
            - Solid line + solid color: coefficient value c
            - Transparent fill region: uncertainty range c +/- c_se
        """
        coeffs = ['c1', 'c2', 'c3', 'c4', 'c5']
        fig = make_subplots(
            rows=5, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=[f"Coefficient {c.upper()}" for c in coeffs]
        )

        for m_idx, (label, model) in enumerate(self.models.items()):
            # ============================================================
            # Retrieve fitting_results: compatible with Urc wrapper and direct Urc1/Urc1_Surface
            # ============================================================
            df = None
            
            # Case 1: Called via Urc wrapper (Baseline/Adaptive)
            if hasattr(model, 'urc_instance') and hasattr(model.urc_instance, 'fitting_results'):
                df = model.urc_instance.fitting_results.copy()
            # Case 2: Called directly on Urc1, Urc1_Adaptive, or Urc1_Surface
            elif hasattr(model, 'fitting_results'):
                df = model.fitting_results.copy()
            else:
                logger.warning("Model '%s' has no fitting_results attribute, skipping.", label)
                continue
            
            if df is None or df.empty:
                logger.warning("Model '%s' fitting_results is empty, skipping.", label)
                continue
                
            # Process index to ensure time column exists
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            
            # Handle column name differences (date vs timestamp)
            time_col = None
            for col_candidate in ['date', 'Timestamp', 'timestamp']:
                if col_candidate in df.columns:
                    time_col = col_candidate
                    break
            
            if time_col is None:
                logger.warning("Model '%s' has no time column, skipping.", label)
                continue
                
            # ============================================================
            # Color setup: solid color + transparent fill
            # ============================================================
            color = self.colors[m_idx % len(self.colors)]
            # Extract RGB values for transparent fill
            # color format may be 'rgb(R,G,B)' or '#RRGGBB'
            if color.startswith('rgb'):
                rgba_fill = color.replace('rgb', 'rgba').replace(')', ', 0.2)')
            elif color.startswith('#'):
                # Convert hex to rgba
                hex_color = color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                rgba_fill = f'rgba({r}, {g}, {b}, 0.2)'
            else:
                rgba_fill = 'rgba(100, 100, 100, 0.2)'

            for c_idx, c_name in enumerate(coeffs):
                se_name = f"{c_name}_se"
                
                # Check if column exists
                if c_name not in df.columns:
                    continue
                if se_name not in df.columns:
                    # If no SE column, plot coefficient line only
                    temp_df = df.dropna(subset=[c_name]).sort_values(time_col)
                    if temp_df.empty:
                        continue
                    
                    fig.add_trace(go.Scatter(
                        x=temp_df[time_col], 
                        y=temp_df[c_name],
                        mode='lines+markers',
                        name=label if c_idx == 0 else "",
                        legendgroup=label,
                        showlegend=(c_idx == 0),
                        marker=dict(size=4),
                        line=dict(color=color, width=2),
                        hovertemplate=(
                            f"<b>{label}</b><br>" +
                            "Time: %{x}<br>" +
                            f"{c_name}: %{{y:.4e}}<extra></extra>"
                        )
                    ), row=c_idx+1, col=1)
                    continue
                
                # Only plot points where the coefficient was successfully fitted
                temp_df = df.dropna(subset=[c_name, se_name]).sort_values(time_col)
                
                if temp_df.empty:
                    continue
                
                # ============================================================
                # Draw transparent fill region (uncertainty band)
                # ============================================================
                fig.add_trace(go.Scatter(
                    x=list(temp_df[time_col]) + list(temp_df[time_col])[::-1],
                    y=list(temp_df[c_name] + temp_df[se_name]) + list(temp_df[c_name] - temp_df[se_name])[::-1],
                    fill='toself',
                    fillcolor=rgba_fill,
                    line=dict(color='rgba(255,255,255,0)'),  # Transparent border
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=label
                ), row=c_idx+1, col=1)

                # ============================================================
                # Draw solid coefficient line (overlaid on top of fill region)
                # ============================================================
                fig.add_trace(go.Scatter(
                    x=temp_df[time_col], 
                    y=temp_df[c_name],
                    mode='lines+markers',
                    name=label if c_idx == 0 else "",
                    legendgroup=label,
                    showlegend=(c_idx == 0),
                    marker=dict(size=4, color=color),
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        f"<b>{label}</b><br>" +
                        "Time: %{x}<br>" +
                        f"{c_name}: %{{y:.4e}}<br>" +
                        f"SE: ±%{{customdata:.4e}}<extra></extra>"
                    ),
                    customdata=temp_df[se_name]
                ), row=c_idx+1, col=1)

        fig.update_layout(
            height=1300,
            title_text="Raw Model Coefficients Analysis (Independent of I_ref)",
            template="plotly_white",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.show()

    # =========================================================================
    # [Numerical Evaluation] Evaluation Methods
    # =========================================================================
    
    def evaluate_stability(
        self, 
        model_name: str,
        i_target: float, 
        outlier_threshold_method: Union[str, float] = '2rmse'
    ) -> Optional[Dict[str, Union[float, int]]]:
        """Compute stability metrics for a single model at a given current density.

        Args:
            model_name: Model name (must be a key in self.models).
            i_target: Target current density (e.g., 1.5).
            outlier_threshold_method: Outlier detection criterion.
                - '2rmse': deviation > 2 * RMSE (default)
                - '3rmse': deviation > 3 * RMSE
                - float: absolute deviation threshold in mV

        Returns:
            Dictionary of 6 metrics, or None if data is unavailable.
        """
        if model_name not in self.models:
            logger.warning("Model '%s' not found in comparator.", model_name)
            return None
            
        urc_instance = self.models[model_name]
        
        # --- 1. Get data and regression model ---
        df = urc_instance.get_Urc_results(i_target)
        
        if df is None or df.empty:
            logger.warning("No Urc results found for %s A/cm2", i_target)
            return None

        # Find the closest reference current
        closest_i = min(urc_instance.Iref, key=lambda x: abs(x - i_target))
        
        if closest_i not in urc_instance.degradation_results:
            logger.warning("No regression model found for %s A/cm2", closest_i)
            return None

        reg_result = urc_instance.degradation_results[closest_i]
        
        # Extract parameters and unify units
        slope_v_per_h = reg_result.aging_uV_per_h / 1e6
        intercept_v = reg_result.bol_V
        
        # --- 2. Compute core vectors ---
        y_true = df['Urc']
        y_pred = df['calh'] * slope_v_per_h + intercept_v
        residuals_v = y_true - y_pred
        residuals_mv = residuals_v * 1000

        # --- 3. Compute 6 key metrics ---
        
        # Metric 1: RMSE (global fit quality)
        mse = mean_squared_error(y_true, y_pred)
        rmse_mv = np.sqrt(mse) * 1000 
        n = len(y_true)
        
        # Metric 2: Slope Uncertainty (degradation rate confidence)
        slope_sigma_uv_h = np.sqrt(reg_result.uncertainty_uV2)
        
        # Metric 3: Monotonicity (trend consistency)
        mono_rank = calculate_rank_monotonicity(y_true)
        mono_sign = calculate_sign_monotonicity(y_true)

        # Metric 4: Outlier Count (number of anomalous points)
        if outlier_threshold_method == '2rmse':
            limit_mv = 2 * rmse_mv
        elif outlier_threshold_method == '3rmse':
            limit_mv = 3 * rmse_mv
        elif isinstance(outlier_threshold_method, (int, float)):
            limit_mv = float(outlier_threshold_method)
        else:
            limit_mv = 2 * rmse_mv
            
        outlier_count = (residuals_mv.abs() > limit_mv).sum()
        
        # Metric 5: Max Residual (largest residual)
        max_res_mv = residuals_mv.abs().max()
        
        # Metric 6: Mean SE (average standard error)
        mean_se_mv = df['Urc_se'].mean() * 1000

        # --- 4. Package results ---
        metrics = {
            "Model Name": model_name,
            "Target Current (A/cm2)": i_target,
            "Data Points (n)": n,
            "RMSE (mV)": round(rmse_mv, 3),
            "Slope Sigma (uV/h)": round(slope_sigma_uv_h, 3),
            "Mono (Rank) [0-1]": round(mono_rank, 3),
            "Mono (Sign) [0-1]": round(mono_sign, 3),
            "Outlier Count": outlier_count,
            "Outlier Limit (mV)": round(limit_mv, 2), 
            "Max Residual (mV)": round(max_res_mv, 3),
            "Mean SE (mV)": round(mean_se_mv, 3)
        }
        
        return metrics

    def compare_all(
        self, 
        i_target: float,
        outlier_threshold_method: Union[str, float] = '2rmse'
    ) -> pd.DataFrame:
        """Batch comparison of all models.

        Args:
            i_target: Target current density.
            outlier_threshold_method: Outlier detection criterion.

        Returns:
            DataFrame containing metrics for all models.
        """
        results = []
        
        for name in self.models.keys():
            metrics = self.evaluate_stability(name, i_target, outlier_threshold_method)
            if metrics:
                results.append(metrics)
                
        if not results:
            return pd.DataFrame()
            
        df_res = pd.DataFrame(results)
        
        # Cache results
        self._cached_metrics[i_target] = df_res
        
        return df_res

    def get_summary_table(
        self, 
        i_targets: List[float] = [0.6, 1.0, 1.5]
    ) -> pd.DataFrame:
        """Retrieve a summary table across multiple current densities.

        Args:
            i_targets: List of current densities to evaluate.

        Returns:
            Concatenated summary DataFrame.
        """
        all_results = []
        
        for i_target in i_targets:
            df = self.compare_all(i_target)
            if not df.empty:
                all_results.append(df)
                
        if not all_results:
            return pd.DataFrame()
            
        return pd.concat(all_results, ignore_index=True)

    def plot_metrics_radar(self, i_target: float = 1.5, save: bool = False):
        """Radar chart visualization: compare model performance across multiple metrics.

        Note: Metrics are normalized to [0, 1]; higher values indicate better performance.
        """
        # Retrieve or compute metrics
        if i_target not in self._cached_metrics:
            self.compare_all(i_target)
        
        df = self._cached_metrics.get(i_target)
        if df is None or df.empty:
            logger.warning("No data available for radar plot.")
            return
        
        # Select metrics for radar chart (to be normalized)
        metrics_to_plot = [
            'RMSE (mV)', 
            'Slope Sigma (uV/h)', 
            'Mono (Rank) [0-1]',
            'Max Residual (mV)', 
            'Mean SE (mV)'
        ]
        
        # Metric direction: True = higher is better, False = lower is better
        higher_is_better = {
            'RMSE (mV)': False,
            'Slope Sigma (uV/h)': False,
            'Mono (Rank) [0-1]': True,
            'Max Residual (mV)': False,
            'Mean SE (mV)': False
        }
        
        fig = go.Figure()
        
        for idx, row in df.iterrows():
            model_name = row['Model Name']
            values = []
            
            for metric in metrics_to_plot:
                val = row[metric]
                # Normalize (simplified: min-max on current data)
                col_vals = df[metric]
                if higher_is_better[metric]:
                    # Higher is better: normalize directly
                    normalized = (val - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-9)
                else:
                    # Lower is better: invert
                    normalized = 1 - (val - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-9)
                values.append(normalized)
            
            # Close the radar chart
            values.append(values[0])
            categories = metrics_to_plot + [metrics_to_plot[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model_name,
                line=dict(color=self.colors[idx % len(self.colors)])
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"Model Performance Radar Chart (@{i_target} A/cm²)",
            template="plotly_white"
        )
        
        fig.show()
        if save:
            fig.write_html(f"plots/Comparison_Radar_{i_target}A.html")

    def print_comparison_report(self, i_target: float = 1.5):
        """Print a formatted comparison report."""
        df = self.compare_all(i_target)
        
        if df.empty:
            logger.warning("No data available for comparison report.")
            return
        
        print("=" * 70)
        print(f"  MODEL COMPARISON REPORT @ {i_target} A/cm²")
        print("=" * 70)
        
        for _, row in df.iterrows():
            print(f"\n  {row['Model Name']}")
            print("-" * 40)
            print(f"  Data Points:      {row['Data Points (n)']}")
            print(f"  RMSE:             {row['RMSE (mV)']:.3f} mV")
            print(f"  Slope Sigma:      {row['Slope Sigma (uV/h)']:.3f} μV/h")
            print(f"  Monotonicity:     {row['Mono (Rank) [0-1]']:.3f} (Rank)")
            print(f"  Outliers:         {row['Outlier Count']}")
            print(f"  Max Residual:     {row['Max Residual (mV)']:.3f} mV")
            print(f"  Mean SE:          {row['Mean SE (mV)']:.3f} mV")
        
        print("\n" + "=" * 70)
        
        # Simple ranking
        best_rmse = df.loc[df['RMSE (mV)'].idxmin(), 'Model Name']
        best_mono = df.loc[df['Mono (Rank) [0-1]'].idxmax(), 'Model Name']
        
        print(f"  Best RMSE:        {best_rmse}")
        print(f"  Best Monotonicity: {best_mono}")
        print("=" * 70)


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Backward compatibility alias
UrcComparator = ModelComparison