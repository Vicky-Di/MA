import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from IPython.display import display
from datetime import timedelta

class Urc1OutlierInvestigator:
    """
    Tool to investigate a continuous time range in Urc1 processing.
    Updated to support multi-day analysis to diagnose trends across intervals.
    """

    def __init__(self, urc1_instance, raw_data):
        """
        Initialize the investigator.
        :param urc1_instance: An executed instance of the Urc1 class.
        :param raw_data: The raw DataFrame containing all data.
        """
        self.urc1 = urc1_instance
        self.raw_data = raw_data
        
        # Capture the exact installation time alignment
        self.anchor_time = self.urc1.installation_time
        
        self.interval_start = None
        self.interval_end = None
        self.df_raw_slice = None           
        self.df_urc1_preprocessed = None   
        self.period_str = "Not Set"

        print(f"Urc1 Investigator Initialized.")
        print(f"  Anchor Time:     {self.anchor_time}")
        # Default fitting interval length
        print(f"  Standard Interval Length: {self.urc1.len_interval} days")

    def set_time_range(self, start_date_str, duration_days=None):
        """
        Set a continuous time range for analysis.
        
        :param start_date_str: Start date string (e.g., '2023-01-01').
        :param duration_days: (Optional) How many days to analyze. 
                              If None, defaults to the standard single interval length.
        """
        target_dt = pd.to_datetime(start_date_str)
        
        # 1. Align Start Time to anchor (crucial for matching fitting grid)
        self.interval_start = target_dt.replace(
            hour=self.anchor_time.hour,
            minute=self.anchor_time.minute,
            second=self.anchor_time.second,
            microsecond=self.anchor_time.microsecond
        )
        
        # 2. Determine Duration
        if duration_days is None:
            # Default to one standard fitting interval
            days_to_add = self.urc1.len_interval
        else:
            # Custom duration (e.g., 7 days to see trends)
            days_to_add = duration_days
            
        self.interval_end = self.interval_start + timedelta(days=days_to_add)
        self.period_str = f"{self.interval_start} to {self.interval_end} ({days_to_add} days)"
        
        print(f"\nAnalysis Range Locked: {self.period_str}")
        self._extract_and_process_data()

    def _extract_and_process_data(self):
        """
        Extract raw data and Urc1-preprocessed data for the ENTIRE range.
        """
        # 1. Extract Raw Data
        mask_raw = (self.raw_data.index >= self.interval_start) & (self.raw_data.index <= self.interval_end)
        self.df_raw_slice = self.raw_data.loc[mask_raw].copy()
        
        # 2. Extract Urc1 Preprocessed Data
        mask_valid = (self.urc1.data.index >= self.interval_start) & (self.urc1.data.index <= self.interval_end)
        df_subset = self.urc1.data.loc[mask_valid].copy()
        
        # 3. Feature Engineering
        if not df_subset.empty:
            df_subset["IxT"] = df_subset.currentDensity * df_subset.temperature
            df_subset["I2"] = df_subset.currentDensity ** 2
            
            if 'h_since_last_start' in df_subset.columns:
                df_subset["log_h"] = np.log(df_subset.h_since_last_start)
            else:
                print("Warning: 'h_since_last_start' missing")

            self.df_urc1_preprocessed = df_subset
        else:
            self.df_urc1_preprocessed = pd.DataFrame()

        # Report
        print(f"Data Extraction Report:")
        print(f"  1. Raw Data Points:        {len(self.df_raw_slice)}")
        print(f"  2. Urc1 Preprocessed Pts:  {len(self.df_urc1_preprocessed)}")

    # ==========================================
    # Visualization & Diagnosis Methods
    # ==========================================

    def show_basic_stats(self):
        """
        Display stats and ALL fitting results found within the time range.
        Modified to show a table of coefficients for multiple intervals.
        """
        if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty: return

        print(f"\n{'='*20} 1. Basic Statistics (Data Distribution) {'='*20}")
        cols_to_show = ['currentDensity', 'voltage', 'temperature', 'log_h']
        display(self.df_urc1_preprocessed[cols_to_show].describe())
        
        # Diagnosis: Check I range
        diff_I = self.df_urc1_preprocessed['currentDensity'].max() - self.df_urc1_preprocessed['currentDensity'].min()
        if diff_I < 0.05:
            print(f"[Diagnosis] WARNING: Overall Current range is very small ({diff_I:.3f}) in this period.")

        # --- Part 2: Fitting Results (Multi-Interval) ---
        print(f"\n{'='*20} 2. Fitting Results in this Range {'='*20}")
        
        try:
            results_df = self.urc1.fitting_results.copy()
            
            # Reset index to make filtering easier (assuming MultiIndex [date, name])
            if isinstance(results_df.index, pd.MultiIndex):
                results_df_flat = results_df.reset_index(level='name', drop=True)
            else:
                results_df_flat = results_df
            
            # Filter results that fall strictly within the start and end time
            # Note: The 'index' of fitting_results is usually the START time of the interval
            mask_fits = (results_df_flat.index >= self.interval_start) & \
                        (results_df_flat.index < self.interval_end)
            
            relevant_fits = results_df_flat.loc[mask_fits]
            
            if not relevant_fits.empty:
                print(f"Found {len(relevant_fits)} fitting intervals within this range:")
                
                # Select key columns to display
                cols_output = ['c1', 'c2', 'c3', 'c4', 'c5', 'cond', 'R2', 'quality']
                # Add adaptive status if available
                if 'adaptive_status' in relevant_fits.columns:
                    cols_output.append('adaptive_status')
                
                valid_cols = [c for c in cols_output if c in relevant_fits.columns]
                
                display(relevant_fits[valid_cols])
                
                # Check for "Anchoring Effect" (Repeated Coefficients)
                if len(relevant_fits) > 1:
                    c1_values = relevant_fits['c1'].dropna()
                    if len(c1_values) > 1 and c1_values.nunique() == 1:
                         print(f"\n[Diagnosis] ⚠️ ANCHORING DETECTED: Coefficients are identical across {len(relevant_fits)} intervals.")
                         print("             This confirms the algorithm is backtracking to the same historical window.")
            else:
                print("No fitting start-times found strictly within this range.")

        except Exception as e:
            print(f"Error retrieving coefficients: {e}")

    def plot_time_series(self):
        """Plot Trends over the entire duration using interactive Plotly."""
        if self.df_raw_slice is None: return
        
        print(f"\n{'='*20} 2. Time Series Investigation (Interactive) {'='*20}")
        
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Create subplots: 3 rows (Voltage, Current, Temperature)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                f"Voltage ({self.period_str})",
                "Current Density",
                "Temperature"
            )
        )
        
        # Generate interval boundary lines
        interval_boundaries = []
        curr = self.interval_start
        while curr <= self.interval_end:
            interval_boundaries.append(curr)
            curr += timedelta(days=self.urc1.len_interval)
        
        # 1. Voltage
        fig.add_trace(
            go.Scatter(x=self.df_raw_slice.index, y=self.df_raw_slice['voltage'],
                       mode='lines', name='Raw Voltage', line=dict(color='lightgrey')),
            row=1, col=1
        )
        if not self.df_urc1_preprocessed.empty:
            fig.add_trace(
                go.Scatter(x=self.df_urc1_preprocessed.index, y=self.df_urc1_preprocessed['voltage'],
                           mode='markers', name='Used Voltage', marker=dict(color='blue', size=4)),
                row=1, col=1
            )
        
        # 2. Current Density
        fig.add_trace(
            go.Scatter(x=self.df_raw_slice.index, y=self.df_raw_slice['currentDensity'],
                       mode='lines', name='Raw Current', line=dict(color='lightgrey'), showlegend=False),
            row=2, col=1
        )
        if not self.df_urc1_preprocessed.empty:
            fig.add_trace(
                go.Scatter(x=self.df_urc1_preprocessed.index, y=self.df_urc1_preprocessed['currentDensity'],
                           mode='markers', name='Used Current', marker=dict(color='orange', size=4)),
                row=2, col=1
            )
        
        # 3. Temperature
        fig.add_trace(
            go.Scatter(x=self.df_raw_slice.index, y=self.df_raw_slice['temperature'],
                       mode='lines', name='Raw Temp', line=dict(color='lightgrey'), showlegend=False),
            row=3, col=1
        )
        if not self.df_urc1_preprocessed.empty:
            fig.add_trace(
                go.Scatter(x=self.df_urc1_preprocessed.index, y=self.df_urc1_preprocessed['temperature'],
                           mode='markers', name='Used Temp', marker=dict(color='red', size=4)),
                row=3, col=1
            )
        
        # Add vertical dashed lines for fitting interval boundaries on ALL subplots
        for boundary in interval_boundaries:
            for row in [1, 2, 3]:
                fig.add_vline(
                    x=boundary, 
                    line=dict(color='black', width=1, dash='dash'),
                    opacity=0.5,
                    row=row, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Time Series Investigation: {self.period_str}",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
        fig.update_yaxes(title_text="Current [A/cm²]", row=2, col=1)
        fig.update_yaxes(title_text="Temp [°C]", row=3, col=1)
        
        # Update x-axis
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        fig.show()

    def analyze_correlation_features(self):
        """
        Check for collinearity among features (I, T, IxT, log_h, I^2).
        High correlation causes matrix singularity and parameter instability.
        """
        if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty: return

        print(f"\n{'='*20} 3-1. Correlation Matrix (Collinearity Check) {'='*20}")
        corr_matrix = self.df_urc1_preprocessed[['currentDensity', 'temperature', 'log_h','voltage']].corr()
        print(corr_matrix)
        

        print(f"\n{'='*20} 3-2. Correlation Matrix (Collinearity Check) {'='*20}")
        corr_matrix = self.df_urc1_preprocessed[['currentDensity', 'temperature', 'IxT', 'log_h', 'I2']].corr()
        print(corr_matrix)


        print(f"\n{'='*20} 3-3. Collinearity Check (Features) {'='*20}")
        
        feature_cols = ['currentDensity', 'temperature', 'IxT', 'log_h', 'I2']
        cols = [c for c in feature_cols if c in self.df_urc1_preprocessed.columns]
        
        corr_matrix = self.df_urc1_preprocessed[cols].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.show()
        
        # Diagnosis: Check if temperature correlates with start-up time (thermal transient)
        if 'log_h' in cols and 'temperature' in cols:
            corr_val = corr_matrix.loc['log_h', 'temperature']
            if abs(corr_val) > 0.85:
                print(f"[Diagnosis] High correlation between log_h and T ({corr_val:.2f}).")
                print("  This suggests the interval captures a thermal transient (heating phase).")
                print("  The regression may fail to distinguish between thermal effects and timing effects.")

    def plot_3d_interactive(self, sample_rate=1):
        """
        4D Visualization: 
        Space (x,y,z) = (I, T, log_h)
        """
        if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty: return
        
        print(f"\n{'='*20} 4. Interactive 3D Spatial Distribution {'='*20}")
        
        # Downsample to avoid browser lag
        plot_data = self.df_urc1_preprocessed[::sample_rate].reset_index()
        
        fig = px.scatter_3d(plot_data, 
                            x='currentDensity', 
                            y='temperature', 
                            z='log_h',
                            color='timestamp', 
                            title=f'Interactive 3D View ({self.period_str})',
                            labels={'currentDensity': 'Current (I)', 'temperature': 'Temp (T)', 'log_h': 'logh'},
                            opacity=0.6,
                            template="plotly_dark")

        fig.update_traces(marker=dict(size=3, line=dict(width=0)))
        fig.show()

    def plot_pair_relationships(self):
        """
        Pair Plot: Visualize relationships between ALL 4 variables at once.
        This is the standard way to analyze 4+ dimensions.
        """
        if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty: return

        print(f"\n{'='*20} 5. Pairwise Relationships (4D Scan) {'='*20}")
        
        # Select variables to analyze
        cols = ['currentDensity', 'temperature', 'log_h', 'voltage']
        # Filter out missing columns
        cols_to_plot = [c for c in cols if c in self.df_urc1_preprocessed.columns]
        
        # Plot using Seaborn
        g = sns.pairplot(self.df_urc1_preprocessed[cols_to_plot], 
                         diag_kind='kde', 
                         plot_kws={'alpha': 0.5, 's': 10},
                         corner=False) 
        
        g.fig.suptitle("Pair Plot: I, T, log_h, U", y=1.02)
        plt.show()

    def plot_pair_relationships_variables(self):
        """
        Pair Plot: Visualize relationships between ALL 4 variables at once.
        This is the standard way to analyze 4+ dimensions.
        """
        if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty: return

        print(f"\n{'='*20} 5. Pairwise Relationships (4D Scan) {'='*20}")
        
        # Select variables to analyze
        cols = ['currentDensity', 'IxT','log_h', 'I2', 'voltage']
        # Filter out missing columns
        cols_to_plot = [c for c in cols if c in self.df_urc1_preprocessed.columns]
        
        # Plot using Seaborn
        g = sns.pairplot(self.df_urc1_preprocessed[cols_to_plot], 
                         diag_kind='kde', 
                         plot_kws={'alpha': 0.5, 's': 10},
                         corner=False) 
        
        g.fig.suptitle("Pair Plot: I, IxT, log_h, I2, voltage", y=1.02)
        plt.show()


    # def plot_joint_distribution(self, kind="reg"):
    #     """6. Joint Distribution Plot (Seaborn Jointplot)"""
    #     if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty: return

    #     print(f"\n{'='*20} 6. Joint Distribution (Currevoltage vs Temp) {'='*20}")
    #     g = sns.jointplot(data=self.df_urc1_preprocessed, 
    #                       x="currentDensity", 
    #                       y="temperature", 
    #                       kind=kind, 
    #                       color="#4CB391", 
    #                       height=8)
        
    #     g.set_axis_labels("Current Density [A/cm2]", "Temperature [°C]")
    #     g.fig.suptitle(f"Joint Distribution: {self.period_str}", y=1.02)
    #     plt.show()


    #     print(f"\n{'='*20} 6. Joint Distribution (logh vs Temp) {'='*20}")
    #     g = sns.jointplot(data=self.df_urc1_preprocessed, 
    #                       x="log_h", 
    #                       y="temperature", 
    #                       kind=kind, 
    #                       color="#4CB391", 
    #                       height=8)
        
    #     g.set_axis_labels("Log_h", "Temperature [°C]")
    #     g.fig.suptitle(f"Joint Distribution: {self.period_str}", y=1.02)
    #     plt.show()

    def plot_joint_distribution_log_h(self, kind="reg"):
        """6. Joint Distribution Plot (Seaborn Jointplot)"""
        if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty: return

        print(f"\n{'='*20} 6. Joint Distribution (voltage vs log_h) {'='*20}")
        g = sns.jointplot(data=self.df_urc1_preprocessed, 
                          x="voltage", 
                          y="log_h", 
                          kind=kind, 
                          color="#4CB391", 
                          height=8)
        
        g.set_axis_labels("Voltage", "log_h")
        g.fig.suptitle(f"Joint Distribution: {self.period_str}", y=1.02)
        plt.show()


        print(f"\n{'='*20} 6. Joint Distribution (Current vs log_h) {'='*20}")
        g = sns.jointplot(data=self.df_urc1_preprocessed, 
                          x="currentDensity", 
                          y="log_h", 
                          kind=kind, 
                          color="#4CB391", 
                          height=8)
        
        g.set_axis_labels("currentDensity", "log_h")
        g.fig.suptitle(f"Joint Distribution: {self.period_str}", y=1.02)
        plt.show()

    def run_full_diagnosis(self, start_date_str, duration_days=None):
        """
        Execute diagnosis.
        :param duration_days: Number of days to look ahead. If None, uses 1 interval length.
        """
        # Step 1: Set range
        self.set_time_range(start_date_str, duration_days)
        
        if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty:
            print("No valid data for Urc1 in this period.")
            if self.df_raw_slice is not None and not self.df_raw_slice.empty:
                self.plot_time_series()
            return

        # Step 2: Show stats (now includes table of coefficients over time)
        self.show_basic_stats()
        
        # Step 3: Plots
        self.plot_time_series()
        self.analyze_correlation_features()
        
        # Only run heavy plots if data isn't huge
        if len(self.df_urc1_preprocessed) < 5000:
            self.plot_pair_relationships_variables()
            self.plot_joint_distribution_log_h()
        else:
            print("Skipping PairPlot/JointPlot due to large dataset size (>5000 pts).")