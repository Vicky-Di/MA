import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from IPython.display import display
from datetime import timedelta

class Urc1OutlierInvestigator:
    """
    Tool to investigate specific intervals in Urc1 processing.
    It replicates the exact time-slicing and feature engineering logic of Urc1 
    to diagnose causes of outliers (e.g., instability, data sparsity, collinearity).
    """

    def __init__(self, urc1_instance, raw_data):
        """
        Initialize the investigator.

        :param urc1_instance: An executed instance of the Urc1 class.
        :param raw_data: The raw DataFrame containing all data (including filtered points).
        """
        self.urc1 = urc1_instance
        self.raw_data = raw_data
        
        # Capture the exact installation time (down to seconds) to align intervals correctly
        self.anchor_time = self.urc1.installation_time
        
        self.interval_start = None
        self.interval_end = None
        self.df_raw_slice = None           # Raw data background (includes noise)
        self.df_urc1_preprocessed = None   # Data actually used by Urc1 for fitting
        self.period_str = "Not Set"

        print(f"Urc1 Investigator Initialized.")
        print(f"  Anchor Time:     {self.anchor_time}")
        print(f"  Interval Length: {self.urc1.len_interval} days")

    def set_target_date(self, target_date_str):
        """
        Set the target date and calculate the exact interval start/end times.
        Aligns the time to the installation_time to match Urc1's internal slicing logic.
        """
        target_dt = pd.to_datetime(target_date_str)
        
        # Align hour/minute/second to anchor_time to ensure exact match with Urc1 intervals
        self.interval_start = target_dt.replace(
            hour=self.anchor_time.hour,
            minute=self.anchor_time.minute,
            second=self.anchor_time.second,
            microsecond=self.anchor_time.microsecond
        )
        
        self.interval_end = self.interval_start + timedelta(days=self.urc1.len_interval)
        self.period_str = f"{self.interval_start} to {self.interval_end}"
        
        print(f"\nInterval Locked: {self.period_str}")
        self._extract_and_process_data()

    def _extract_and_process_data(self):
        """
        Extract raw data and Urc1-preprocessed data, then replicate feature engineering.
        """
        # 1. Extract Raw Data (for background visualization)
        mask_raw = (self.raw_data.index >= self.interval_start) & (self.raw_data.index <= self.interval_end)
        self.df_raw_slice = self.raw_data.loc[mask_raw].copy()
        
        # 2. Extract Urc1 Preprocessed Data (already filtered by I_min, T_min, etc.)
        mask_valid = (self.urc1.data.index >= self.interval_start) & (self.urc1.data.index <= self.interval_end)
        df_subset = self.urc1.data.loc[mask_valid].copy()
        
        # 3. Replicate Feature Engineering (crucial for collinearity checks)
        if not df_subset.empty:
            df_subset["IxT"] = df_subset.currentDensity * df_subset.temperature
            df_subset["I2"] = df_subset.currentDensity ** 2
            
            # Add log_h (logarithm of hours since last start)
            if 'h_since_last_start' in df_subset.columns:
                df_subset["log_h"] = np.log(df_subset.h_since_last_start)
            else:
                print("Warning: 'h_since_last_start' missing in urc1.data")

            self.df_urc1_preprocessed = df_subset
        else:
            self.df_urc1_preprocessed = pd.DataFrame()

        # Report counts
        n_raw = len(self.df_raw_slice)
        n_urc1 = len(self.df_urc1_preprocessed)
        
        print(f"Data Extraction Report:")
        print(f"  1. Raw Data Points:        {n_raw}")
        print(f"  2. Urc1 Preprocessed Pts:  {n_urc1} (Used for fitting)")

        if n_raw > 0:
            t_start = self.df_raw_slice.index.min()

            slice_max = self.df_raw_slice.index.max()
            global_max = self.raw_data.index.max()
            t_end = min(slice_max, global_max)
            duration = (t_end - t_start).total_seconds() / 3600
            
            print(f"  3. Actual Data Coverage:   {t_start}  ->  {t_end}")
            print(f"      (Duration Covered:      {duration:.1f} hours)")
        else:
            print(f"  3. Actual Data Coverage:   [NO DATA FOUND]")
        
        if n_urc1 < self.urc1.min_num_data_required_for_fit:
             print(f"  WARNING: Data count ({n_urc1}) below threshold ({self.urc1.min_num_data_required_for_fit}).")
             print("      This point is likely unreliable or skipped.")

    # ==========================================
    # Visualization & Diagnosis Methods
    # ==========================================

    def show_basic_stats(self):
        """Display basic statistics for the preprocessed data used in fitting."""
        if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty: return

        print(f"\n{'='*20} 1. Basic Statistics (Urc1 Preprocessed) {'='*20}")
        cols_to_show = ['currentDensity', 'voltage', 'temperature', 'log_h']
        display(self.df_urc1_preprocessed[cols_to_show].describe())
        
        # Check for sufficient excitation in Current
        diff_I = self.df_urc1_preprocessed['currentDensity'].max() - self.df_urc1_preprocessed['currentDensity'].min()
        if diff_I < 0.05:
            print(f"[Diagnosis] Current range too small ({diff_I:.3f}). Risk of ill-conditioned matrix.")

        # --- Part 2: Fitting Results ---
        print(f"\n{'='*20} 2. Fitting Coefficients (Model Output) {'='*20}")
        
        try:
            results_df = self.urc1.fitting_results
            target_idx = self.interval_start
            
            if target_idx in results_df.index:
                # 1. Extract result
                row = results_df.loc[target_idx]
                
                # If multiple results found for same timestamp, take the first one
                if isinstance(row, pd.DataFrame):
                    print(f"Found {len(row)} results for this time. Showing the first one.")
                    row = row.iloc[0] 
                
                # 2. Define columns to display
                cols_output = ['c1', 'c1_se', 'c2', 'c2_se', 'c3', 'c3_se', 'c4', 'c4_se', 'c5', 'c5_se']
                valid_cols = [c for c in cols_output if c in row.index]
                
                # 3. Print (Series to Frame and Transpose)
                display(row[valid_cols].to_frame().T)
                
                
                if 'cond' in row:
                    cond_val = row['cond']
                    print(f"\nCondition Number (Matrix Health): {cond_val:.2e}")
                    
                # 4. Automatic Diagnosis
                print("\nCoefficient Quality Check:")
                for i in range(1, 6):
                    c_name = f'c{i}'
                    se_name = f'c{i}_se'
                    if c_name in row and se_name in row:
                        c_val = row[c_name]
                        c_se = row[se_name]
                        # Calculate ratio only if coefficient is non-zero
                        if abs(c_val) > 1e-9:
                            ratio = c_se / abs(c_val)
                            
                            if ratio > 1.0:
                                print(f"  c{i} is UNRELIABLE! (Error > Value). Ratio: {ratio:.1f}")
                            elif ratio > 0.5:
                                print(f"  c{i} is Unstable. (Error is large). Ratio: {ratio:.1f}")
                            else:
                                pass # Coefficient quality is acceptable
            else:
                print(f"No fitting result found for timestamp: {target_idx}")


        except Exception as e:
            print(f"Error retrieving coefficients: {e}")


    def plot_time_series(self):
        """
        Plot Time Series Trends for Voltage, Current, Temperature, and Duration.
        Visualizes 'Raw Data' vs 'Used Data'.
        """
        if self.df_raw_slice is None: return
        
        print(f"\n{'='*20} 2. Time Series Investigation (4-Way Split) {'='*20}")
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        # --- Subplot 1: Voltage (Target) ---
        ax = axes[0]
        ax.plot(self.df_raw_slice.index, self.df_raw_slice['voltage'], 
                color='lightgrey', label='Raw (Filtered Out)', alpha=0.8)
        if not self.df_urc1_preprocessed.empty:
            ax.scatter(self.df_urc1_preprocessed.index, self.df_urc1_preprocessed['voltage'], 
                       color='blue', s=15, label='Used (Preprocessed)')
        ax.set_ylabel("Voltage [V]")
        ax.set_title(f"1. Voltage Trend (Interval: {self.period_str})")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # --- Subplot 2: Current Density ---
        ax = axes[1]
        ax.plot(self.df_raw_slice.index, self.df_raw_slice['currentDensity'], 
                color='lightgrey', label='Raw I')
        if not self.df_urc1_preprocessed.empty:
            ax.scatter(self.df_urc1_preprocessed.index, self.df_urc1_preprocessed['currentDensity'], 
                       color='orange', s=15, label='Used I')
        # Add threshold line
        ax.axhline(self.urc1.data_filter_i_min, color='orange', linestyle='--', alpha=0.5, label='Min Threshold')
        ax.set_ylabel("Current [A/cm2]")
        ax.set_title("2. Current Density")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # --- Subplot 3: Temperature ---
        ax = axes[2]
        ax.plot(self.df_raw_slice.index, self.df_raw_slice['temperature'], 
                color='lightgrey', label='Raw T')
        if not self.df_urc1_preprocessed.empty:
            ax.scatter(self.df_urc1_preprocessed.index, self.df_urc1_preprocessed['temperature'], 
                       color='red', s=15, label='Used T')
        # Add threshold line
        ax.axhline(self.urc1.data_filter_T_min, color='red', linestyle='--', alpha=0.5, label='Min Threshold')
        ax.set_ylabel("Temperature [°C]")
        ax.set_title("3. Temperature")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # --- Subplot 4: Hours Since Last Start ---
        ax = axes[3]
        col_h = 'h_since_last_start'
        
        if col_h in self.df_raw_slice.columns:
            ax.plot(self.df_raw_slice.index, self.df_raw_slice[col_h], 
                    color='lightgrey', label='Raw Duration')
            
        if not self.df_urc1_preprocessed.empty and col_h in self.df_urc1_preprocessed.columns:
            ax.scatter(self.df_urc1_preprocessed.index, self.df_urc1_preprocessed[col_h], 
                       color='green', s=15, label='Used Duration')
            
        ax.set_ylabel("Hours since start [h]")
        ax.set_title("4. Duration Since Start")
        ax.set_xlabel("Time")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

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
        cols = ['IxT', 'I2', 'voltage']
        # Filter out missing columns
        cols_to_plot = [c for c in cols if c in self.df_urc1_preprocessed.columns]
        
        # Plot using Seaborn
        g = sns.pairplot(self.df_urc1_preprocessed[cols_to_plot], 
                         diag_kind='kde', 
                         plot_kws={'alpha': 0.5, 's': 10},
                         corner=False) 
        
        g.fig.suptitle("Pair Plot: IxT, I2, voltage", y=1.02)
        plt.show()

    def plot_joint_distribution(self, kind="reg"):
        """6. Joint Distribution Plot (Seaborn Jointplot)"""
        if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty: return

        print(f"\n{'='*20} 6. Joint Distribution (Current vs Temp) {'='*20}")
        g = sns.jointplot(data=self.df_urc1_preprocessed, 
                          x="currentDensity", 
                          y="temperature", 
                          kind=kind, 
                          color="#4CB391", 
                          height=8)
        
        g.set_axis_labels("Current Density [A/cm2]", "Temperature [°C]")
        g.fig.suptitle(f"Joint Distribution: {self.period_str}", y=1.02)
        plt.show()


        print(f"\n{'='*20} 6. Joint Distribution (logh vs Temp) {'='*20}")
        g = sns.jointplot(data=self.df_urc1_preprocessed, 
                          x="log_h", 
                          y="temperature", 
                          kind=kind, 
                          color="#4CB391", 
                          height=8)
        
        g.set_axis_labels("Log_h", "Temperature [°C]")
        g.fig.suptitle(f"Joint Distribution: {self.period_str}", y=1.02)
        plt.show()

    def run_full_diagnosis(self, target_date_str):
        """Execute the full diagnosis pipeline."""
        self.set_target_date(target_date_str)
        
        if self.df_urc1_preprocessed is None or self.df_urc1_preprocessed.empty:
            print("No valid data for Urc1 in this period.")
            if self.df_raw_slice is not None and not self.df_raw_slice.empty:
                self.plot_time_series()
            return

        self.show_basic_stats()
        self.plot_time_series()
        self.analyze_correlation_features()
        self.plot_pair_relationships()
        self.plot_pair_relationships_variables()
        self.plot_3d_interactive()
        self.plot_joint_distribution()