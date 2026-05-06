import pandas as pd
import numpy as np
import math
import datetime
from typing import List, Tuple, Union, Any

class UrcOutlierDetector:
    """
    A dedicated class for detecting outliers in Urc (Voltage under Reference Conditions) results.
    """
    
    def __init__(
        self, 
        urc_instance: Any, 
        percentage: float = 1.0, 
        method: str = "vertical"
    ):
        """
        Initialize the detector.

        Args:
            urc_instance: An executed instance of the Urc class containing results.
                          We store this to access regression parameters and data points later.
            percentage (float): The percentage of data points to identify as outliers (e.g., 1.0 for top 1%).
            method (str): The method used to calculate distance. Currently supports 'vertical'.
        """
        self.urc = urc_instance
        self.percentage = percentage
        self.method = method

    def get_outliers(self, i_target: float) -> Tuple[List[datetime.date], pd.DataFrame]:
        """
        Identifies the top n% of data points furthest from the regression line at a specific current density.

        Args:
            i_target (float): The target current density (e.g., 1.5).

        Returns:
            Tuple[List[datetime.date], pd.DataFrame]: 
                - A list of dates corresponding to the outliers.
                - A DataFrame containing the detailed data of these outliers.
        """
        
        # --- Step 1: Retrieve Raw Data Points ---
        
        df_urc = self.urc.get_Urc_results(i_target)
        
        if df_urc is None or df_urc.empty:
            print(f"⚠️ Warning: No Urc results found for {i_target} A/cm2")
            return [], pd.DataFrame()

        # --- Step 2: Retrieve Regression Parameters (y = mx + b) ---
        
        # Logic: Find the exact key in the dictionary. User might input 1.5, but key might be 1.5000.
        closest_i = min(self.urc.Iref, key=lambda x: abs(x - i_target))
        
        if closest_i not in self.urc.degradation_results:
             print(f"⚠️ Warning: No fitting model found for {closest_i} A/cm2")
             return [], pd.DataFrame()

        reg_result = self.urc.degradation_results[closest_i]
        
        # ⚠️ Critical Unit Conversion:
        # The 'aging_uV_per_h' is stored in micro-volts/hour (uV/h).
        # The 'Urc' data is in Volts (V).
        # To calculate y = mx + b, we must convert slope to V/h by dividing by 1,000,000.
        slope_v_per_h = reg_result.aging_uV_per_h / 1e6  
        intercept_v = reg_result.bol_V # Intercept is already in V.

        # --- Step 3: Vectorized Residual Calculation (Leverage Pandas/Numpy vectorization) ---
        df_urc['Urc_pred'] = (df_urc['calh'] * slope_v_per_h) + intercept_v

        if self.method == "vertical":
            df_urc['distance_to_line'] = (df_urc['Urc'] - df_urc['Urc_pred']).abs()
        else:
            raise NotImplementedError(f"Method '{self.method}' is not currently implemented.")

        # --- Step 4: Filter Top N% (Core Logic) ---
        n_total = len(df_urc)
        n_outliers = math.ceil(n_total * (self.percentage / 100.0))
        
        # Fallback: Always return at least 1 point for analysis if data exists.
        n_outliers = max(1, n_outliers) 

        outliers_df = df_urc.nlargest(n_outliers, 'distance_to_line').copy()

        # --- Step 5: Format Output (date)---
        if 'timestamp' in outliers_df.columns:
            dates_list = outliers_df['timestamp'].dt.date.tolist()
        else:
            dates_list = outliers_df.index.date.tolist()

        print(f"=== 🔍 Outlier Detection for {i_target} A/cm2 ===")
        print(f"Total points: {n_total}. Selecting top {self.percentage}% -> {n_outliers} points.")
        
        if not outliers_df.empty:
            max_dist_mv = outliers_df['distance_to_line'].max() * 1000
            min_dist_mv = outliers_df['distance_to_line'].min() * 1000
            print(f"Max distance: {max_dist_mv:.2f} mV")
            print(f"Min distance (in selected outliers): {min_dist_mv:.2f} mV")

        return dates_list, outliers_df