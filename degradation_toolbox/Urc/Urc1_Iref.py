"""
Urc1_Iref: Ref-dependent Interval Filtering with Leverage and Quantile Checks
==============================================================================

Key improvement: Different reference currents get different reliable interval sets.

For each interval to be considered reliable for a specific Iref, it must satisfy:
    1. cond < threshold (original check)
    2. quality == 'good' (original vertex check)
    3. h0 < h_max (leverage check - detects extrapolation risk)
    4. Iref, Tref, OHref all within quantile range (dimension-wise coverage check)

This ensures that Urc predictions are only generated from intervals where:
    - The model fit is good (cond, quality)
    - The reference point is close to the data center (leverage)
    - Each reference dimension is well-covered by data (quantiles)
"""

import datetime
import numpy as np
import pandas as pd
from numpy.linalg import inv

from degradation_toolbox.Urc.Urc1 import Urc1


class Urc1_Iref(Urc1):
    """
    Urc1 with ref-dependent reliable interval filtering.
    
    Each Iref value gets its own set of reliable intervals based on:
    - Leverage: Is the reference point close to the interval's data center?
    - Quantile: Does the interval contain data near each reference dimension?
    
    Parameters
    ----------
    leverage_threshold_factor : float, default=2.5
        Threshold multiplier for leverage. h0 must be < factor * (p/n).
        Typical values: 2.0 (strict) to 3.0 (relaxed).
        
    quantile_range : tuple, default=(0.05, 0.95)
        Reference values must fall within [Q_low, Q_high] of interval data.
        E.g., (0.05, 0.95) means Iref must be between 5th and 95th percentile.
        
    All other parameters inherited from Urc1.
    """
    
    def __init__(
        self, 
        data,
        leverage_threshold_factor=2.5,
        quantile_range=(0.05, 0.95),
        **kwargs
    ):
        self.leverage_threshold_factor = leverage_threshold_factor
        self.quantile_range = quantile_range
        
        # Store interval data for statistics computation
        self._interval_data_cache = {}
        self._xtx_inv_cache = {}
        
        # Call parent __init__ (will run data preprocessing and model fitting)
        super().__init__(data, **kwargs)
        
        # Compute interval-level statistics (quantiles, etc.)
        print("\nComputing interval statistics for ref-dependent filtering...")
        self._compute_interval_stats()
        
        # Build XTX inverse cache once to accelerate leverage checks.
        self._build_xtx_inv_cache()

        # Filter reliable intervals separately for each Iref
        print("Filtering reliable intervals per Iref...")
        self.fitting_results_reliable_per_iref = {}
        for iref in self.Iref:
            reliable_df = self._filter_reliable_for_iref(iref)
            self.fitting_results_reliable_per_iref[iref] = reliable_df
            print(f"  Iref={iref}: {len(reliable_df)} reliable intervals")
        
        print(f"\nRef-dependent filtering complete.")
        print(f"  Leverage threshold factor: {self.leverage_threshold_factor}")
        print(f"  Quantile range: {self.quantile_range}")
    
    def model_fitting(self):
        """
        [Override] Enhanced model fitting that caches interval data for later stats.
        """
        # Call parent's model_fitting
        results = super().model_fitting()
        
        # Now cache the interval data for statistics computation
        installation_day = self.installation_time.date()
        end_day = max(self.data.index).date()
        num_days = (end_day - installation_day).days
        num_intervals = int(num_days / self.slide)
        max_index_time = self.data.index[-1]
        ts = self.data.index.to_numpy()
        
        for interval in range(0, num_intervals + 2):
            current_day = self.installation_time + datetime.timedelta(days=interval * self.slide)
            interval_end = current_day + datetime.timedelta(days=self.len_interval)
            
            if interval_end > max_index_time + datetime.timedelta(days=1):
                break
            
            left = np.searchsorted(ts, np.datetime64(current_day), side="left")
            right = np.searchsorted(ts, np.datetime64(interval_end), side="right")
            data_this_interval = self.data.iloc[left:right]
            
            if len(data_this_interval) >= self.min_num_data_required_for_fit:
                self._interval_data_cache[(current_day, self.name)] = data_this_interval
        
        return results
    
    def _compute_interval_stats(self):
        """
        Compute data distribution statistics for each interval.
        
        For each interval, store:
        - Quantiles: I_q05, I_q95, T_q05, T_q95, OH_q05, OH_q95
        - Count: n_points
        """
        q_low, q_high = self.quantile_range
        
        for idx, row in self.fitting_results.iterrows():
            if idx not in self._interval_data_cache:
                continue
            
            df_interval = self._interval_data_cache[idx]
            
            # Extract physical dimensions
            I = df_interval['currentDensity'].values
            T = df_interval['temperature'].values
            OH = df_interval['h_since_last_start'].values
            
            # Compute quantiles
            self.fitting_results.loc[idx, 'I_q_low'] = np.quantile(I, q_low)
            self.fitting_results.loc[idx, 'I_q_high'] = np.quantile(I, q_high)
            self.fitting_results.loc[idx, 'T_q_low'] = np.quantile(T, q_low)
            self.fitting_results.loc[idx, 'T_q_high'] = np.quantile(T, q_high)
            self.fitting_results.loc[idx, 'OH_q_low'] = np.quantile(OH, q_low)
            self.fitting_results.loc[idx, 'OH_q_high'] = np.quantile(OH, q_high)
            self.fitting_results.loc[idx, 'n_points'] = len(df_interval)
    
    def _build_xtx_inv_cache(self):
        """Precompute inverse(XTX) for valid fitted intervals."""
        self._xtx_inv_cache = {}
        if "XTX" not in self.fitting_results.columns:
            return

        for idx, xtx_value in self.fitting_results["XTX"].items():
            xtx = self._coerce_xtx_matrix(xtx_value)
            if xtx is None:
                continue
            try:
                self._xtx_inv_cache[idx] = inv(xtx)
            except Exception:
                continue

    def _compute_leverage(self, row, iref):
        """
        Compute leverage h0 for a given reference current.
        
        h0 = x0^T * (X^T X)^-1 * x0
        
        where x0 = [I_scaled, IT_scaled, logH_scaled, I2_scaled, 1]
        
        Parameters
        ----------
        row : pd.Series
            Fitting result row containing XTX matrix
        iref : float
            Reference current density
            
        Returns
        -------
        float
            Leverage value h0. Larger = further from data center = riskier.
        """
        row_idx = row.name
        XTX_inv = self._xtx_inv_cache.get(row_idx)
        if XTX_inv is None:
            XTX = self._coerce_xtx_matrix(row.get("XTX"))
            if XTX is None:
                return np.nan
            try:
                XTX_inv = inv(XTX)
                self._xtx_inv_cache[row_idx] = XTX_inv
            except Exception:
                return np.nan
        
        resolved = self._resolve_reference_input(iref)
        if resolved is None:
            return np.nan
        closest_i, tref, ohref, _ = resolved

        # Construct reference feature vector
        x0 = np.array([
            self.scaler_I.scale(closest_i),
            self.scaler_IxT.scale(tref * closest_i),
            self.scaler_log_h.scale(np.log(ohref)),
            self.scaler_I2.scale(closest_i ** 2),
            1.0
        ])
        
        try:
            # Compute leverage
            h0 = x0.T @ XTX_inv @ x0
            return h0
        except Exception:
            return np.nan

    def _filter_reliable_for_iref(self, iref):
        """
        Filter reliable intervals for a specific Iref.
        
        An interval is reliable if:
        1. cond < threshold (numerical stability)
        2. quality == 'good' (physical plausibility - vertex check)
        3. h0 < h_max (leverage - not extrapolating)
        
        Parameters
        ----------
        iref : float
            Reference current density
            
        Returns
        -------
        pd.DataFrame
            Subset of fitting_results that are reliable for this Iref
        """
        # Start with a copy
        results = self.fitting_results.copy()
        
        # Compute leverage for this Iref
        results[f'leverage_{iref}'] = results.apply(
            lambda row: self._compute_leverage(row, iref), axis=1
        )
        
        # Compute leverage threshold
        # h_max = factor * (p / n), where p=5 features
        results[f'h_max_{iref}'] = results['n_points'].apply(
            lambda n: self.leverage_threshold_factor * 5 / n if pd.notna(n) and n > 0 else np.nan
        )
        
        # Apply all filters
        mask = (
            ((results['cond'] < self.threshold) &                       # Original: cond check
             (results['quality'] == 'good')) |                          # Original: vertex check
            (results[f'leverage_{iref}'] < results[f'h_max_{iref}'])    # New: leverage check
        )
        
        reliable = results[mask].reset_index()
        return reliable
    
    def get_reliable_results(self, iref=None):
        """
        Get reliable results for a specific Iref.
        
        Parameters
        ----------
        iref : float, optional
            Reference current. If None, returns the first Iref's results.
            
        Returns
        -------
        pd.DataFrame
            Reliable fitting results for the specified Iref
        """
        if iref is None:
            iref = self.Iref[0]
        
        # Find closest available Iref
        closest_iref = min(self.Iref, key=lambda x: abs(x - iref))
        
        if abs(iref - closest_iref) > 0.01:
            print(f"Warning: Requested Iref={iref}, using closest available Iref={closest_iref}")
        
        return self.fitting_results_reliable_per_iref.get(closest_iref, pd.DataFrame())
    
    def print_summary(self):
        """
        Print a summary of reliable intervals per Iref.
        """
        print("\n" + "=" * 70)
        print("Urc1_Iref Summary: Ref-Dependent Reliable Intervals")
        print("=" * 70)
        print(f"Total intervals fitted: {len(self.fitting_results)}")
        print(f"Original reliable (cond + quality): {len(self.fitting_results_reliable)}")
        print(f"\nRef-dependent reliable intervals:")
        
        for iref in self.Iref:
            n_reliable = len(self.fitting_results_reliable_per_iref[iref])
            pct = n_reliable / len(self.fitting_results) * 100 if len(self.fitting_results) > 0 else 0
            print(f"  Iref = {iref:>5.2f} A/cm²: {n_reliable:>4d} intervals ({pct:>5.1f}%)")
        
        print("=" * 70)
    
    def plot_leverage_distribution(self, save_path=None):
        """
        Visualize leverage distribution for each Iref.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(len(self.Iref), 1, figsize=(10, 4 * len(self.Iref)))
            if len(self.Iref) == 1:
                axes = [axes]
            
            for ax, iref in zip(axes, self.Iref):
                results = self.fitting_results.copy()
                results['leverage'] = results.apply(
                    lambda row: self._compute_leverage(row, iref), axis=1
                )
                results = results.dropna(subset=['leverage'])
                
                if len(results) == 0:
                    continue
                
                # Plot leverage distribution
                ax.hist(results['leverage'], bins=30, alpha=0.7, edgecolor='black')
                
                # Mark threshold
                if 'n_points' in results.columns:
                    avg_n = results['n_points'].mean()
                    h_max = self.leverage_threshold_factor * 5 / avg_n
                    ax.axvline(h_max, color='red', linestyle='--', linewidth=2, 
                              label=f'Threshold={h_max:.4f}')
                
                ax.set_xlabel('Leverage h0')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Leverage Distribution for Iref={iref} A/cm²')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Leverage plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available. Skipping plot.")


if __name__ == '__main__':
    print(__doc__)
