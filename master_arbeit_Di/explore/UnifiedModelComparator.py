"""Unified model comparison tool for degradation analysis.

Integrates visual diagnostics (from Comparison_tools) and numerical
evaluation (from stability_evaluation) into a single comparator.

Supported model types:
    - Urc (wrapper)
    - Urc1 / Urc1_Adaptive / Urc1_GPR / Urc1_Iref / Urc1_c3 (and other
      direct subclasses)
    - Urc1_Surface / Urc1_Surface_Stats

Core data flow:
    1. _extract_reliable_data() - Unified extraction, compatible with all
       model parameter variations.
    2. _ensure_degradation_results() - Ensure degradation_results are
       available.
    3. evaluate_stability() - Integrated numerical evaluation (6+ metrics).
    4. compare_all() - Batch comparison returning a DataFrame.
    5. plot_interactive_trends() - Multi-model Urc trend comparison.
    6. plot_coefficient_diagnostic() - Multi-model coefficient comparison.
    7. plot_coverage_gantt() - (Optional) Coverage visualization for
       adaptive / expanding-window methods.

Key interface attributes:
    - Iref: Present in all models.
    - degradation_results: Directly available in Urc/Surface; Urc1 types
      require get_degradation_rate() or derivation from results.
    - get_Urc_results(): Available in Urc/Surface; Urc1 types derive from
      fitting_results_reliable.
    - fitting_results_reliable: Most models have this; Urc1_Iref uses
      fitting_results_reliable_per_iref.
    - len_interval: Most models have this; Urc1_c3 uses dynamic intervals
      requiring special handling.
"""

import datetime
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress, spearmanr
from sklearn.metrics import mean_squared_error

# Configure logger for this module
logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
try:
    from ...degradation_toolbox.Urc.Urc1_surface import Urc1_Surface
except ImportError:
    Urc1_Surface = None

try:
    from ...degradation_toolbox.Urc.Urc1_surface_stats import Urc1_Surface_Stats
except ImportError:
    Urc1_Surface_Stats = None


# =============================================================================
# Monotonicity Helper Functions
# =============================================================================

def calculate_rank_monotonicity(y: Union[pd.Series, np.ndarray]) -> float:
    """[Macro Trend] Spearman's Rank Correlation."""
    y_clean = pd.Series(y).dropna()
    n = len(y_clean)
    if n < 2:
        return 0.0
    t_clean = np.arange(n)
    corr, _ = spearmanr(y_clean, t_clean)
    if np.isnan(corr):
        return 0.0
    return abs(corr)



# =============================================================================
# Main Unified Comparator
# =============================================================================

class UnifiedModelComparator:
    """Unified model comparator integrating visualization and numerical evaluation.

    Usage:
        >>> models = {
        ...     "Baseline": urc_baseline,
        ...     "Adaptive": urc_adaptive,
        ...     "GPR": urc1_gpr,
        ...     "Surface": urc1_surface
        ... }
        >>> comparator = UnifiedModelComparator(models)
        >>> metrics_df = comparator.compare_all(i_target=1.5)
        >>> comparator.plot_interactive_trends(target_i=1.5)
        >>> comparator.plot_coefficient_diagnostic()
    """

    def __init__(self, models_dict: Dict[str, Any]):
        """
        Args:
            models_dict: Dict mapping "Model Name" -> Urc instance
        """
        self.models = models_dict
        self.colors = pc.qualitative.Plotly
        self._cached_metrics: Dict[float, pd.DataFrame] = {}
        self.gt_uref: Optional[pd.Series] = None  # Single GT (backward compat)
        self.gt_uref_per_iref: Dict[float, pd.Series] = {}  # Per-Iref GT: {Iref -> Series}
        # Strict matching mode: only use GT with the same Iref; return None
        # instead of falling back to nearest-neighbor if missing.
        self.strict_gt_iref_match: bool = True
        self.gt_iref_match_tolerance: float = 1e-9

    def _sanitize_path_component(self, value: Any, fallback: str) -> str:
        """Convert arbitrary text to a filesystem-safe path component."""
        text = str(value).strip() if value is not None else ""
        if not text:
            text = fallback
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = text.strip("._")
        return text or fallback

    def _extract_dataset_name(self, model: Any) -> str:
        """Best-effort dataset name extraction from model or nested model."""
        dataset_name = getattr(model, "name", None)
        if dataset_name:
            return self._sanitize_path_component(dataset_name, "unknown_dataset")

        nested = getattr(model, "urc_instance", None)
        if nested is not None:
            dataset_name = getattr(nested, "name", None)
            if dataset_name:
                return self._sanitize_path_component(dataset_name, "unknown_dataset")

        return "unknown_dataset"

    def _default_plots_root(self) -> Path:
        """Return canonical plots root: master_arbeit_Di/plots/adaptive."""
        return Path(__file__).resolve().parents[1] / "plots" / "adaptive"

    def _resolve_plot_save_dirs(self, output_dir: Optional[Union[str, Path]] = None) -> List[Path]:
        """Resolve one or more save directories for figure export.

        If output_dir is provided, all figures are saved there.
        Otherwise save to per-model folders:
            master_arbeit_Di/plots/<model_name>/<dataset_name>/
        """
        if output_dir is not None:
            out = Path(output_dir)
            if not out.is_absolute():
                out = (Path.cwd() / out).resolve()
            out.mkdir(parents=True, exist_ok=True)
            return [out]

        root = self._default_plots_root()
        save_dirs: List[Path] = []
        for label, model in self.models.items():
            model_name = getattr(model, "model_label", None) or label or model.__class__.__name__
            model_component = self._sanitize_path_component(model_name, "unknown_model")
            dataset_component = self._extract_dataset_name(model)
            target_dir = root / model_component / dataset_component
            target_dir.mkdir(parents=True, exist_ok=True)
            save_dirs.append(target_dir)

        # Keep unique directories in stable order
        unique_dirs: List[Path] = []
        seen = set()
        for d in save_dirs:
            ds = str(d)
            if ds not in seen:
                unique_dirs.append(d)
                seen.add(ds)
        return unique_dirs

    def _save_figure_html(self, fig: go.Figure, file_name: str,
                          output_dir: Optional[Union[str, Path]] = None) -> List[Path]:
        """Save one figure into resolved output directories."""
        save_dirs = self._resolve_plot_save_dirs(output_dir)
        saved_paths: List[Path] = []
        for out_dir in save_dirs:
            out_path = out_dir / file_name
            fig.write_html(str(out_path))
            saved_paths.append(out_path)
        return saved_paths
    
    
    # =========================================================================
    # [Core] Data Extraction & Validation
    # =========================================================================

    def load_ground_truth_bundle(
        self,
        input_dir: str,
        dataset_name: Optional[str] = None,
    ) -> Dict[float, int]:
        """Load saved GT files from a directory and register them by Iref.

        Suitable for reusing offline-generated GT data across different
        models and notebooks.

        Args:
            input_dir: Directory containing GT files.
            dataset_name: Optional dataset-name prefix to filter files.

        Returns:
            Dict mapping each Iref to the number of loaded data points.
        """
        from ...degradation_toolbox.utils.ground_truth import load_ground_truth_bundle

        gt_by_iref, metadata_by_iref = load_ground_truth_bundle(
            input_dir=input_dir,
            dataset_name=dataset_name,
        )

        results: Dict[float, int] = {}
        for iref, gt_series in gt_by_iref.items():
            self.set_ground_truth(gt_series, iref=iref)
            results[iref] = int(len(gt_series))
            source_name = metadata_by_iref.get(iref, {}).get("csv_file", "unknown")
            logger.info(
                "GT loaded @ Iref=%s: %d data points from %s",
                iref,
                len(gt_series),
                source_name,
            )

        return results

    def set_ground_truth(self, gt_uref: Optional[pd.Series], iref: Optional[float] = None):
        """Set ground truth data for GT benchmarking.

        Supports two usage patterns:

        1. Single GT (backward compatible)::

            comparator.set_ground_truth(gt_uref)
            # This GT is used for all Iref values (not recommended;
            # physically incorrect).

        2. Iref-specific GT (recommended)::

            comparator.set_ground_truth(gt_uref_0p6, iref=0.6)
            comparator.set_ground_truth(gt_uref_1p0, iref=1.0)
            comparator.set_ground_truth(gt_uref_1p5, iref=1.5)
            # Each Iref uses its corresponding GT data.

        Args:
            gt_uref: pd.Series with DatetimeIndex, values = reference
                voltage [V]. Should be extracted under the corresponding
                reference conditions (Iref, Tref, OHref).
            iref: Current density [A/cm^2] for this GT. If None, stored as
                a "global" GT (backward compatibility only).

        Note:
            Setting Iref-specific GT for each current density is strongly
            recommended over using a single global GT.
        """
        if gt_uref is not None and not isinstance(gt_uref.index, pd.DatetimeIndex):
            gt_uref = gt_uref.copy()
            gt_uref.index = pd.to_datetime(gt_uref.index)
        
        if iref is None:
            # Backward compatible: set global GT
            self.gt_uref = gt_uref
        else:
            # Set Iref-specific GT
            self.gt_uref_per_iref[float(iref)] = gt_uref

    def set_gt_matching_policy(self, strict_match: bool = True, tolerance: float = 1e-9):
        """Configure the GT-to-target_i matching policy.

        Args:
            strict_match: If True, only match the same Iref (within
                *tolerance*); return None otherwise. If False, fall back
                to the nearest-neighbor Iref when no exact match exists.
            tolerance: Floating-point tolerance for strict matching.
        """
        self.strict_gt_iref_match = bool(strict_match)
        self.gt_iref_match_tolerance = float(max(tolerance, 0.0))

    def set_gt_from_csv_files(
        self,
        csv_dir: str,
        dataset_pattern: Optional[str] = None,
        tref: float = 60,
        ohref: float = 72,
    ) -> Dict[float, int]:
        """Load multi-Iref GT data from CSV files with strict error checking.

        Expected filename format (daily_regression_full_coverage.csv
        naming convention)::

            GxMx__gt_diagram__gxmx_xxx_load__iref_X__tref_Y__ohref_Z__
            daily_regression_full_coverage.csv

        Extracts *iref* from the filename, then loads voltage data from the
        CSV as ground truth. If any file fails to parse, a detailed error is
        logged and processing continues with the remaining files.

        Args:
            csv_dir: Directory containing GT CSV files.
            dataset_pattern: Optional dataset-name prefix to filter files
                (e.g. ``"G1M1_new"``).
            tref: Reference temperature (for logging / validation only).
            ohref: Reference operating hours (for logging / validation only).

        Returns:
            Dict mapping each Iref to the number of successfully loaded
            data points.

        Raises:
            ValueError: If *csv_dir* does not exist.
        """
        from pathlib import Path
        import re
        
        csv_path = Path(csv_dir)
        if not csv_path.exists():
            error_msg = f"GT CSV directory not found: {csv_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Find all _daily_regression_full_coverage.csv files
        pattern = "*daily_regression_full_coverage.csv"
        csv_files = list(csv_path.glob(pattern))
        
        if dataset_pattern:
            original_count = len(csv_files)
            csv_files = [f for f in csv_files if dataset_pattern in f.name]
            logger.debug(f"Filtered {original_count} files to {len(csv_files)} by pattern '{dataset_pattern}'")
        
        if not csv_files:
            warning_msg = f"No CSV files matching '{pattern}' found in {csv_dir}"
            if dataset_pattern:
                warning_msg += f" (with pattern '{dataset_pattern}')"
            logger.warning(warning_msg)
            return {}
        
        logger.info(f"Found {len(csv_files)} CSV file(s) to process")
        
        results: Dict[float, int] = {}
        failed_files: List[tuple] = []  # (filename, reason)
        
        for csv_file in csv_files:
            try:
                # Extract iref from the filename
                # Format: ...iref_X__tref_Y__ohref_Z__...
                match = re.search(r'iref_([0-9.]+)', csv_file.name)
                if not match:
                    reason = "Could not extract iref from filename pattern 'iref_[0-9.]+''"
                    logger.warning(f"[{csv_file.name}] {reason}")
                    failed_files.append((csv_file.name, reason))
                    continue
                
                iref = float(match.group(1))
                logger.debug(f"Extracted iref={iref} from {csv_file.name}")
                
                # Read CSV
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    reason = f"CSV read error: {type(e).__name__}: {str(e)}"
                    logger.error(f"[{csv_file.name}] {reason}")
                    failed_files.append((csv_file.name, reason))
                    continue
                
                if df.empty:
                    reason = "CSV file is empty"
                    logger.warning(f"[{csv_file.name}] {reason}")
                    failed_files.append((csv_file.name, reason))
                    continue
                
                logger.debug(f"Loaded CSV with shape {df.shape}, columns: {list(df.columns)}")
                
                # Find timestamp and voltage columns
                ts_col = None
                for col in ['timestamp', 'Timestamp', 'date', 'Date']:
                    if col in df.columns:
                        ts_col = col
                        break
                
                volt_col = None
                for col in ['gt_uref_regression', 'gt_uref_raw', 'gt_uref', 'voltage', 'Voltage']:
                    if col in df.columns:
                        volt_col = col
                        break
                
                if ts_col is None:
                    reason = f"No timestamp column found among {list(df.columns)}"
                    logger.error(f"[{csv_file.name}] {reason}")
                    failed_files.append((csv_file.name, reason))
                    continue
                
                if volt_col is None:
                    reason = f"No voltage column found among {list(df.columns)}"
                    logger.error(f"[{csv_file.name}] {reason}")
                    failed_files.append((csv_file.name, reason))
                    continue
                
                # Construct Series and validate
                try:
                    gt_series = pd.Series(
                        df[volt_col].values,
                        index=pd.to_datetime(df[ts_col])
                    )
                    initial_length = len(gt_series)
                    gt_series = gt_series.dropna()
                    dropped_count = initial_length - len(gt_series)
                    if dropped_count > 0:
                        logger.info(f"Dropped {dropped_count} nan values from {csv_file.name}")
                    
                    if gt_series.empty:
                        reason = "All values are NaN after dropping missing data"
                        logger.warning(f"[{csv_file.name}] {reason}")
                        failed_files.append((csv_file.name, reason))
                        continue
                    
                except Exception as e:
                    reason = f"Series construction error: {type(e).__name__}: {str(e)}"
                    logger.error(f"[{csv_file.name}] {reason}")
                    failed_files.append((csv_file.name, reason))
                    continue
                
                # Register GT
                self.set_ground_truth(gt_series, iref=iref)
                results[iref] = len(gt_series)
                logger.info(f"GT loaded @ Iref={iref}: {len(gt_series)} points from {csv_file.name}")
                
            except Exception as e:
                # Catch-all for unexpected errors
                reason = f"Unexpected error: {type(e).__name__}: {str(e)}"
                logger.exception(f"[{csv_file.name}] {reason}")
                failed_files.append((csv_file.name, reason))
        
        # Summarize results
        logger.info(f"\n=== GT CSV Loading Summary ===")
        logger.info(f"Successfully loaded: {len(results)} file(s)")
        for iref, n_points in sorted(results.items()):
            logger.info(f"  Iref={iref}: {n_points} points")
        
        if failed_files:
            logger.warning(f"Failed to load: {len(failed_files)} file(s)")
            for fname, reason in failed_files:
                logger.warning(f"  {fname}: {reason}")
        
        return results
    
    def _get_gt_for_iref(self, target_i: float) -> Optional[pd.Series]:
        """Retrieve the GT data corresponding to *target_i*.

        Strategy:
            1. If ``gt_uref_per_iref`` contains an exact (or near-exact)
               Iref match, use it.
            2. Otherwise, fall back to the global ``gt_uref`` (backward
               compatible).
            3. If neither is available, return ``None``.

        Args:
            target_i: Target current density.

        Returns:
            The matching GT Series, or None.
        """
        # Strategy 1: prefer exact Iref match (allow minor float tolerance)
        if self.gt_uref_per_iref:
            exact_candidates = [
                iref for iref in self.gt_uref_per_iref.keys()
                if abs(float(iref) - float(target_i)) <= self.gt_iref_match_tolerance
            ]
            if exact_candidates:
                exact_iref = min(exact_candidates, key=lambda x: abs(float(x) - float(target_i)))
                gt_exact = self.gt_uref_per_iref.get(exact_iref)
                if gt_exact is not None and not gt_exact.empty:
                    return gt_exact

            # Strict mode: no nearest-neighbor fallback
            if self.strict_gt_iref_match:
                return None

            # Non-strict mode: fall back to nearest-neighbor Iref
            closest_iref = min(self.gt_uref_per_iref.keys(), key=lambda x: abs(x - target_i))
            gt = self.gt_uref_per_iref.get(closest_iref)
            if gt is not None and not gt.empty:
                return gt
        
        # Strategy 2: fall back to global GT (backward compatible)
        if self.gt_uref is not None and not self.gt_uref.empty:
            return self.gt_uref
        
        return None
    
    def _is_surface_model(self, model):
        """Check whether *model* is a Surface model (requires special handling)."""
        if isinstance(model, str) and model.lower().startswith('surface'):
            return True
        if Urc1_Surface is not None and isinstance(model, Urc1_Surface):
            return True
        if Urc1_Surface_Stats is not None and isinstance(model, Urc1_Surface_Stats):
            return True
        return False
    
    
    def _ensure_degradation_results(self, model, target_i):
        """Ensure degradation_results are available.

        For bare Urc1 subclasses that lack a ``get_degradation_rate`` method,
        a linear degradation rate is derived from the Urc points in
        ``fitting_results_reliable``.
        """
        closest_i = min(model.Iref, key=lambda x: abs(x - target_i))
        
        # degradation_results already available; return directly
        if hasattr(model, 'degradation_results') and closest_i in model.degradation_results:
            return model.degradation_results[closest_i]
        
        # Try calling get_degradation_rate() (available in Urc wrapper)
        if hasattr(model, 'get_degradation_rate'):
            try:
                result = model.get_degradation_rate(closest_i)
                if result:
                    return result
            except:
                pass
        
        # Fallback: derive from fitting_results_reliable
        if hasattr(model, 'fitting_results_reliable'):
            df = model.fitting_results_reliable
        elif hasattr(model, 'Urc1_results'):
            df = model.Urc1_results
        else:
            return None
        
        if df.empty or 'calh' not in df.columns:
            return None
        
        urc_col = f"Urc_{closest_i}"
        if urc_col not in df.columns:
            return None
        
        # Linear regression
        mask = df['calh'].notna() & df[urc_col].notna()
        if mask.sum() < 2:
            return None
        
        slope, intercept, _, _, stderr = linregress(df.loc[mask, 'calh'], df.loc[mask, urc_col])
        
        # Construct a compatible result object
        from collections import namedtuple
        DegResult = namedtuple('DegResult', ['aging_uV_per_h', 'bol_V', 'uncertainty_uV2'])
        return DegResult(
            aging_uV_per_h=slope * 1e6,  # V/h -> uV/h
            bol_V=intercept,
            uncertainty_uV2=stderr**2
        )
    
    
    def _extract_reliable_data(self, model, target_i):
        """Unified data extraction handling all Urc-type parameter variations.

        Key adaptations:
            - Urc1_Iref: uses ``fitting_results_reliable_per_iref[closest_i]``.
            - Urc1_c3: handles dynamic intervals (no fixed ``len_interval``).
            - Surface: uses ``get_Urc_results()``.
            - Others: uses ``fitting_results_reliable``.
        """
        closest_i = min(model.Iref, key=lambda x: abs(x - target_i))
        
        # === Surface model special handling ===
        if self._is_surface_model(model):
            if hasattr(model, 'get_Urc_results'):
                try:
                    df = model.get_Urc_results(target_i)
                    if df is not None and not df.empty:
                        if 'timestamp' in df.columns:
                            df['Timestamp'] = df['timestamp']
                        elif 'date' in df.columns:
                            df['Timestamp'] = df['date']
                        else:
                            df['Timestamp'] = df.index
                        
                        if 'calh' not in df.columns:
                            df['calh'] = (df['Timestamp'] - model.installation_time) / pd.Timedelta(hours=1)
                        
                        df['Urc'] = df['Urc']
                        df['Urc_se'] = df['Urc_se']
                        df['Windows'] = 1
                        df['Cond'] = np.nan
                        df['Status'] = 'Surface'
                        df['Time_Span_Hours'] = getattr(model, 'len_interval', 2) * 24
                        df['End_Time'] = df['Timestamp'] + pd.Timedelta(days=getattr(model, 'len_interval', 2))
                        df['Start_Time'] = df['Timestamp']
                        df = df.dropna(subset=['Urc'])
                        return df
                except Exception as e:
                    logger.warning("Surface get_Urc_results failed: %s", e)
        
        # === Urc1_Iref special handling ===
        if hasattr(model, 'fitting_results_reliable_per_iref'):
            if closest_i in model.fitting_results_reliable_per_iref:
                df = model.fitting_results_reliable_per_iref[closest_i].copy()
            else:
                return None
        # === Standard path ===
        elif hasattr(model, 'Urc1_results') and model.Urc1_results is not None:
            df = model.Urc1_results.copy()
        elif hasattr(model, 'fitting_results_reliable'):
            df = model.fitting_results_reliable.copy()
        elif hasattr(model, 'Urc1_results_incl_unreliable'):
            raw_df = model.Urc1_results_incl_unreliable.copy()
            threshold = getattr(model, 'threshold', 1e6)
            query_str = f"cond < {threshold}"
            if 'quality' in raw_df.columns:
                query_str += " and quality == 'good'"
            try:
                df = raw_df.query(query_str).reset_index()
            except:
                return None
        else:
            return None
        
        if df is None or df.empty:
            return None
        
        # === Unify column names ===
        col_urc = f"Urc_{closest_i}"
        col_se = f"Urc_se_{closest_i}"
        
        if col_urc not in df.columns:
            return None
        
        # Timestamp column handling
        if 'date' in df.columns:
            df['Timestamp'] = df['date']
        elif 'timestamp' in df.columns:
            df['Timestamp'] = df['timestamp']
        else:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df['Timestamp'] = df['date'] if 'date' in df.columns else df.iloc[:, 0]
            else:
                df['Timestamp'] = df.index
        
        # === Handle Adaptive vs. Baseline field differences ===
        if 'num_windows_used' not in df.columns:
            # Baseline / Urc1_c3
            df['Windows'] = 1
            interval_hours = getattr(model, 'len_interval', 2) * 24
            df['Time_Span_Hours'] = interval_hours
            if 'adaptive_status' in df.columns:
                df['Status'] = df['adaptive_status']
            else:
                df['Status'] = 'Baseline'
        else:
            # Adaptive
            df['Windows'] = df['num_windows_used']
            df['Time_Span_Hours'] = df['time_span_hours']
            df['Status'] = df['adaptive_status']
        
        # Time-range calculation - Urc1_c3 uses dynamic intervals
        if hasattr(model, 'intervals') and model.intervals is not None and not model.intervals.empty:
            # === Urc1_c3: look up true end_time from the intervals table ===
            intervals_df = model.intervals
            # Build a start_time -> end_time mapping for fast lookup
            interval_map = intervals_df.set_index('start_time')['end_time'].to_dict()
            df['Start_Time'] = df['Timestamp']
            df['End_Time'] = df['Timestamp'].map(interval_map)
            # Use default fallback for unmatched records
            missing_mask = df['End_Time'].isna()
            if missing_mask.any():
                df.loc[missing_mask, 'End_Time'] = (
                    df.loc[missing_mask, 'Timestamp'] + pd.Timedelta(days=2)
                )
            # Update Time_Span_Hours to reflect actual interval lengths
            df['Time_Span_Hours'] = (
                (df['End_Time'] - df['Start_Time']) / pd.Timedelta(hours=1)
            )
        else:
            # === Urc1 / Adaptive: fixed or adaptive interval ===
            len_interval_days = getattr(model, 'len_interval', 2)
            df['End_Time'] = df['Timestamp'] + pd.Timedelta(days=len_interval_days)
            if 'time_span_hours' in df.columns:
                span = df['Time_Span_Hours'].fillna(len_interval_days * 24)
                df['Start_Time'] = df['End_Time'] - pd.to_timedelta(span, unit='h')
            else:
                df['Start_Time'] = df['Timestamp']
        
        # Numeric extraction
        df['Urc'] = df[col_urc]
        df['Urc_se'] = df[col_se]
        df['Cond'] = df['cond'] if 'cond' in df.columns else np.nan
        
        # Compute calh if not present
        if 'calh' not in df.columns:
            df['calh'] = (df['Timestamp'] - model.installation_time) / pd.Timedelta(hours=1)
        
        return df
    
    
    def _extract_all_cond_values(self, model, target_i: float) -> np.ndarray:
        """Extract all valid condition-number values from raw fitting_results.

        Key characteristics:
            - Draws from the full ``fitting_results`` (not the reliable-only
              subset), including intervals flagged as ``quality != "good"``.
            - Only excludes intervals where the fit itself failed (status
              contains ``"Error"`` or ``"Insufficient"``).
            - Useful for a comprehensive assessment of fitting stability
              (not limited to reliable fits).

        Compatibility:
            - Urc (wrapper): accesses ``urc_instance.fitting_results``.
            - Urc1_Iref: accesses per-Iref ``fitting_results``.
            - Urc1_c3: accesses ``fitting_results`` (dynamic intervals).
            - Surface models: returns an empty array (not applicable).

        Args:
            model: A Urc model instance.
            target_i: Target current density (used for Iref selection).

        Returns:
            np.ndarray of all valid condition-number values (NaN and inf
            removed).
        """
        # === Surface model: return empty array ===
        if self._is_surface_model(model):
            return np.array([])
        
        closest_i = min(model.Iref, key=lambda x: abs(x - target_i))
        all_conds_list = []
        
        # === Strategy 1: If the model is a Urc wrapper ===
        if hasattr(model, 'urc_instance'):
            """Urc wrapper case"""
            inner_model = model.urc_instance
            if hasattr(inner_model, 'fitting_results') and inner_model.fitting_results is not None:
                df = inner_model.fitting_results.copy()
                # Filter successfully-fitted rows (do not filter by quality)
                if 'status' in df.columns:
                    mask = ~df['status'].astype(str).str.contains('Error|Insufficient', na=False, regex=True)
                    df = df[mask]
                # Extract valid Cond values
                if 'cond' in df.columns:
                    conds = df['cond'].dropna().values
                    conds = conds[~np.isinf(conds)]
                    all_conds_list.extend(conds)
        
        # === Strategy 2: Urc1_Iref (per-Iref fitting_results) ===
        elif hasattr(model, 'fitting_results_per_iref'):
            """Urc1_Iref case: access fitting_results for the specified Iref"""
            if closest_i in model.fitting_results_per_iref:
                df = model.fitting_results_per_iref[closest_i].copy()
                if 'status' in df.columns:
                    mask = ~df['status'].astype(str).str.contains('Error|Insufficient', na=False, regex=True)
                    df = df[mask]
                if 'cond' in df.columns:
                    conds = df['cond'].dropna().values
                    conds = conds[~np.isinf(conds)]
                    all_conds_list.extend(conds)
        
        # === Strategy 3: standard path (fitting_results) ===
        elif hasattr(model, 'fitting_results') and model.fitting_results is not None:
            """Standard case for Urc1, Urc1_c3, etc."""
            df = model.fitting_results.copy()
            # Filter successfully-fitted rows
            if 'status' in df.columns:
                mask = ~df['status'].astype(str).str.contains('Error|Insufficient', na=False, regex=True)
                df = df[mask]
            # Extract valid Cond values
            if 'cond' in df.columns:
                conds = df['cond'].dropna().values
                conds = conds[~np.isinf(conds)]
                all_conds_list.extend(conds)
        
        # === Fallback strategy 4: from fitting_results_all (GPR, etc.) ===
        elif hasattr(model, 'fitting_results_all') and model.fitting_results_all is not None:
            df = model.fitting_results_all.copy()
            if 'status' in df.columns:
                mask = ~df['status'].astype(str).str.contains('Error|Insufficient', na=False, regex=True)
                df = df[mask]
            if 'cond' in df.columns:
                conds = df['cond'].dropna().values
                conds = conds[~np.isinf(conds)]
                all_conds_list.extend(conds)
        
        return np.array(all_conds_list) if all_conds_list else np.array([])

    def _extract_fitting_runtime_seconds(self, model) -> float:
        """Extract model fitting runtime in seconds when available."""
        direct_runtime = getattr(model, 'fitting_runtime_seconds', np.nan)
        if pd.notna(direct_runtime):
            return float(direct_runtime)

        if hasattr(model, 'timing_metrics'):
            timed_value = model.timing_metrics.get('model_fitting_seconds', np.nan)
            if pd.notna(timed_value):
                return float(timed_value)

        if hasattr(model, 'urc_instance'):
            nested_runtime = getattr(model.urc_instance, 'fitting_runtime_seconds', np.nan)
            if pd.notna(nested_runtime):
                return float(nested_runtime)

            nested_metrics = getattr(model.urc_instance, 'timing_metrics', {})
            timed_value = nested_metrics.get('model_fitting_seconds', np.nan)
            if pd.notna(timed_value):
                return float(timed_value)

        return np.nan
    
    
    def _calculate_gt_rmse(
        self,
        model,
        target_i: float,
    ) -> Optional[Dict[str, Union[float, int]]]:
        """Compute error metrics between model predictions and ground truth.

        Automatically selects the GT data matching *target_i*:
            - Uses Iref-specific GT if available.
            - Falls back to global GT otherwise.

        Args:
            model: A Urc model instance.
            target_i: Target current density.

        Returns:
            Dict with keys ``"GT RMSE (mV)"``, ``"GT MAE (mV)"``,
            ``"GT Valid Intervals"``.
        """
        # Select GT data matching target_i
        gt_uref = self._get_gt_for_iref(target_i)
        if gt_uref is None or gt_uref.empty:
            return {"GT RMSE (mV)": np.nan, "GT MAE (mV)": np.nan, "GT Valid Intervals": 0}
        
        closest_i = min(model.Iref, key=lambda x: abs(x - target_i))
        urc_col = f"Urc_{closest_i}"
        
        # Retrieve reliable data
        if hasattr(model, 'fitting_results_reliable'):
            df_rel = model.fitting_results_reliable.copy()
        elif hasattr(model, 'Urc1_results'):
            df_rel = model.Urc1_results.copy()
        elif hasattr(model, 'urc_instance') and hasattr(model.urc_instance, 'fitting_results_reliable'):
            df_rel = model.urc_instance.fitting_results_reliable.copy()
        else:
            return {"GT RMSE (mV)": np.nan, "GT MAE (mV)": np.nan, "GT Valid Intervals": 0}
        
        if df_rel.empty or urc_col not in df_rel.columns:
            return {"GT RMSE (mV)": np.nan, "GT MAE (mV)": np.nan, "GT Valid Intervals": 0}
        
        # Get interval length
        len_interval_days = getattr(model, 'len_interval', 2)
        interval_td = datetime.timedelta(days=len_interval_days)
        
        # Ensure GT has a DatetimeIndex (gt_uref obtained via _get_gt_for_iref)
        gt_uref = gt_uref.copy()
        if not isinstance(gt_uref.index, pd.DatetimeIndex):
            gt_uref.index = pd.to_datetime(gt_uref.index)
        
        # Iterate over each interval and compute errors
        urc_values = []
        utrue_values = []
        
        for idx, row in df_rel.iterrows():
            # Get interval start time
            # Priority: 'date' column → 'timestamp' column → DataFrame index
            interval_start = row.get('date', None)
            if interval_start is None:
                interval_start = row.get('timestamp', None)
            if interval_start is None:
                # GPR and some models store date as the DataFrame index
                # Handle MultiIndex case: if idx is a tuple, take the first element
                try:
                    if isinstance(idx, tuple):
                        interval_start = idx[0]  # MultiIndex: first element is usually the date
                    else:
                        interval_start = idx  # Single index
                except Exception:
                    continue
            if interval_start is None or pd.isna(interval_start):
                continue
            if not isinstance(interval_start, pd.Timestamp):
                try:
                    interval_start = pd.Timestamp(interval_start)
                except (TypeError, ValueError):
                    continue
            
            interval_end = interval_start + interval_td
            
            # Get Urc value
            urc_val = row.get(urc_col, np.nan)
            if pd.isna(urc_val):
                continue
            
            # Get GT values within this interval
            mask = (gt_uref.index >= interval_start) & (gt_uref.index <= interval_end)
            gt_in_interval = gt_uref.loc[mask]
            
            if len(gt_in_interval) == 0:
                continue
            
            # utrue = median of GT values in interval
            utrue = gt_in_interval.median()
            
            urc_values.append(urc_val)
            utrue_values.append(utrue)
        
        # Compute RMSE and MAE
        n_valid = len(urc_values)
        if n_valid < 1:
            return {"GT RMSE (mV)": np.nan, "GT MAE (mV)": np.nan, "GT Valid Intervals": 0}
        
        urc_arr = np.array(urc_values)
        utrue_arr = np.array(utrue_values)
        diff_mv = (urc_arr - utrue_arr) * 1000
        
        rmse_mv = np.sqrt(np.mean(diff_mv ** 2))
        mae_mv = np.mean(np.abs(diff_mv))
        
        return {
            "GT RMSE (mV)": round(rmse_mv, 3),
            "GT MAE (mV)": round(mae_mv, 3),
            "GT Valid Intervals": n_valid,
        }
    
    
    # =========================================================================
    # [Graphics] Visualization
    # =========================================================================
    
    def plot_interactive_trends(
        self,
        target_i=1.5,
        save=False,
        show_gt=False,
        output_dir="plots",
        uncertainty_style="bars",
        uncertainty_opacity=0.12,
        show_series_line=True,
        rate_precision=4,
    ):
        """[Plot 1] Multi-model Urc trend comparison - core interactive plot.

        Displays:
            - Urc scatter points for each model at the target current.
            - Optional uncertainty visualization via dense error bars,
              transparent band, or disabled uncertainty display.
            - Optional series line connecting model estimates.
            - Linear degradation fit lines.
            - Degradation rates shown directly in the legend.
            - (Optional) GT benchmark line.

        Args:
            target_i: Target current density.
            save: Whether to save the plot as an HTML file.
            show_gt: Whether to display the GT benchmark line (requires
                ``set_ground_truth()`` to have been called first).
            output_dir: Directory used when *save* is True.
            uncertainty_style: One of ``"bars"``, ``"band"``, or
                ``"none"``.
            uncertainty_opacity: Fill opacity for ``uncertainty_style='band'``.
            show_series_line: Whether to draw the model Urc series as a solid
                line in addition to markers.
            rate_precision: Decimal precision used for degradation-rate labels.
        """
        fig = go.Figure()

        def _rgba(color_value: str, alpha: float) -> str:
            alpha = float(min(max(alpha, 0.0), 1.0))
            if isinstance(color_value, str) and color_value.startswith('rgb('):
                return color_value.replace('rgb(', 'rgba(').replace(')', f', {alpha})')
            if isinstance(color_value, str) and color_value.startswith('rgba('):
                parts = color_value[color_value.find('(') + 1: color_value.rfind(')')].split(',')
                if len(parts) >= 3:
                    return f"rgba({parts[0].strip()}, {parts[1].strip()}, {parts[2].strip()}, {alpha})"
            if isinstance(color_value, str) and color_value.startswith('#') and len(color_value) == 7:
                r = int(color_value[1:3], 16)
                g = int(color_value[3:5], 16)
                b = int(color_value[5:7], 16)
                return f"rgba({r}, {g}, {b}, {alpha})"
            return f"rgba(100, 100, 100, {alpha})"
        
        for idx, (label, model) in enumerate(self.models.items()):
            df = self._extract_reliable_data(model, target_i)
            if df is None or df.empty:
                continue

            df = df.sort_values('Timestamp').copy()
            
            color = self.colors[idx % len(self.colors)]

            if uncertainty_style == 'band' and 'Urc_se' in df.columns:
                se = pd.to_numeric(df['Urc_se'], errors='coerce')
                valid_band = df['Urc'].notna() & se.notna()
                if valid_band.any():
                    x_band = df.loc[valid_band, 'Timestamp']
                    y_center = pd.to_numeric(df.loc[valid_band, 'Urc'], errors='coerce')
                    y_upper = y_center + se.loc[valid_band]
                    y_lower = y_center - se.loc[valid_band]

                    fig.add_trace(go.Scatter(
                        x=list(x_band) + list(x_band[::-1]),
                        y=list(y_upper) + list(y_lower[::-1]),
                        fill='toself',
                        fillcolor=_rgba(color, uncertainty_opacity),
                        line=dict(color='rgba(0,0,0,0)', width=0),
                        name=f"{label} Uncertainty",
                        legendgroup=f"{label}_uncertainty",
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            data_trace_kwargs = {}
            if uncertainty_style == 'bars' and 'Urc_se' in df.columns:
                data_trace_kwargs['error_y'] = dict(
                    type='data',
                    array=df['Urc_se'],
                    visible=True,
                    thickness=1,
                    width=0,
                )

            # Scatter points and optional center line - INDEPENDENTLY TOGGLEABLE
            fig.add_trace(go.Scatter(
                x=df['Timestamp'],
                y=df['Urc'],
                mode='lines+markers' if show_series_line else 'markers',
                name=f"{label} Data",
                legendgroup=f"{label}_data",  # Separate legend group for data
                showlegend=True,
                marker=dict(color=color, size=5, opacity=0.75 if show_series_line else 1.0),
                line=dict(color=color, width=2) if show_series_line else None,
                hovertemplate=f"<b>{label}</b><br>Urc: %{{y:.4f}} V<br>Time: %{{x}}<extra></extra>",
                **data_trace_kwargs,
            ))
            
            # Fit line (only when degradation_results are available) - INDEPENDENTLY TOGGLEABLE
            try:
                deg_res = self._ensure_degradation_results(model, target_i)
                if deg_res and not df.empty:
                    x_range = np.array([df['calh'].min(), df['calh'].max()])
                    y_range = (x_range * deg_res.aging_uV_per_h / 1e6) + deg_res.bol_V
                    
                    fig.add_trace(go.Scatter(
                        x=[df['Timestamp'].min(), df['Timestamp'].max()],
                        y=y_range,
                        mode='lines',
                        name=f"{label} Fit ({deg_res.aging_uV_per_h:.{int(rate_precision)}f} uV/h)",
                        legendgroup=f"{label}_fit",  # Separate legend group for fit line
                        showlegend=True,
                        line=dict(color=color, dash='dash', width=2),
                        hoverinfo='skip'
                    ))
            except Exception:
                # Not enough info to derive degradation rate (e.g. bare
                # Urc1 / Urc1_c3); skip the fit line.
                pass
        
        # Add GT benchmark (scatter + regression line)
        if show_gt:
            gt_for_plot = self._get_gt_for_iref(target_i)
            if gt_for_plot is not None and not gt_for_plot.empty:
                gt_for_plot = gt_for_plot.dropna()
                if not gt_for_plot.empty:
                    # GT scatter
                    fig.add_trace(go.Scatter(
                        x=gt_for_plot.index, y=gt_for_plot.values,
                        mode='markers',
                        name='Ground Truth Points',
                        marker=dict(size=6, color='darkgreen', opacity=0.7),
                        hovertemplate="<b>GT</b><br>Voltage: %{y:.4f} V<br>Time: %{x}<extra></extra>"
                    ))
                    
                    # GT regression line (linear fit)
                    if len(gt_for_plot) >= 2:
                        x_ts = gt_for_plot.index
                        y_v = gt_for_plot.values
                        
                        # Convert to hours for regression
                        x_hours = np.array([(ts - x_ts[0]).total_seconds() / 3600 for ts in x_ts])
                        
                        # Linear regression
                        slope_v_h, intercept_v = np.polyfit(x_hours, y_v, 1)
                        y_fit = slope_v_h * x_hours + intercept_v
                        
                        fig.add_trace(go.Scatter(
                            x=x_ts, y=y_fit,
                            mode='lines',
                            name=f"GT Regression ({slope_v_h * 1e6:.{int(rate_precision)}f} \u00b5V/h)",
                            line=dict(color='darkgreen', dash='dash', width=2),
                            hoverinfo='skip'
                        ))
        
        fig.update_layout(
            title=f"Multi-Model Urc Trend Comparison (@{target_i} A/cm\u00b2)",
            yaxis_title="Voltage [V]",
            xaxis_title="Time",
            height=600,
            template="plotly_white",
            hovermode='x unified'
        )
        if save:
            os.makedirs(output_dir, exist_ok=True)
            fig.write_html(os.path.join(output_dir, f"Urc_Trends_{target_i}A.html"))
        else:
            fig.show()

    def plot_interactive_trends_without_error(self, target_i=1.5, save=True, show_gt=False, rate_precision=4):
        """[Plot 1b] Multi-model Urc trend comparison without error bars.

        Displays:
            - Urc scatter points for each model at the target current.
            - Linear degradation fit lines.
            - Degradation rates shown directly in the legend.
            - (Optional) GT benchmark line.

        Args:
            target_i: Target current density.
            save: Whether to save the plot as an HTML file.
            show_gt: Whether to display the GT benchmark line (requires
                ``set_ground_truth()`` to have been called first).
            rate_precision: Decimal precision used for degradation-rate labels.
        """
        fig = go.Figure()
        
        for idx, (label, model) in enumerate(self.models.items()):
            df = self._extract_reliable_data(model, target_i)
            if df is None or df.empty:
                continue
            
            color = self.colors[idx % len(self.colors)]
            
            # Scatter points (no error bars)
            fig.add_trace(go.Scatter(
                x=df['Timestamp'], y=df['Urc'],
                mode='markers',
                name=f"{label} Data",
                legendgroup=label,
                marker=dict(color=color, size=5),
                hovertemplate=f"<b>{label}</b><br>Urc: %{{y:.4f}} V<br>Time: %{{x}}<extra></extra>"
            ))
            
            # Fit line (only when degradation_results are available)
            try:
                deg_res = self._ensure_degradation_results(model, target_i)
                if deg_res and not df.empty:
                    x_range = np.array([df['calh'].min(), df['calh'].max()])
                    y_range = (x_range * deg_res.aging_uV_per_h / 1e6) + deg_res.bol_V
                    
                    fig.add_trace(go.Scatter(
                        x=[df['Timestamp'].min(), df['Timestamp'].max()],
                        y=y_range,
                        mode='lines',
                        name=f"{label} Fit ({deg_res.aging_uV_per_h:.{int(rate_precision)}f} uV/h)",
                        legendgroup=label,
                        line=dict(color=color, dash='dash', width=2),
                        hoverinfo='skip'
                    ))
            except Exception:
                # Not enough info to derive degradation rate; skip fit line.
                pass
        
        # Add GT benchmark (scatter + regression line)
        if show_gt:
            gt_for_plot = self._get_gt_for_iref(target_i)
            if gt_for_plot is not None and not gt_for_plot.empty:
                gt_for_plot = gt_for_plot.dropna()
                if not gt_for_plot.empty:
                    # GT scatter
                    fig.add_trace(go.Scatter(
                        x=gt_for_plot.index, y=gt_for_plot.values,
                        mode='markers',
                        name='Ground Truth Points',
                        marker=dict(size=6, color='darkgreen', opacity=0.7),
                        hovertemplate="<b>GT</b><br>Voltage: %{y:.4f} V<br>Time: %{x}<extra></extra>"
                    ))
                    
                    # GT regression line (linear fit)
                    if len(gt_for_plot) >= 2:
                        x_ts = gt_for_plot.index
                        y_v = gt_for_plot.values
                        
                        # Convert to hours for regression
                        x_hours = np.array([(ts - x_ts[0]).total_seconds() / 3600 for ts in x_ts])
                        
                        # Linear regression
                        slope_v_h, intercept_v = np.polyfit(x_hours, y_v, 1)
                        y_fit = slope_v_h * x_hours + intercept_v
                        
                        fig.add_trace(go.Scatter(
                            x=x_ts, y=y_fit,
                            mode='lines',
                            name=f"GT Regression ({slope_v_h * 1e6:.{int(rate_precision)}f} µV/h)",
                            line=dict(color='darkgreen', dash='dash', width=2),
                            hoverinfo='skip'
                        ))
        
        fig.update_layout(
            title=f"Multi-Model Urc Trend Comparison (@{target_i} A/cm²)",
            yaxis_title="Voltage [V]",
            xaxis_title="Time",
            height=600,
            template="plotly_white",
            hovermode='x unified'
        )
        if save:
            os.makedirs("plots", exist_ok=True)
            fig.write_html(f"plots/Urc_Trends_no_error_{target_i}A.html")
        else:
            fig.show()
    
    
    def plot_coefficient_diagnostic(self, save=False, include_c6=True,
                                    output_dir: Optional[Union[str, Path]] = None):
        """[Plot 2] Multi-model coefficient comparison - physical parameter evolution.

        Displays:
            - Time evolution of 5 coefficients c1--c5 (common to all models).
            - Time evolution of c6 (only for models that include c6, e.g.
              cbrt/sigmoid; shown when *include_c6* is True).
            - Uncertainty bands (c +/- c_se).
            - Suitable for inspecting coefficient stability and drift.

        Args:
            save: Whether to save the plot as an HTML file.
            include_c6: Whether to append a c6 subplot (only effective for
                models with c6; default True).
            output_dir: Optional custom output directory. If not provided,
                figures are saved under plots/<model>/<dataset>/.
        """
        # Check whether any model contains a c6 column
        has_c6_any = False
        if include_c6:
            for label, model in self.models.items():
                df_chk = None
                if hasattr(model, 'urc_instance') and hasattr(model.urc_instance, 'fitting_results'):
                    df_chk = model.urc_instance.fitting_results
                elif hasattr(model, 'fitting_results_all'):
                    df_chk = model.fitting_results_all
                elif hasattr(model, 'fitting_results'):
                    df_chk = model.fitting_results
                if df_chk is not None and not df_chk.empty and 'c6' in df_chk.columns:
                    has_c6_any = True
                    break

        # Subplot titles: c1--c5 fixed; c6 appended as needed
        coeffs = ['c1', 'c2', 'c3', 'c4', 'c5']
        if has_c6_any:
            coeffs.append('c6')

        n_rows = len(coeffs)
        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=[f"Coefficient {c.upper()}" for c in coeffs]
        )
        
        for m_idx, (label, model) in enumerate(self.models.items()):
            # Get fitting_results
            # Prefer fitting_results_all (Urc1_GPR stores GPR-smoothed
            # coefficients here); fall back to fitting_results (Baseline).
            df = None
            if hasattr(model, 'urc_instance') and hasattr(model.urc_instance, 'fitting_results'):
                df = model.urc_instance.fitting_results.copy()
            elif hasattr(model, 'fitting_results_all'):
                df = model.fitting_results_all.copy()
            elif hasattr(model, 'fitting_results'):
                df = model.fitting_results.copy()
            else:
                logger.warning("'%s' has no fitting_results", label)
                continue
            
            if df is None or df.empty:
                continue
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            
            # Time column
            time_col = None
            for col_candidate in ['date', 'Timestamp', 'timestamp']:
                if col_candidate in df.columns:
                    time_col = col_candidate
                    break
            
            if time_col is None:
                logger.warning("'%s' has no time column", label)
                continue
            
            # Color handling
            color = self.colors[m_idx % len(self.colors)]
            if color.startswith('rgb'):
                rgba_fill = color.replace('rgb', 'rgba').replace(')', ', 0.2)')
            elif color.startswith('#'):
                hex_color = color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                rgba_fill = f'rgba({r}, {g}, {b}, 0.2)'
            else:
                rgba_fill = 'rgba(100, 100, 100, 0.2)'
            
            # Plot each coefficient
            for c_idx, c_name in enumerate(coeffs):
                # Prefer GPR-smoothed column (e.g. c1_gpr); fall back to raw (c1)
                actual_col = f"{c_name}_gpr" if f"{c_name}_gpr" in df.columns else c_name
                actual_se = f"{actual_col}_se"

                if actual_col not in df.columns:
                    continue
                
                temp_df = df.dropna(subset=[actual_col]).sort_values(time_col)
                if temp_df.empty:
                    continue
                
                if actual_se in df.columns:
                    # SE available: draw uncertainty band
                    temp_df = temp_df.dropna(subset=[actual_se])
                    if not temp_df.empty:
                        # Uncertainty band fill
                        fig.add_trace(go.Scatter(
                            x=list(temp_df[time_col]) + list(temp_df[time_col])[::-1],
                            y=list(temp_df[actual_col] + temp_df[actual_se]) + list(temp_df[actual_col] - temp_df[actual_se])[::-1],
                            fill='toself',
                            fillcolor=rgba_fill,
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo='skip',
                            showlegend=False,
                            legendgroup=label
                        ), row=c_idx+1, col=1)
                
                # Coefficient line
                fig.add_trace(go.Scatter(
                    x=temp_df[time_col],
                    y=temp_df[actual_col],
                    mode='lines+markers',
                    name=label if c_idx == 0 else "",
                    legendgroup=label,
                    showlegend=(c_idx == 0),
                    marker=dict(size=4, color=color),
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "Time: %{x}<br>" +
                        f"{actual_col}: %{{y:.4e}}<extra></extra>"
                    )
                ), row=c_idx+1, col=1)
        
        fig.update_layout(
            height=200 + 220 * n_rows,
            title_text=f"Model Coefficients Evolution (c1~c{'6' if has_c6_any else '5'})",
            template="plotly_white",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        if save:
            title_suffix = "c1c6" if has_c6_any else "c1c5"
            self._save_figure_html(
                fig,
                f"Coefficients_Diagnostic_{title_suffix}.html",
                output_dir=output_dir,
            )
        else:
            fig.show()
    
    
    def plot_coverage_gantt(self, target_i=1.5, save=False,
                            output_dir: Optional[Union[str, Path]] = None):
        """
        [Plot 3] (Optional) Data coverage Gantt chart for adaptive/expanding-window methods.

        Scenario: show the window coverage range of adaptive methods.
        """
        # Filter out Surface models
        filtered_models = {
            k: v for k, v in self.models.items()
            if not self._is_surface_model(v)
        }
        
        if not filtered_models:
            logger.warning("No viable models for coverage plot (Surface excluded)")
            return
        
        all_dfs = []
        for label, model in filtered_models.items():
            df = self._extract_reliable_data(model, target_i)
            if df is not None and not df.empty:
                df['Method'] = label
                all_dfs.append(df)
        
        if not all_dfs:
            return
        
        df_merged = pd.concat(all_dfs)
        
        fig = px.timeline(
            df_merged,
            x_start="Start_Time",
            x_end="End_Time",
            y="Method",
            color="Windows",
            hover_data=["Timestamp", "Urc", "Urc_se", "Cond"],
            title=f"Effective Data Coverage Intervals (@{target_i} A/cm²)",
            height=400 + 50*len(filtered_models),
            color_continuous_scale="Viridis"
        )
        
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Timeline", template="plotly_white")
        if save:
            self._save_figure_html(
                fig,
                f"Coverage_Gantt_{target_i}A.html",
                output_dir=output_dir,
            )
        else:
            fig.show()
    
    
    def plot_fit_quality(self, save=False,
                         output_dir: Optional[Union[str, Path]] = None):
        """
        [Plot] Fitting-quality comparison via RMSE and R2 violin plots.

        Shows per-interval RMSE and R2 distributions for all models.
        Lower RMSE and higher R2 indicate better fit quality.
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["<b>RMSE Distribution</b>", "<b>R² Distribution</b>"],
            horizontal_spacing=0.12,
        )

        any_trace = False
        for m_idx, (label, model) in enumerate(self.models.items()):
            # Get fitting_results_reliable (compatible with all Urc variants).
            rel = None
            if hasattr(model, 'fitting_results_reliable'):
                rel = model.fitting_results_reliable
            elif hasattr(model, 'Urc1_results'):
                rel = model.Urc1_results
            elif hasattr(model, 'urc_instance') and hasattr(model.urc_instance, 'fitting_results_reliable'):
                rel = model.urc_instance.fitting_results_reliable

            if rel is None or rel.empty:
                logger.warning("'%s' has no fitting_results_reliable, skipping.", label)
                continue

            color = self.colors[m_idx % len(self.colors)]

            for col_key, col_idx, y_scale in [('RMSE', 1, 1000), ('R2', 2, 1)]:
                if col_key not in rel.columns:
                    continue
                vals = rel[col_key].dropna() * y_scale
                fig.add_trace(go.Violin(
                    y=vals,
                    name=label,
                    marker_color=color,
                    box_visible=True,
                    meanline_visible=True,
                    legendgroup=label,
                    showlegend=(col_key == 'RMSE'),
                ), row=1, col=col_idx)
                any_trace = True

        if not any_trace:
            logger.warning("No data available for fit quality plot.")
            return

        fig.update_yaxes(title_text="RMSE [mV]", row=1, col=1)
        fig.update_yaxes(title_text="R²", row=1, col=2)
        fig.update_layout(
            title="Fit Quality: RMSE & R² Distributions",
            height=480,
            template="plotly_white",
            violinmode="group",
        )
        if save:
            self._save_figure_html(
                fig,
                "Fit_Quality.html",
                output_dir=output_dir,
            )
        else:
            fig.show()


    # =========================================================================
    # [Metrics] Numerical Evaluation
    # =========================================================================
    
    def evaluate_stability(
        self,
        model_name: str,
        i_target: float,
        outlier_threshold_method: Union[str, float] = '2rmse',
        include_gt_metrics: bool = False,
        include_all_cond_metrics: bool = False
    ) -> Optional[Dict[str, Union[float, int]]]:
        """
        Compute integrated stability metrics for one model.

        Returns 9+ metrics:
        - Data Points (n)
        - RMSE (mV)
        - Slope Sigma (uV/h)
        - Mono (Rank) [macro monotonicity]
        - Outlier Count / %
        - Max Residual (mV)
        - Mean SE (mV)
        - Cond (Median log10, P95 log10)
        - (optional) GT RMSE (mV), GT MAE (mV), GT Valid Intervals
        
        Args:
            model_name: Model label in the comparator registry.
            i_target: Target current density.
            outlier_threshold_method: Outlier threshold strategy.
            include_gt_metrics: Whether to include GT alignment metrics.
            include_all_cond_metrics: Whether to compute Cond statistics
                from all valid fitted intervals or only reliable intervals.
        """
        if model_name not in self.models:
            logger.warning("Model '%s' not found", model_name)
            return None
        
        model = self.models[model_name]
        df = self._extract_reliable_data(model, i_target)
        
        if df is None or df.empty:
            logger.warning("No data for %s @ %s A/cm2", model_name, i_target)
            return None
        
        closest_i = min(model.Iref, key=lambda x: abs(x - i_target))
        deg_res = self._ensure_degradation_results(model, i_target)
        
        if deg_res is None:
            logger.warning("No degradation result for %s @ %s", model_name, closest_i)
            return None
        
        # Core calculation: pre-filter NaN/inf to avoid metric errors.
        slope_v_per_h = deg_res.aging_uV_per_h / 1e6
        intercept_v = deg_res.bol_V
        y_true_raw = pd.to_numeric(df['Urc'], errors='coerce').to_numpy()
        calh_raw = pd.to_numeric(df['calh'], errors='coerce').to_numpy()
        y_pred_raw = calh_raw * slope_v_per_h + intercept_v

        finite_mask = np.isfinite(y_true_raw) & np.isfinite(y_pred_raw)
        y_true = y_true_raw[finite_mask]
        y_pred = y_pred_raw[finite_mask]
        residuals_mv = (y_true - y_pred) * 1000

        n = len(y_true)
        if n < 2:
            logger.warning(
                "Insufficient finite points for %s @ %s A/cm2",
                model_name,
                closest_i,
            )
            return None

        mse = mean_squared_error(y_true, y_pred)
        rmse_mv = np.sqrt(mse) * 1000
        
        slope_sigma_uv_h = np.sqrt(deg_res.uncertainty_uV2)
        mono_rank = calculate_rank_monotonicity(y_true)
        # mono_sign = calculate_sign_monotonicity(y_true)
        
        # Outlier
        if outlier_threshold_method == '2rmse':
            limit_mv = 2 * rmse_mv
        elif outlier_threshold_method == '3rmse':
            limit_mv = 3 * rmse_mv
        elif isinstance(outlier_threshold_method, (int, float)):
            limit_mv = float(outlier_threshold_method)
        else:
            limit_mv = 2 * rmse_mv
        
        outlier_count = (np.abs(residuals_mv) > limit_mv).sum()
        outlier_pct = (outlier_count / n * 100) if n > 0 else 0
        max_res_mv = np.abs(residuals_mv).max()
        mean_se_mv = pd.to_numeric(df.loc[finite_mask, 'Urc_se'], errors='coerce').mean() * 1000
        
        # Condition-number scope switch:
        # - include_all_cond_metrics=True  -> all valid fitted intervals
        # - include_all_cond_metrics=False -> reliable intervals only
        if include_all_cond_metrics:
            cond_values = self._extract_all_cond_values(model, i_target)
            cond_scope = "all_fitted"
        else:
            cond_values = pd.to_numeric(df['Cond'], errors='coerce').to_numpy()
            cond_values = cond_values[np.isfinite(cond_values)]
            cond_scope = "reliable_only"

        if len(cond_values) > 0:
            log10_conds = np.log10(cond_values)
            cond_median_log10 = np.median(log10_conds)
            cond_p95_log10 = np.percentile(log10_conds, 95)
            cond_count = len(cond_values)
        else:
            cond_median_log10 = np.nan
            cond_p95_log10 = np.nan
            cond_count = 0
        
        metrics = {
            "Model Name": model_name,
            "Target Current (A/cm2)": i_target,
            "Fitting Time (s)": round(self._extract_fitting_runtime_seconds(model), 3),
            "Data Points (n)": n,
            "Degradation Rate (uV/h)": round(deg_res.aging_uV_per_h, 6),
            "RMSE (mV)": round(rmse_mv, 3),
            "Slope Sigma (uV/h)": round(slope_sigma_uv_h, 3),
            "Mono (Rank) [0-1]": round(mono_rank, 3),
            # "Mono (Sign) [0-1]": round(mono_sign, 3),
            "Outlier Count": outlier_count,
            "Outlier (%)": round(outlier_pct, 2),
            "Max Residual (mV)": round(max_res_mv, 3),
            "Mean SE (mV)": round(mean_se_mv, 3),
            "Cond Median log10": round(cond_median_log10, 3),
            "Cond P95 log10": round(cond_p95_log10, 3),
            "Cond Count (Used Scope)": cond_count,
            "Cond Scope": cond_scope,
        }
        
        # Add GT alignment metrics.
        if include_gt_metrics:
            gt_metrics = self._calculate_gt_rmse(model, i_target)
            metrics.update(gt_metrics)
        
        return metrics
    
    
    def compare_all(
        self,
        i_target: float,
        outlier_threshold_method: Union[str, float] = '2rmse',
        include_gt_metrics: bool = False,
        include_all_cond_metrics: bool = False
    ) -> pd.DataFrame:
        """
        Compare all registered models and return a standardized DataFrame.
        
        Args:
            i_target: Target current density.
            outlier_threshold_method: Outlier threshold strategy.
            include_gt_metrics: Whether to include GT metrics.
            include_all_cond_metrics: Whether to use all valid fitted
                intervals for condition-number statistics.
        """
        results = []
        
        for name in self.models.keys():
            metrics = self.evaluate_stability(
                name,
                i_target,
                outlier_threshold_method,
                include_gt_metrics=include_gt_metrics,
                include_all_cond_metrics=include_all_cond_metrics,
            )
            if metrics:
                results.append(metrics)
        
        if not results:
            return pd.DataFrame()
        
        df_res = pd.DataFrame(results)
        self._cached_metrics[i_target] = df_res
        
        return df_res

    def get_degradation_rate_table(
        self,
        i_targets: List[float],
        ref_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build a cross-reference degradation-rate comparison table."""
        rows: List[Dict[str, Union[str, float]]] = []

        if ref_names is None:
            ref_names = [f"Ref_{idx + 1}" for idx in range(len(i_targets))]

        for ref_name, i_target in zip(ref_names, i_targets):
            df_metrics = self.compare_all(i_target=i_target)
            if df_metrics.empty:
                continue

            for _, row in df_metrics.iterrows():
                rows.append({
                    "Reference": ref_name,
                    "Target Current (A/cm2)": i_target,
                    "Model Name": row["Model Name"],
                    "Degradation Rate (uV/h)": row.get("Degradation Rate (uV/h)", np.nan),
                    "Slope Sigma (uV/h)": row.get("Slope Sigma (uV/h)", np.nan),
                })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)
    
    
    def get_summary_table(
        self,
        i_targets: List[float] = [0.6, 1.0, 1.5],
        include_gt_metrics: bool = False,
        include_all_cond_metrics: bool = False
    ) -> pd.DataFrame:
        """
        Return summary metrics across multiple target currents.
        
        Args:
            i_targets: List of target current densities.
            include_gt_metrics: Whether to include GT metrics.
            include_all_cond_metrics: Whether to use all valid fitted
                intervals for condition-number statistics.
        """
        all_results = []
        
        for i_target in i_targets:
            df = self.compare_all(
                i_target,
                include_gt_metrics=include_gt_metrics,
                include_all_cond_metrics=include_all_cond_metrics,
            )
            if not df.empty:
                all_results.append(df)
        
        if not all_results:
            return pd.DataFrame()
        
        return pd.concat(all_results, ignore_index=True)
    
    
    def print_comparison_report(
        self,
        i_target: float = 1.5,
        include_gt_metrics: bool = False,
        include_all_cond_metrics: bool = False,
    ):
        """
        Print a formatted model-comparison report.
        
        Args:
            i_target: Target current density.
            include_gt_metrics: Whether to print GT metrics.
            include_all_cond_metrics: Whether to use all valid fitted
                intervals for condition-number statistics.
        """
        df = self.compare_all(
            i_target,
            include_gt_metrics=include_gt_metrics,
            include_all_cond_metrics=include_all_cond_metrics,
        )
        
        if df.empty:
            print("No data available.")
            return
        
        print("=" * 120)
        print(f"  MODEL COMPARISON REPORT @ {i_target} A/cm²")
        print("=" * 120)
        
        for _, row in df.iterrows():
            print(f"\n📊 {row['Model Name']}")
            print("-" * 60)
            print(f"  Data Points:      {row['Data Points (n)']}")
            print(f"  Deg Rate:         {row['Degradation Rate (uV/h)']:.6f} μV/h")
            print(f"  RMSE:             {row['RMSE (mV)']:.3f} mV")
            print(f"  Slope Sigma:      {row['Slope Sigma (uV/h)']:.3f} μV/h")
            print(f"  Mono (Rank):      {row['Mono (Rank) [0-1]']:.3f}")
            print(f"  Outliers:         {row['Outlier Count']} ({row['Outlier (%)']:.1f}%)")
            print(f"  Max Residual:     {row['Max Residual (mV)']:.3f} mV")
            print(f"  Mean SE:          {row['Mean SE (mV)']:.3f} mV")
            print(f"  Cond (Median):    10^{row['Cond Median log10']:.1f}")
            print(f"  Cond Scope:       {row['Cond Scope']} (n={int(row['Cond Count (Used Scope)'])})")
            
            # Add GT metrics when available.
            if include_gt_metrics:
                if 'GT RMSE (mV)' in row:
                    print(f"  GT RMSE:          {row['GT RMSE (mV)']:.3f} mV (n={row['GT Valid Intervals']:.0f})")
                    print(f"  GT MAE:           {row['GT MAE (mV)']:.3f} mV")
        
        print("\n" + "=" * 120)
        best_rmse = df.loc[df['RMSE (mV)'].idxmin(), 'Model Name']
        best_mono = df.loc[df['Mono (Rank) [0-1]'].idxmax(), 'Model Name']
        print(f"🏆 Best RMSE (vs model data):    {best_rmse}")
        print(f"🏆 Best Monotonicity:             {best_mono}")
        
        if include_gt_metrics and 'GT RMSE (mV)' in df.columns:
            best_gt = df.loc[df['GT RMSE (mV)'].idxmin(), 'Model Name']
            print(f"🏆 Best GT RMSE (vs ground truth): {best_gt}")
        
        print("=" * 120)
