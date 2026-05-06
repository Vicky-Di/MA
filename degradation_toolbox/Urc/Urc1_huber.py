"""Huber robust regression baseline for Urc1.

Replaces OLS (curve_fit) in each sliding-window interval with
scipy.optimize.least_squares using Huber loss, making per-interval
voltage model fitting robust to outliers while keeping the same
linear voltage model:

    U_scaled = c1*I_s + c2*(I*T)_s + c3*log(h)_s + c4*I^2_s + c5

Supports 4 modes via (epsilon, alpha) combinations:
    Mode         | epsilon     | alpha
    -------------|-------------|------
    OLS          | (use Urc1)  | --
    Huber only   | 1.35        | 0.0
    Ridge only   | 1e6 (huge)  | 0.1
    Huber+Ridge  | 1.35        | 0.1
"""

import datetime
import logging
import math
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

from degradation_toolbox.Urc.helpers import (
    calculate_degradation_rate_and_uncertainty,
    degradation_result,
)
from degradation_toolbox.Urc.Urc1 import OCV, Urc1, func


logger = logging.getLogger(__name__)


class Urc1BaseHuber(Urc1):
    """Robust Urc1 baseline using ``scipy.optimize.least_squares``.

    Inherits everything from Urc1; only overrides model_fitting().
    Preprocessing, Urc calculation, quality checks, and SE calculation
    are all inherited unchanged.

    Parameters
    ----------
    epsilon : float, default 1.35
        Huber threshold. Controls robustness vs efficiency trade-off.
        - 1.35 (sklearn default): ~95% asymptotic efficiency for Gaussian data
        - 1.0 : more aggressive outlier rejection
        - 1e6 : effectively disables Huber -> Ridge-only mode
    alpha : float, default 0.0
        L2 (Ridge) regularization strength on c1-c4 (intercept c5 excluded).
        - 0.0 : pure Huber, no regularization
        - >0  : adds Ridge penalty alpha*||w||^2 to the Huber loss
    max_iter : int, default 1000
        Maximum iterations for the HuberRegressor optimizer.
    **kwargs
        All other parameters forwarded to Urc1 (data, name, Iref, Tref, etc.).
    """

    _PARAMETER_COUNT = 5
    _INTERCEPT_INDEX = 4

    def __init__(
        self,
        data: pd.DataFrame,
        epsilon: float = 1.35,
        alpha: float = 0.0,
        max_iter: int = 1000,
        **kwargs: Any,
    ) -> None:
        # Store Huber parameters BEFORE super().__init__() -- it calls model_fitting()
        self.epsilon = epsilon
        self.huber_alpha = alpha
        self.huber_max_iter = max_iter
        self.run_in = kwargs.get("run_in", 0)
        self.regression_method = kwargs.get("regression_method", "orth")
        self.fix_point = kwargs.get("fix_point", None)
        self.huber_config = {
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "max_iter": int(max_iter),
            "run_in": float(self.run_in),
            "regression_method": self.regression_method,
            "fix_point": self.fix_point,
        }

        # Determine mode label for logging
        self._mode_label = self._build_mode_label(epsilon=epsilon, alpha=alpha)

        # Parent constructor: preprocess -> _init_scalers -> model_fitting -> Urc_calc -> checks
        super().__init__(data, **kwargs)

        # Comparator-friendly aliases and metadata.
        self._refresh_comparator_aliases()

        print(f"  [{self._mode_label}] epsilon={self.epsilon}, alpha={self.huber_alpha}")

    @staticmethod
    def _build_mode_label(epsilon: float, alpha: float) -> str:
        """Create a human-readable label for the active robust-fit mode."""
        has_huber = epsilon < 1e4
        has_ridge = alpha > 0
        if has_huber and has_ridge:
            return "Huber+Ridge"
        if has_huber:
            return "Huber"
        if has_ridge:
            return "Ridge"
        return "OLS-like"

    @classmethod
    def _get_parameter_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Match the baseline Urc1 coefficient bounds for fair comparison."""
        lower = np.array([0.0, -np.inf, -np.inf, -np.inf, -np.inf], dtype=float)
        upper = np.array([np.inf, 0.0, np.inf, np.inf, np.inf], dtype=float)
        return lower, upper

    @classmethod
    def _get_ridge_penalty_mask(cls) -> np.ndarray:
        """Regularize only c1-c4; keep intercept c5 unpenalized."""
        mask = np.ones(cls._PARAMETER_COUNT, dtype=float)
        mask[cls._INTERCEPT_INDEX] = 0.0
        return mask

    def _refresh_comparator_aliases(self) -> None:
        """Synchronize aliases expected by comparison and evaluation tools."""
        self.model_family = "Urc1BaseHuber"
        self.model_label = self._mode_label
        self.model_results = self.fitting_results.copy()
        self.Urc1_results_incl_unreliable = self.fitting_results.reset_index()
        self.Urc1_results = self.fitting_results_reliable.copy()
        self.Urc_results = pd.DataFrame()
        self.degradation_results = {}
        self.degradation_results_Urc1 = {}
        self.degradation_results_by_ref = {}

        for i_ref in self.Iref:
            result = self.get_degradation_rate(i_ref)
            if result is None:
                logger.debug("No degradation result available for Iref=%s", i_ref)

        for ref_name in self.ref_name_to_i:
            result = self.get_degradation_rate(ref_name)
            if result is None:
                logger.debug("No degradation result available for reference=%s", ref_name)

    def get_Urc_results(self, i: float) -> pd.DataFrame:
        """Return reliable Urc points in a comparator-compatible schema."""
        resolved = self._resolve_reference_input(i)
        if resolved is None:
            return pd.DataFrame()
        closest_i, _, _, ref_name = resolved

        if self.Urc1_results is None or self.Urc1_results.empty:
            return pd.DataFrame()

        col_urc = f"Urc_{closest_i}"
        col_se = f"Urc_se_{closest_i}"
        if col_urc not in self.Urc1_results.columns or col_se not in self.Urc1_results.columns:
            return pd.DataFrame()

        df = self.Urc1_results.copy()
        if "date" not in df.columns:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            elif df.index.name == "date":
                df = df.reset_index()
            else:
                df = df.reset_index().rename(columns={df.columns[0]: "date"})

        if "date" not in df.columns:
            return pd.DataFrame()

        to_save = df[["date", "calh", col_urc, col_se]].copy()
        to_save["method"] = self._mode_label
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

    def get_degradation_rate(self, i: float) -> Optional[degradation_result]:
        """Compute degradation slope and cache in degradation_results."""
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

        time = urc_results_excl_run_in["calh"].to_numpy(dtype=float)
        urc = urc_results_excl_run_in["Urc"].to_numpy(dtype=float)
        urc_se = urc_results_excl_run_in["Urc_se"].to_numpy(dtype=float)

        finite_se = np.isfinite(urc_se) & (urc_se > 0)
        urc_se_for_fit = urc_se if finite_se.sum() >= 2 else None

        try:
            result = calculate_degradation_rate_and_uncertainty(
                time=time,
                Urc=urc,
                Urc_se=urc_se_for_fit,
                fix_point=self.fix_point,
                regression_method=self.regression_method,
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

    def model_fitting(self) -> pd.DataFrame:
        """Per-interval Huber + Ridge regression via scipy least_squares.

        For each time window:
        1. Prepare scaled X, y via parent's _prepare_matrices()
        2. Define residual function with Ridge penalty
        3. Optimize with least_squares(loss='huber')
        4. Compute covariance from Jacobian matrix
        5. Record R-squared, RMSE, condition number, outlier stats
        """
        installation_day = self.installation_time.date()
        end_day = max(self.data.index).date()
        num_days = (end_day - installation_day).days
        num_intervals = math.floor(num_days / self.slide)

        # Cache max index to avoid repeated O(n) scan on 300K+ rows
        max_data_index = self.data.index.max()

        # --- Build output DataFrame ---
        dates = [self.installation_time + datetime.timedelta(days=i * self.slide)
                 for i in range(num_intervals + 5)]
        index = pd.MultiIndex.from_product([dates, [self.name]], names=['date', 'name'])
        results = pd.DataFrame(index=index)

        # Numeric columns
        num_cols = [
            "day_since_install", "calh", "R2", "RMSE", "cond", "sigma2",
            "c1", "c2", "c3", "c4", "c5",
            "c1_se", "c2_se", "c3_se", "c4_se", "c5_se",
            "outlier_pct",
        ]
        for c in num_cols:
            results[c] = np.nan

        # Object columns
        obj_cols = ["XTX", "quality", "status"]
        for c in obj_cols:
            results[c] = pd.Series([None] * len(results), index=index, dtype=object)

        num_no_data = 0
        num_fit_failed = 0
        lower_bounds, upper_bounds = self._get_parameter_bounds()
        ridge_penalty_mask = self._get_ridge_penalty_mask()
        ts = self.data.index.to_numpy()

        # Residual function for least_squares
        def residuals_with_ridge(
            params: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            alpha: float,
        ) -> np.ndarray:
            """
            Residual vector = [model_residuals; ridge_penalty]

            Args:
                params: [c1, c2, c3, c4, c5] model coefficients, shape (5,).
                X: Feature matrix (excluding intercept), shape (n, 4).
                y: Target scaled voltage, shape (n,).
                alpha: Ridge regularization strength. L2 penalty on coefficients.

            Returns:
                Concatenation of data residuals and Ridge penalty terms,
                shape (n + 5,). least_squares minimizes sum(residuals^2),
                equivalent to the standard Ridge objective with penalty
                alpha*||w||^2.
            """
            # Model prediction: c1*x1 + c2*x2 + c3*x3 + c4*x4 + c5
            X_aug = np.c_[X, np.ones(len(X))]
            y_pred = X_aug @ params
            data_res = y_pred - y
            
            # Ridge penalty: sqrt(alpha) * params
            # Minimizing sum(data_res^2 + ridge_res^2) yields alpha*||params||^2
            ridge_res = np.sqrt(alpha) * params * ridge_penalty_mask
            
            return np.concatenate([data_res, ridge_res])

        for interval in range(num_intervals + 2):
            current_day = self.installation_time + datetime.timedelta(days=interval * self.slide)
            interval_end = current_day + datetime.timedelta(days=self.len_interval)

            # Stop when we run past the data range
            if interval_end > max_data_index + datetime.timedelta(days=1):
                break

            idx = (current_day, self.name)
            results.loc[idx, "day_since_install"] = (current_day - self.installation_time).days
            results.loc[idx, "calh"] = (current_day - self.installation_time) / np.timedelta64(1, "h")

            # Slice data for this interval (faster than query inside large loops)
            left = np.searchsorted(ts, np.datetime64(current_day), side="left")
            right = np.searchsorted(ts, np.datetime64(interval_end), side="right")
            data_interval = self.data.iloc[left:right]

            if len(data_interval) < self.min_num_data_required_for_fit:
                num_no_data += 1
                results.loc[idx, "status"] = (
                    f"Insufficient Data ({len(data_interval)} < {self.min_num_data_required_for_fit})"
                )
                continue

            try:
                # Step 1: Shared feature engineering & scaling
                X, y = self._prepare_matrices(data_interval)

                # Step 2: Initial guess (OLS solution or zeros)
                X_aug = np.c_[X, np.ones(len(X))]
                try:
                    initial_guess = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    initial_guess = np.zeros(self._PARAMETER_COUNT)
                except ValueError:
                    initial_guess = np.zeros(self._PARAMETER_COUNT)

                initial_guess = np.clip(initial_guess, lower_bounds, upper_bounds)

                # Step 3: Optimize with least_squares
                # f_scale controls Huber threshold (approx epsilon * sigma)
                # Compute data-adaptive scale: MAD-based initial estimate
                X_aug_init = np.c_[X, np.ones(len(X))]
                initial_residuals = y - (X_aug_init @ initial_guess)
                scale_init = np.median(np.abs(initial_residuals)) / 0.6745
                if scale_init < 1e-10:
                    scale_init = 1e-10
                f_scale_adapt = self.epsilon * scale_init  # Adaptive f_scale
                
                res_lsq = least_squares(
                    residuals_with_ridge,
                    initial_guess,
                    args=(X, y, self.huber_alpha),
                    loss='huber',
                    f_scale=f_scale_adapt,  # Data-adaptive Huber threshold
                    max_nfev=self.huber_max_iter,
                    bounds=(lower_bounds, upper_bounds),
                )

                if not res_lsq.success:
                    raise RuntimeError(res_lsq.message)

                popt = res_lsq.x
                residuals_final = res_lsq.fun
                n_data = len(y)

                # Separate data residuals from Ridge penalty
                data_residuals = residuals_final[:n_data]
                
                # Step 4: Compute Huber weights and covariance
                # Scale estimate (similar to HuberRegressor.scale_)
                abs_res = np.abs(data_residuals)
                scale = np.median(abs_res) / 0.6745  # MAD-based scale
                if scale < 1e-10:
                    scale = 1e-10

                # Huber weights
                scaled_res = data_residuals / scale
                abs_sr = np.abs(scaled_res)
                weights = np.where(abs_sr <= self.epsilon, 1.0, self.epsilon / abs_sr)
                outlier_pct = np.sum(abs_sr > self.epsilon) / len(abs_sr) * 100

                # Weighted Hessian approximation from Jacobian
                J = res_lsq.jac[:n_data, :]  # Only use data part, not Ridge penalty part
                XtWX = J.T @ (weights[:, None] * J)
                
                # Add Ridge contribution to Hessian:
                # The Ridge penalty sqrt(alpha)*||params|| contributes alpha*I
                if self.huber_alpha > 0:
                    XtWX += self.huber_alpha * np.diag(ridge_penalty_mask)

                sigma2 = scale ** 2

                try:
                    XtWX_inv = inv(XtWX)
                    pcov = sigma2 * XtWX_inv
                    condition_number = np.linalg.cond(XtWX)
                except np.linalg.LinAlgError:
                    pcov = np.full((5, 5), np.nan)
                    condition_number = np.inf

                # Step 5: Record results
                U_predicted = self.scaler_U.unscale(func(X, *popt)) + OCV(data_interval.temperature)
                U_true = data_interval.voltage

                results.loc[idx, "R2"] = r2_score(U_true, U_predicted)
                results.loc[idx, "RMSE"] = np.sqrt(mean_squared_error(U_true, U_predicted))
                results.loc[idx, "cond"] = condition_number
                results.loc[idx, "outlier_pct"] = outlier_pct
                results.loc[idx, "sigma2"] = sigma2

                for k in range(5):
                    results.loc[idx, f"c{k+1}"] = popt[k]
                    if not np.isnan(pcov[k, k]) and pcov[k, k] > 0:
                        results.loc[idx, f"c{k+1}_se"] = np.sqrt(pcov[k, k])
                    else:
                        results.loc[idx, f"c{k+1}_se"] = np.nan

                # Store XtWX directly as a numpy matrix object for fast SE calculation
                results.at[idx, "XTX"] = XtWX

                if condition_number >= self.threshold:
                    results.loc[idx, "status"] = f"High Cond ({condition_number:.1e})"
                else:
                    results.loc[idx, "status"] = "Fit Success"

            except Exception as e:
                num_fit_failed += 1
                results.loc[idx, "status"] = f"Fit Error: {str(e)}"

        print(f"[{self._mode_label} eps={self.epsilon} alpha={self.huber_alpha}] Fitting Stats: "
              f"{num_no_data} intervals low data, {num_fit_failed} fit failed.")
        return results
