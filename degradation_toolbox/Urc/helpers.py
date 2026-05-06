"""Degradation rate estimation and regression helpers.

Provides utilities for computing degradation rates with uncertainty
propagation, including orthogonal distance regression (ODR),
mixed-direction regression, and standard OLS.

Main Components:
    calculate_degradation_rate_and_uncertainty: End-to-end degradation
        slope estimation combining regression uncertainty and measurement
        error.
    mylinregress: Flexible linear regression supporting OLS, mixed, and
        orthogonal distance regression with optional fixed-point constraints.
    degradation_result: Container for degradation analysis outputs.
    result: Container for regression outputs (slope, intercept, stderr).
"""

import numpy as np
import pandas as pd
import scipy.odr
import scipy.optimize
import scipy.stats as stats


def calculate_degradation_rate_and_uncertainty(
    time: np.ndarray,
    Urc: np.ndarray,
    Urc_se: np.ndarray | None = None,
    fix_point: tuple | None = None,
    regression_method: str = 'orth',
) -> 'degradation_result':
    """Compute degradation rate and combined uncertainty.

    Combines regression uncertainty with measurement error (via Monte Carlo
    simulation) to produce a total uncertainty estimate for the degradation
    slope.

    Args:
        time: Calendar hours array for each Urc data point.
        Urc: Voltage under reference conditions [V].
        Urc_se: Standard errors of Urc values. None disables weighting.
        fix_point: Optional (x, y) anchor point for zero-intercept regression.
        regression_method: Regression variant ('orth', 'mix', or 'plain').

    Returns:
        A degradation_result with aging rate [uV/h], BOL voltage [V],
        and combined uncertainty [uV^2].
    """
    # 1. Regression on actual data
    reg = mylinregress(time, Urc, Urc_se, fix_point, regression_method)
    uncertainty_from_regression = (reg.stderr * 1e6) ** 2  # [uV^2]

    # 2. Measurement error via Monte Carlo
    # For small sample sizes (< num_random_data), repeated regressions on
    # synthetic noise yield an empirical estimate of the slope variance.
    # Threshold derived by Lennard: '20241120_samplingMeaserror.pptx'
    num_random_data = 5000
    np.random.seed(seed=1)  # Reproducible results

    # Measurement noise sigma = 5.33 mV, derived from redundant-sensor
    # calibration (Schenkolyse test): RSS of two independent sensors
    # with ~7.5 mV std each gives 7.54 / sqrt(2) = 5.33 mV.
    measurement_noise_sigma = 0.00533  # [V]

    if len(time) > num_random_data:
        y = stats.norm.rvs(size=len(time), loc=2024, scale=measurement_noise_sigma)
        r = mylinregress(time, y, mixed=regression_method)
        uncertainty_from_measurement_error = (r.stderr * 1e6) ** 2  # [uV^2]
    else:
        slope_samples, variance_samples = [], []
        for _ in range(int(num_random_data / len(time))):
            # loc is arbitrary (non-zero); it does not affect the uncertainty.
            y = stats.norm.rvs(size=len(time), loc=2024, scale=measurement_noise_sigma)
            r = mylinregress(time, y, mixed=regression_method)
            slope_samples.append(r.slope * 1e6)  # [uV]
            variance_samples.append((r.stderr * 1e6) ** 2)  # [uV^2]
        uncertainty_from_measurement_error = max(
            np.var(slope_samples), np.average(variance_samples)
        )  # [uV^2]

    # TODO: Calibrate individual Urc1/Urc2 uncertainties to account for
    #       different fitting parameters. Incorporate per-point Urc
    #       uncertainty via the same "fake regression" approach.
    uncertainty_uV2 = uncertainty_from_regression + uncertainty_from_measurement_error  # [uV^2]
    return degradation_result(reg.slope * 1e6, reg.intercept, uncertainty_uV2)


class degradation_result:
    """Container for degradation rate analysis results.

    Attributes:
        aging_uV_per_h: Degradation slope [uV/h].
        bol_V: Beginning-of-life voltage intercept [V].
        uncertainty_uV2: Combined uncertainty [uV^2].
    """

    def __init__(self, aging_uV_per_h: float, bol_V: float, uncertainty_uV2: float):
        self.aging_uV_per_h = aging_uV_per_h
        self.bol_V = bol_V
        self.uncertainty_uV2 = uncertainty_uV2


def mylinregress(
    x: np.ndarray,
    y: np.ndarray,
    sy: np.ndarray | None = None,
    fix_point: tuple | None = None,
    mixed: str = 'orth',
) -> 'result':
    """Flexible linear regression with multiple method options.

    Supports ordinary least squares ('plain'), geometric-mean regression
    ('mix'), and orthogonal distance regression ('orth') via scipy.odr.
    Optionally passes through a user-supplied fixed point.

    Args:
        x: Independent variable array.
        y: Dependent variable array.
        sy: Standard errors of y values for weighted ODR. None for unweighted.
        fix_point: Optional (x0, y0) point through which the line must pass.
            When provided, regression is performed without intercept on
            shifted coordinates (x - x0, y - y0).
        mixed: Regression method. One of 'plain', 'mix', or 'orth'.

    Returns:
        A result object with slope, intercept, and stderr attributes.
    """
    r = stats.linregress(x, y)

    if mixed == 'mix':
        ri = stats.linregress(y, x)
        slope = np.sqrt(r.slope / ri.slope) * np.sign(r.slope)
        intercept = 0.5 * (r.intercept - ri.intercept / ri.slope)
        error_avg = np.sqrt(r.stderr ** 2 + (ri.stderr / ri.slope ** 2) ** 2)
    elif mixed == 'plain':
        slope = r.slope
        intercept = r.intercept
        error_avg = r.stderr
    else:
        # Orthogonal distance regression (ODR)
        if fix_point:
            # Regression without intercept through (fix_x, fix_y)
            fix_x = fix_point[0]
            fix_y = fix_point[1]
            x = x - fix_x
            y = y - fix_y
            # Normalize X to unit length by dividing by max(x).
            xn = np.array(x) / max(x)
            yn = np.array(y) / np.average(y)
        else:
            # Normalize: x to unit length, y relative to mean
            xn = (np.array(x) - min(x)) / (max(x) - min(x))
            yn = np.array(y) / np.average(y)

        # If sy is provided, perform weighted orthogonal regression.
        if sy is not None:
            sy = np.array(sy)
            sy.flags.writeable = True
            # Replace NaN weights with 10x the maximum valid weight
            fix = np.isnan(sy)
            sy[fix] = 10.0 * max(sy)
            # Normalize sy for ODR (see scipy.odr.RealData docs)
            wy = 1 / (sy ** 2)
            wy = wy / sum(wy)
            mysy = 1 / np.sqrt(wy)
            mydata = scipy.odr.RealData(xn, yn, sy=mysy)
        else:
            mydata = scipy.odr.RealData(xn, yn)

        if not fix_point:
            def linf(B, x):
                """Linear function y = m*x + b."""
                return B[0] * x + B[1]

            linear = scipy.odr.Model(linf)
            s_n = r.slope / np.average(y) * (max(x) - min(x))
            i_n = r.intercept / np.average(y) + min(x) / (max(x) - min(x)) * s_n
            myodr = scipy.odr.ODR(mydata, linear, beta0=[s_n * 2, i_n * 2])
            myoutput = myodr.run()
            sno = myoutput.beta[0]
            ino = myoutput.beta[1]

            error_avg = myoutput.sd_beta[0] * np.average(y) / (max(x) - min(x))
            slope = sno * np.average(y) / (max(x) - min(x))
            intercept = (ino - min(x) / (max(x) - min(x)) * sno) * np.average(y)
        else:
            # Zero-intercept ODR through fixed point
            def linf(B, x):
                return B[0] * x

            linear = scipy.odr.Model(linf)
            s_n = np.average(y) / np.average(x)
            myodr = scipy.odr.ODR(mydata, linear, beta0=[s_n])
            myoutput = myodr.run()
            sno = myoutput.beta[0]

            # Reverse normalization: slope = sno * avg(y) / max(x)
            error_avg = myoutput.sd_beta[0] * np.average(y) / max(x)
            slope = sno * np.average(y) / max(x)
            # Regression through (x0, y0): y - y0 = slope * (x - x0)
            intercept = -slope * fix_x + fix_y

    myres = result(slope, intercept, error_avg)
    return myres


class result:
    """Container for linear regression output.

    Attributes:
        slope: Regression slope.
        intercept: Regression intercept.
        stderr: Standard error of the slope.
    """

    def __init__(self, slope: float, intercept: float, stderr: float):
        self.slope = slope
        self.intercept = intercept
        self.stderr = stderr


def StdErr(x, y, i, s):  # Not used
    """Standard error of data vs. an arbitrary linear model.

    Note:
        This is the residual standard error, NOT the slope standard error.

    Args:
        x: Independent variable values.
        y: Dependent variable values.
        i: Intercept of the linear model.
        s: Slope of the linear model.

    Returns:
        Standard error of the residuals.
    """
    N = len(x)
    e = np.sqrt(sum((i + s * x - y) ** 2) / sum((x - np.average(x)) ** 2) / (N - 2))
    return e


def od(x, y, i, sdummy=1):  # Not used
    """Cumulated orthogonal distance to an arbitrary linear model.

    Computes the sum of squared orthogonal distances from data points to
    the regression line, using normalized coordinates.

    Args:
        x: Independent variable values.
        y: Dependent variable values.
        i: Intercept of the linear model.
        sdummy: Slope of the linear model (or list [intercept, slope]).

    Returns:
        Scaled cumulated orthogonal distance (multiplied by 1000).
    """
    if type(sdummy) == type([]):
        i = i[0]
        s = i[1]
    else:
        s = sdummy

    # Normalize: unit-length x, mean-centered y
    xn = (np.array(x) - min(x)) / (max(x) - min(x))
    yn = np.array(y) / np.average(y)

    s_n = s / np.average(y) * (max(x) - min(x))
    i_n = i / np.average(y) + min(x) / (max(x) - min(x)) * s_n

    # Orthogonal projections onto regression line
    xi = (yn + xn / s_n - i_n) / (s_n + 1 / s_n)
    yi = i_n + s_n * xi

    d = sum((xn - xi) ** 2 + (yn - yi) ** 2)
    return 1000 * d
