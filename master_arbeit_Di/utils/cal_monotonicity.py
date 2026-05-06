import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from typing import Dict, Any, Union, Optional

def calculate_rank_monotonicity(y: Union[pd.Series, np.ndarray]) -> float:
    """
    [Macro Trend] Spearman's Rank Correlation Method.
    Formula: Monotonicity = | corr( rank(X), rank(T) ) |
    
    Sensitivity:
    - High for global trends (e.g., aging over 1000h).
    - Low for local noise (robust against small jitters).
    
    Returns:
        float: [0, 1]. 1.0 means perfect monotonic trend (regardless of linearity).
    """
    y_clean = pd.Series(y).dropna()
    n = len(y_clean)
    
    if n < 2:
        return 0.0
    
    t_clean = np.arange(n)
    
    # scipy.stats.spearmanr calculates Pearson correlation of ranks
    corr, p_value = spearmanr(y_clean, t_clean)
    
    if np.isnan(corr):
        return 0.0
        
    return abs(corr)

def calculate_sign_monotonicity(y: Union[pd.Series, np.ndarray]) -> float:
    """
    [Micro Stability] Signum Function Method.
    Formula: Monotonicity = | sum(sgn(x_{k+1} - x_k)) / (N-1) |
    
    Sensitivity:
    - High for local noise. Any step in the wrong direction reduces the score.
    - Best for evaluating "smoothness" or "stability" of the algorithm.
    
    Returns:
        float: [0, 1]. 1.0 means strictly monotonic (no jitter).
    """
    y_arr = np.array(y)
    y_arr = y_arr[~np.isnan(y_arr)] # Remove NaNs
    
    # 1. Calculate Differences: x(k+1) - x(k)
    diffs = np.diff(y_arr)
    n_diffs = len(diffs)
    
    if n_diffs == 0:
        return 0.0
    
    # 2. Apply Sign Function: +1 (up), -1 (down), 0 (flat)
    signs = np.sign(diffs)
    
    # 3. Sum and Normalize
    # Strict implementation of: | (N_pos - N_neg) / (N-1) |
    # which is mathematically equivalent to | sum(signs) / n_diffs |
    monotonicity = np.abs(np.sum(signs)) / n_diffs
    
    return monotonicity
