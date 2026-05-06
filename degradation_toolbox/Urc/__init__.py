"""
Urc: Electrolyzer Reference Voltage (Urc) Models Module

This module provides a comprehensive suite of voltage degradation modeling tools 
for water electrolyzers, including:
- Base model (Urc1) and variants using different regression techniques
- Unified model comparator for performance evaluation
- Ground truth (GT) data integration and alignment
- Visualization and diagnostic tools

Core Classes
-----------
Urc1
    Base voltage degradation model using ordinary least squares (OLS) fitting
    
Urc1BaseHuber
    Robust variant using Huber regression for outlier rejection
    
Urc1_GPR
    Gaussian Process Regression variant for uncertainty quantification
    
Urc1_Iref
    Specialized model with Iref-specific filtering logic
    
Urc1_Surface_Stats
    Surface-based voltage degradation model

UnifiedModelComparator
    Unified tool for comparing multiple models, GT alignment, and metrics

References
----------
See individual class docstrings for detailed descriptions and usage examples.

Example
-------
>>> from degradation_toolbox.Urc.Urc1 import Urc1
>>> from master_arbeit_Di.explore.UnifiedModelComparator import UnifiedModelComparator
>>> model = Urc1(data, name="Cell1", Iref=[0.6, 1.0, 1.5])
>>> comparator = UnifiedModelComparator({"Model1": model})
>>> metrics = comparator.compare_all(i_target=1.0)
"""

# --- Configuration Management ---
from .config import (
    PhysicsConstants,
    ScalingConfig,
    DataFilterConfig,
    RefConfig,
    RefConfigBundle,
    ModelConfig,
)

# --- Core Model Classes (always available) ---
from .Urc1 import Urc1
from .Urc1_huber import Urc1BaseHuber
from .Urc1_Iref import Urc1_Iref
from .Urc1_gpr import Urc1_GPR

# --- Subclass imports (may fail if dependencies are missing) ---
# Urc1_c6_sigmoid.py defines class named `Urc1` (conflicts with base);
# import it under an alias to avoid shadowing.
try:
    from .Urc1_c6_sigmoid import Urc1 as Urc1_C6Sigmoid
except ImportError:
    Urc1_C6Sigmoid = None

try:
    from .Urc1_c3 import Urc1_c3
except ImportError:
    Urc1_c3 = None

try:
    from .Urc1_surface_stats import Urc1_Surface_Stats
except ImportError:
    Urc1_Surface_Stats = None

try:
    from .Urc1_surface import Urc1_Surface
except ImportError:
    Urc1_Surface = None

# Note: Urc1_adaptive_updated.py is currently disabled (code commented out).
# from .Urc1_adaptive_updated import Urc1_Adaptive  # not available

# --- Comparison & Visualization Tools ---
from master_arbeit_Di.explore.UnifiedModelComparator import UnifiedModelComparator
from master_arbeit_Di.explore.Comparison_tools import ModelComparison
from master_arbeit_Di.explore.deg_rate import deg_rate_multi_electrolyzers

# --- Helper Utilities ---
from .helpers import (
    calculate_degradation_rate_and_uncertainty,
    degradation_result,
)

# --- Module Public API ---
__all__ = [
    # Configuration
    "PhysicsConstants",
    "ScalingConfig",
    "DataFilterConfig",
    "RefConfig",
    "RefConfigBundle",
    "ModelConfig",
    
    # Core models
    "Urc1",
    "Urc1BaseHuber",
    "Urc1_C6Sigmoid",
    "Urc1_c3",
    "Urc1_Iref",
    "Urc1_GPR",
    "Urc1_Surface_Stats",
    "Urc1_Surface",
    
    # Comparison & visualization
    "UnifiedModelComparator",
    "ModelComparison",
    "deg_rate_multi_electrolyzers",
    
    # Helpers
    "calculate_degradation_rate_and_uncertainty",
    "degradation_result",
]

__version__ = "2.0.0"
