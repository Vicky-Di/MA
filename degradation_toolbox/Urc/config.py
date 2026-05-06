"""
Configuration Management for Electrolyzer Reference Voltage Models

This module provides centralized configuration for the Urc model suite:
- Physical constants (OCV coefficients, etc.)
- Data filtering parameters (current, voltage, temperature ranges)
- Feature scaling ranges (min/max for normalization)
- Reference condition bundles (Low/Medium/High operating points)

Usage
-----
From dictionary:
    >>> from degradation_toolbox.Urc.config import PhysicsConstants, ScalingConfig
    >>> physics = PhysicsConstants()
    >>> scaling = ScalingConfig()

From YAML file (requires pyyaml):
    >>> config = ModelConfig.from_yaml("config.yaml")
    >>> model = Urc1(data, **config.to_dict())
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
from pathlib import Path

# Optional dependency — yaml is only needed for to_yaml() / from_yaml()
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class PhysicsConstants:
    """
    Physical constants for electrolyzer voltage calculations.
    
    The OCV (Open Circuit Voltage) model approximates the thermodynamic voltage
    as a linear function of temperature based on CoolProp results at 1 atm.
    
    Attributes
    ----------
    OCV_COEFFICIENT : float
        Temperature sensitivity [V/°C]. Typically negative (voltage decreases with T).
        Default based on H2O→H2+O2 at atmospheric pressure.
    OCV_INTERCEPT : float
        OCV at 0°C [V]. Reference point for the linear OCV model.
    MIN_CURRENT_DENSITY : float
        Minimum physically realistic current density [A/cm²].
    MAX_CURRENT_DENSITY : float
        Maximum realistic current density [A/cm²].
    MIN_TEMPERATURE : float
        Minimum operating temperature [°C].
    MAX_TEMPERATURE : float
        Maximum operating temperature [°C].
    """
    
    # OCV Model: U_ocv = OCV_COEFFICIENT * T + OCV_INTERCEPT
    OCV_COEFFICIENT: float = -8.2975e-4  # [V/°C]
    OCV_INTERCEPT: float = 1.24965        # [V]
    OCV_SOURCE: str = "CoolProp linear approximation at 1 atm"
    
    # Operating envelope
    MIN_CURRENT_DENSITY: float = 0.1      # [A/cm²]
    MAX_CURRENT_DENSITY: float = 2.0      # [A/cm²]
    MIN_TEMPERATURE: float = 30            # [°C]
    MAX_TEMPERATURE: float = 80            # [°C]
    


@dataclass
class ScalingConfig:
    """
    Feature scaling (normalization) ranges for the voltage degradation model.
    
    The Urc model uses min-max scaling to normalize features to [0, 1] before fitting:
    X_scaled = (X - min) / (max - min)
    
    Choosing appropriate ranges improves:
    - Numerical stability of least-squares fitting
    - Convergence speed of optimization
    - Physical interpretability of coefficients
    
    Attributes
    ----------
    MIN_I, MAX_I : float
        Current density range [A/cm²]
    MIN_T, MAX_T : float
        Temperature range [°C]
    MIN_U, MAX_U : float
        Voltage range [V] (after OCV subtraction)
    MIN_LOG_H, MAX_LOG_H : float
        Log operating hours range [ln(h)]
    """
    
    # Current density [A/cm²]
    MIN_I: float = 0.3
    MAX_I: float = 1.6
    
    # Temperature [°C]
    MIN_T: float = 50
    MAX_T: float = 65
    
    # Temperature range for I*T product scaling [°C]
    # NOTE: In Urc1._init_scalers(), I*T uses T_range=[55, 65], NOT [50, 65].
    # This narrower range reflects the typical steady-state operating envelope. 
    MIN_T_FOR_IT: float = 55
    MAX_T_FOR_IT: float = 65
    
    # Voltage [V] — AFTER OCV subtraction, so often U - U_ocv ≈ 0-2V
    MIN_U: float = 0
    MAX_U: float = 2.3 - 1.23  # Typical max - typical min
    
    # Logarithmic operating hours [ln(h)]
    MIN_LOG_H: float = 0          # ln(1) = 0
    MAX_LOG_H: float = 5.29       # ln(200) ≈ 5.3
    
    @property
    def min_I_T(self) -> float:
        """Product MIN_I * MIN_T_FOR_IT (matches Urc1._init_scalers)."""
        return self.MIN_I * self.MIN_T_FOR_IT
    
    @property
    def max_I_T(self) -> float:
        """Product MAX_I * MAX_T_FOR_IT (matches Urc1._init_scalers)."""
        return self.MAX_I * self.MAX_T_FOR_IT
    
    @property
    def min_I2(self) -> float:
        """I² at minimum current."""
        return self.MIN_I ** 2
    
    @property
    def max_I2(self) -> float:
        """I² at maximum current."""
        return self.MAX_I ** 2


@dataclass
class DataFilterConfig:
    """
    Data filtering thresholds to extract valid operating regions.
    
    The Urc model requires specific operating conditions for valid fitting.
    This config specifies which data points to retain based on measured values.
    
    Attributes
    ----------
    i_min, i_max : float
        Current density bounds [A/cm²]
    u_min, u_max : float
        Voltage bounds [V]
    t_min, t_max : float
        Temperature bounds [°C]
    h_since_last_start_min : float
        Minimum operating hours since last start [h]. Rejects transient startup data.
    resample_rate : int
        Temporal downsampling factor (≥1). 
        1 = use all data, 2 = use every 2nd point, etc.
    min_num_data_required_for_fit : float
        Minimum data points in a sliding window to attempt fitting.
    """
    
    # Current density [A/cm²]
    i_min: float = 0.1
    i_max: float = 1.9
    
    # Voltage [V]
    u_min: float = 1.4
    u_max: float = 2.3
    
    # Temperature [°C]
    t_min: float = 50
    t_max: float = 65
    
    # Operating hours [h] — reject data < 0.5h (startup transient)
    h_since_last_start_min: float = 0.5
    
    # Temporal resampling
    resample_rate: int = 1
    
    # Minimum fitting window size [points]
    # Default: 0.5 * 1 * 24 * 60 = 720 points (half day at 1-min resolution)
    min_num_data_required_for_fit: float = 0.5 * 24 * 60


@dataclass
class RefConfig:
    """
    Reference operating condition for Urc calculation.
    
    Urc (Reference Voltage) is computed at specific reference conditions for 
    degradation monitoring and comparison. This class specifies one such condition set.
    
    Attributes
    ----------
    Iref : float
        Reference current density [A/cm²]
    Tref : float
        Reference temperature [°C]
    OHref : float
        Reference operating hours [h]
    name : str
        Human-readable label (e.g., "Low", "Medium", "High")
    gt_path : str or Path, optional
        Path to Ground Truth data CSV for this ref condition
    """
    
    Iref: float
    Tref: float
    OHref: float
    name: str = "ref"
    gt_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model initialization."""
        return {
            "Iref": self.Iref,
            "Tref": self.Tref,
            "OHref": self.OHref,
        }
    
    def __str__(self) -> str:
        return f"Ref({self.name}: I={self.Iref}A/cm², T={self.Tref}°C, OH={self.OHref}h)"


@dataclass
class RefConfigBundle:
    """
    Bundle of reference conditions across operating envelope (Low/Medium/High).
    
    Modern electrolyzer operation spans a range of current densities from low 
    (standby/part-load) to high (rated load). This class packages three reference
    conditions to enable comparable degradation analysis across the envelope.
    
    Attributes
    ----------
    Low : RefConfig
        Low current operation (e.g., 0.6 A/cm²)
    Medium : RefConfig
        Medium current operation (e.g., 1.0 A/cm²)
    High : RefConfig
        High current operation (e.g., 1.5 A/cm²)
    
    Example
    -------
    >>> bundle = RefConfigBundle(
    ...     Low=RefConfig(Iref=0.6, Tref=55, OHref=24, name="Low"),
    ...     Medium=RefConfig(Iref=1.0, Tref=60, OHref=72, name="Medium"),
    ...     High=RefConfig(Iref=1.5, Tref=65, OHref=100, name="High"),
    ... )
    >>> model = Urc1(data, ref_config=bundle.to_dict())
    """
    
    Low: RefConfig
    Medium: RefConfig
    High: RefConfig
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert to nested dictionary for unified model initialization.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            {"Low": {...}, "Medium": {...}, "High": {...}}
            Each inner dict is {Iref, Tref, OHref, gt_path (optional)}
        """
        result = {}
        for ref_obj in [self.Low, self.Medium, self.High]:
            d = ref_obj.to_dict()
            if ref_obj.gt_path:
                d["gt_path"] = ref_obj.gt_path
            result[ref_obj.name] = d
        return result
    
    def to_list(self) -> list:
        """Convert to list for backward compatibility with old Iref list format."""
        return [self.Low.Iref, self.Medium.Iref, self.High.Iref]
    
    @classmethod
    def standard(cls) -> "RefConfigBundle":
        """Create standard ref bundle (G6M2/G1M1 typical values)."""
        return cls(
            Low=RefConfig(Iref=0.6, Tref=55, OHref=24, name="Low"),
            Medium=RefConfig(Iref=1.0, Tref=60, OHref=72, name="Medium"),
            High=RefConfig(Iref=1.5, Tref=65, OHref=100, name="High"),
        )


@dataclass
class ModelConfig:
    """
    Complete configuration for Urc1 model fitting.
    
    Integrates physics constants, filtering thresholds, scaling ranges,
    and reference conditions into a single configuration object.
    
    Supports YAML serialization for reproducibility and version control.
    """
    
    # Component configs
    physics: PhysicsConstants = field(default_factory=PhysicsConstants)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    data_filter: DataFilterConfig = field(default_factory=DataFilterConfig)
    ref_bundle: RefConfigBundle = field(default_factory=lambda: RefConfigBundle.standard())
    
    # Model-specific parameters
    len_interval: int = 2           # Sliding window size [days]
    slide: int = 1                  # Sliding step [days]
    threshold: float = 1e6          # Condition number threshold for reliability
    plot_fit: bool = False          # Plot intermediate fits (can be slow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary compatible with Urc1.__init__() parameters."""
        return {
            # Filtering & resampling
            "data_filter_i_min": self.data_filter.i_min,
            "data_filter_U_min": self.data_filter.u_min,
            "data_filter_U_max": self.data_filter.u_max,
            "data_filter_T_min": self.data_filter.t_min,
            "data_filter_T_max": self.data_filter.t_max,
            "data_filter_h_since_last_start_min": self.data_filter.h_since_last_start_min,
            "resample": self.data_filter.resample_rate,
            "min_num_data_required_for_fit": self.data_filter.min_num_data_required_for_fit,
            
            # Reference conditions
            "Iref": self.ref_bundle.to_list(),
            "Tref": self.ref_bundle.Medium.Tref,
            "OHref": self.ref_bundle.Medium.OHref,
            
            # Model fitting
            "len_interval": self.len_interval,
            "slide": self.slide,
            "threshold": self.threshold,
            "plot_fit": self.plot_fit,
        }
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file.
        
        Raises:
            ImportError: If pyyaml is not installed.
        """
        if not _HAS_YAML:
            raise ImportError(
                "pyyaml is required for YAML support. "
                "Install it with: pip install pyyaml"
            )
        config_dict = {
            "physics": asdict(self.physics),
            "scaling": asdict(self.scaling),
            "data_filter": asdict(self.data_filter),
            "ref_bundle": self.ref_bundle.to_dict(),
            "model": {
                "len_interval": self.len_interval,
                "slide": self.slide,
                "threshold": self.threshold,
                "plot_fit": self.plot_fit,
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"✓ Configuration saved to {path}")
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load configuration from YAML file.
        
        Raises:
            ImportError: If pyyaml is not installed.
            FileNotFoundError: If the YAML file does not exist.
        """
        if not _HAS_YAML:
            raise ImportError(
                "pyyaml is required for YAML support. "
                "Install it with: pip install pyyaml"
            )
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        physics = PhysicsConstants(**config_dict.get("physics", {}))
        scaling = ScalingConfig(**config_dict.get("scaling", {}))
        data_filter = DataFilterConfig(**config_dict.get("data_filter", {}))
        
        # Parse ref bundle
        ref_dict = config_dict.get("ref_bundle", {})
        ref_bundle = RefConfigBundle(
            Low=RefConfig(**ref_dict.get("Low", {})),
            Medium=RefConfig(**ref_dict.get("Medium", {})),
            High=RefConfig(**ref_dict.get("High", {})),
        )
        
        model_cfg = config_dict.get("model", {})
        
        return cls(
            physics=physics,
            scaling=scaling,
            data_filter=data_filter,
            ref_bundle=ref_bundle,
            **model_cfg
        )
    
    @classmethod
    def default(cls) -> "ModelConfig":
        """Create default configuration."""
        return cls()


# Export publicly
__all__ = [
    "PhysicsConstants",
    "ScalingConfig",
    "DataFilterConfig",
    "RefConfig",
    "RefConfigBundle",
    "ModelConfig",
]
