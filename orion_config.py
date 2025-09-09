from __future__ import annotations

"""ORION Unified Configuration System
Central configuration management for all HFCTM-II subsystems."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field

try:  # pragma: no cover - allow missing optional dependency locally
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception as e:  # pragma: no cover
    raise ImportError(
        "pydantic-settings is required for ORION configuration (Pydantic v2). "
        "Install with: pip install pydantic>=2 pydantic-settings"
    ) from e


class ChiralConfig(BaseSettings):
    """Chiral inversion mechanics configuration"""

    lambda_anti: float = Field(0.1, description="Anti-symmetric learning weight")
    lambda_sym: float = Field(0.05, description="Common-mode symmetry weight")
    corr_threshold: float = Field(0.8, description="Correlation threshold for gating")
    gating_alpha: float = Field(0.3, description="Common-mode gating factor")
    distance_metric: str = Field("js", description="Distance metric: js, kl, l2")
    symmetry_op: str = Field("anti", description="Symmetry operation: anti, align")

    # Numerical stability
    eps_clip: float = Field(1e-6, description="Epsilon for numerical stability")
    cache_max_size: int = Field(1000, description="LRU cache size for tensor inversions")

    # Text inversion
    semantic_confidence_threshold: float = Field(
        0.7, description="Threshold for semantic flip fallback"
    )


class SchedulerConfig(BaseSettings):
    """Recursive scheduler configuration"""

    initial_credits: float = Field(1000.0, description="Initial recursion credits")
    depth_cost_gamma: float = Field(1.5, description="Depth cost scaling γ")
    base_cost: float = Field(1.0, description="Base cost c₀")
    beam_size: int = Field(10, description="Beam search frontier size")
    max_depth: int = Field(20, description="Maximum recursion depth")

    # SLA limits
    wall_time_limit: float = Field(300.0, description="Wall time limit (seconds)")
    memory_limit_mb: float = Field(1024.0, description="Memory limit (MB)")
    token_limit: int = Field(100000, description="Token limit")

    # UCT parameters
    c_explore: float = Field(1.41, description="UCT exploration constant")
    cost_penalty: float = Field(0.1, description="UCT cost penalty λ")

    # Stopping criteria
    marginal_gain_epsilon: float = Field(
        0.01, description="Marginal gain stopping threshold"
    )
    convergence_window: int = Field(
        3, description="Window for marginal gain averaging"
    )

    # Entropy gating
    min_entropy: float = Field(0.1, description="Minimum entropy threshold")
    entropy_convergence_window: int = Field(
        5, description="Entropy convergence window"
    )


class QuantumConfig(BaseSettings):
    """Quantum stabilization configuration"""

    n_qubits: int = Field(4, description="Number of qubits")
    control_strength: float = Field(1.0, description="Control Hamiltonian coupling")
    lyapunov_kappa: List[float] = Field(
        [0.1, 0.1, 0.1, 0.1], description="Lyapunov feedback gains"
    )
    decoherence_rate: float = Field(0.01, description="Lindblad decoherence rate")

    # Counterdiabatic driving
    counterdiabatic_eta: float = Field(0.5, description="CD drive strength η")

    # Floquet analysis
    floquet_period: float = Field(1.0, description="Floquet period")
    max_eigenval_thresh: float = Field(
        1.1, description="Max Floquet eigenvalue threshold"
    )

    # Stability thresholds
    stability_threshold: float = Field(0.95, description="Stability threshold")
    fidelity_threshold: float = Field(0.9, description="Minimum fidelity threshold")

    # Numerical integration
    use_magnus_expansion: bool = Field(
        True, description="Use Magnus expansion for short times"
    )
    magnus_order: int = Field(2, description="Magnus expansion order")
    ode_rtol: float = Field(1e-6, description="ODE integration relative tolerance")
    use_psd_projection: bool = Field(
        False, description="Project density matrices to PSD cone after evolution"
    )


class DefenseConfig(BaseSettings):
    """Egregore defense configuration"""

    correlation_threshold: float = Field(
        0.8, description="τ for λ₁ correlation gating"
    )
    chiral_consistency_threshold: float = Field(
        0.7, description="τ_χ minimum chiral consistency"
    )

    # ESI parameters
    esi_window_size: int = Field(100, description="ESI computation window")

    # Persistence tracking
    persistence_horizon: int = Field(10, description="L-turn persistence horizon")

    # Tolerance thresholds
    utility_tolerance: float = Field(0.05, description="Acceptable utility drop")
    drift_tolerance: float = Field(0.1, description="Procrustes drift tolerance")
    mi_bound: float = Field(0.05, description="κ mutual information bound")
    recovery_tolerance: float = Field(0.02, description="Recovery threshold")

    # Attack evaluation
    attack_success_threshold: float = Field(
        0.5, description="ASR success threshold"
    )
    paraphrase_similarity_threshold: float = Field(
        0.8, description="Paraphrase stability threshold"
    )


class MonitoringConfig(BaseSettings):
    """Monitoring and telemetry configuration"""

    prometheus_port: int = Field(9090, description="Prometheus metrics port")
    metrics_update_interval: float = Field(
        1.0, description="Metrics update interval (seconds)"
    )

    # Logging
    log_level: str = Field("INFO", description="Logging level")
    log_file: Optional[str] = Field(None, description="Log file path")

    # Dashboard
    grafana_dashboard: bool = Field(True, description="Enable Grafana dashboard")
    alert_thresholds: Dict[str, float] = Field(
        {
            "chiral_consistency_min": 0.7,
            "correlation_eigenval_max": 0.8,
            "esi_max": 0.3,
            "asr_max": 0.2,
            "utility_retention_min": 95.0,
        },
        description="Alert thresholds for monitoring",
    )


class ORIONConfig(BaseSettings):
    """Main ORION configuration aggregator"""

    # Subsystem configurations
    chiral: ChiralConfig = Field(default_factory=ChiralConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    quantum: QuantumConfig = Field(default_factory=QuantumConfig)
    defense: DefenseConfig = Field(default_factory=DefenseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Global settings
    environment: str = Field(
        "development", description="Environment: development, staging, production"
    )
    debug: bool = Field(False, description="Enable debug mode")
    seed: int = Field(42, description="Random seed for reproducibility")

    # Model settings
    model_dir: str = Field("models", description="Model directory")
    checkpoint_interval: int = Field(1000, description="Checkpoint save interval")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ORION_", case_sensitive=False
    )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ORIONConfig:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        config_dict = yaml.safe_load(config_path.read_text()) or {}
        return cls(**config_dict)

    def to_yaml(self, config_path: str | Path) -> None:
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.write_text(
            yaml.safe_dump(self.model_dump(), default_flow_style=False, sort_keys=False)
        )

    def validate_config(self) -> List[str]:
        """Validate configuration and return any warnings"""
        warnings: List[str] = []

        # Validate chiral config
        if self.chiral.lambda_anti <= 0:
            warnings.append("chiral.lambda_anti should be positive")

        # Validate scheduler config
        if self.scheduler.depth_cost_gamma <= 1.0:
            warnings.append(
                "scheduler.depth_cost_gamma should be > 1.0 for exponential growth"
            )

        # Validate quantum config
        if len(self.quantum.lyapunov_kappa) != self.quantum.n_qubits:
            warnings.append(
                f"quantum.lyapunov_kappa length should match n_qubits ({self.quantum.n_qubits})"
            )

        # Validate defense config
        if self.defense.chiral_consistency_threshold > 1.0:
            warnings.append("defense.chiral_consistency_threshold should be ≤ 1.0")

        return warnings


class ConfigManager:
    """Configuration manager for ORION"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config = ORIONConfig.from_yaml(config_path)
        else:
            self.config = ORIONConfig()

        warnings = self.config.validate_config()
        if warnings:
            for warning in warnings:
                print(f"CONFIG WARNING: {warning}")

    def get_subsystem_config(self, subsystem: str) -> Any:
        """Get configuration for specific subsystem"""
        return getattr(self.config, subsystem)

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    def save_config(self, path: str) -> None:
        """Save current configuration"""
        self.config.to_yaml(path)


# Default configuration templates

def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary"""
    return {
        "chiral": {
            "lambda_anti": 0.1,
            "lambda_sym": 0.05,
            "corr_threshold": 0.8,
            "gating_alpha": 0.3,
            "distance_metric": "js",
            "symmetry_op": "anti",
        },
        "scheduler": {
            "initial_credits": 1000.0,
            "depth_cost_gamma": 1.5,
            "beam_size": 10,
            "max_depth": 20,
            "sla": {
                "wall_time_limit": 300.0,
                "memory_limit_mb": 1024.0,
                "token_limit": 100000,
            },
        },
        "quantum": {
            "n_qubits": 4,
            "lyapunov_kappa": [0.1, 0.1, 0.1, 0.1],
            "counterdiabatic_eta": 0.5,
            "floquet_period": 1.0,
            "stability_threshold": 0.95,
        },
        "defense": {
            "correlation_threshold": 0.8,
            "chiral_consistency_threshold": 0.7,
            "utility_tolerance": 0.05,
            "mi_bound": 0.05,
            "drift_tolerance": 0.1,
        },
        "monitoring": {
            "prometheus_port": 9090,
            "alert_thresholds": {
                "chiral_consistency_min": 0.7,
                "correlation_eigenval_max": 0.8,
                "esi_max": 0.3,
                "asr_max": 0.2,
                "utility_retention_min": 95.0,
            },
        },
    }


def create_config_file(output_path: str = "orion.yaml") -> None:
    """Create default configuration file"""
    config_dict = create_default_config()
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Created configuration file: {output_path}")


if __name__ == "__main__":
    create_config_file()
    config_manager = ConfigManager("orion.yaml")
    print("Configuration loaded successfully!")
    warnings = config_manager.config.validate_config()
    if not warnings:
        print("Configuration validation passed!")
    chiral_config = config_manager.get_subsystem_config("chiral")
    print(f"Chiral lambda_anti: {chiral_config.lambda_anti}")

# Backwards-compatible alias
OrionConfig = ORIONConfig

__all__ = [
    "ChiralConfig",
    "SchedulerConfig",
    "QuantumConfig",
    "DefenseConfig",
    "MonitoringConfig",
    "ORIONConfig",
    "OrionConfig",
    "ConfigManager",
    "create_default_config",
    "create_config_file",
]
