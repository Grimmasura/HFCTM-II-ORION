from __future__ import annotations

"""Unified ORION configuration module."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class QuantumConfig(BaseModel):
    """Configuration for quantum stabilizer subsystem."""

    n_qubits: int = 4
    control_strength: float = 1.0
    lyapunov_kappa: list[float] | None = None
    decoherence_rate: float = 0.01
    counterdiabatic_eta: float = 0.5
    floquet_period: float = 1.0
    stability_threshold: float = 0.95
    max_eigenval_thresh: float = 1.1
    use_psd_projection: bool = Field(
        False, description="Project density matrices to PSD cone after evolution",
    )

    @field_validator("lyapunov_kappa", mode="after")
    @classmethod
    def _default_kappa(cls, v: list[float] | None, info: Any) -> list[float]:
        n = info.data.get("n_qubits", 4)
        return v if v is not None else [0.1] * n


class OrionConfig(BaseSettings):
    """Top-level settings loaded from environment or YAML."""

    quantum: QuantumConfig = QuantumConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OrionConfig":
        data: dict[str, Any] = {}
        p = Path(path)
        if p.is_file():
            data = yaml.safe_load(p.read_text()) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.write_text(yaml.safe_dump(self.model_dump()))


__all__ = ["QuantumConfig", "OrionConfig"]
