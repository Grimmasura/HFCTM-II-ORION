from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TelemetryRecord(BaseModel):
    """Telemetry schema for per-step logging."""

    step: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_id: str
    version_id: str
    detector_metrics: Dict[str, float] = Field(default_factory=dict)
    action: Any = None
    prev_hash: Optional[str] = None
    hash: Optional[str] = None
    redacted_fields: List[str] = Field(default_factory=list)

    model_config = {"protected_namespaces": ()}
