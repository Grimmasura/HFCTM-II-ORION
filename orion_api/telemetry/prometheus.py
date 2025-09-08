"""Prometheus optional dependency wrappers."""

from __future__ import annotations

import logging

try:  # pragma: no cover - optional dependency
    from prometheus_client import Histogram
except Exception:  # pragma: no cover
    class Histogram:  # type: ignore[misc]
        """No-op Histogram when prometheus_client is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def observe(self, *args, **kwargs) -> None:
            return None

    logging.getLogger(__name__).info(
        "prometheus_client not installed; metrics disabled"
    )

__all__ = ["Histogram"]

