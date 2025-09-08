"""Prometheus metric helpers with graceful fallbacks.

This module exposes thin wrappers around prometheus_client metrics that
handle the optional dependency gracefully. When prometheus_client is not
available, the provided classes become no-op implementations so that the
rest of the application can continue to operate without instrumentation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from prometheus_client import Histogram as _Histogram
    Histogram = _Histogram
except Exception:  # pragma: no cover - import error
    class Histogram:  # type: ignore[misc]
        """No-op fallback when prometheus_client is unavailable."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            pass

        def observe(self, value: float) -> None:
            return None

    logger.info("prometheus_client not installed; metrics disabled")
