from typing import Optional

try:
    from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    PROM_AVAILABLE = False


class _NoOp:
    def __getattr__(self, name):
        def _noop(*a, **k):
            return None
        return _noop


class OrionMetrics:
    """Prometheus-backed metrics with graceful no-op fallback."""
    def __init__(self) -> None:
        self.available = PROM_AVAILABLE
        if self.available:
            self.registry = CollectorRegistry()
            self.recursions_active = Gauge(
                "orion_recursions_active", "Active recursion tasks", registry=self.registry
            )
            self.recursion_depth = Histogram(
                "orion_recursion_depth_histogram",
                "Observed recursion depth",
                buckets=(1, 2, 3, 5, 8, 13, 21, 34),
                registry=self.registry,
            )
            self.tasks_inflight = Gauge(
                "orion_tasks_inflight", "Tasks in flight across agents", registry=self.registry
            )
            self.egregore_anomaly = Gauge(
                "egregore_anomaly_score", "Aggregated egregore anomaly score", registry=self.registry
            )
            self.drift_eigen_max = Gauge(
                "drift_eigenvalue_max", "Max eigenvalue of drift correlation matrix", registry=self.registry
            )
            self.quarantine_events = Counter(
                "policy_quarantine_events_total", "Quarantine events", registry=self.registry
            )
            self.quantum_unavailable = Gauge(
                "quantum_sync_unavailable", "Quantum sync unavailable flag", registry=self.registry
            )
        else:
            self.registry = None
            self.recursions_active = _NoOp()
            self.recursion_depth = _NoOp()
            self.tasks_inflight = _NoOp()
            self.egregore_anomaly = _NoOp()
            self.drift_eigen_max = _NoOp()
            self.quarantine_events = _NoOp()
            self.quantum_unavailable = _NoOp()

    def render(self) -> tuple[str, bytes]:
        if not self.available:
            return ("text/plain", b"prometheus_unavailable 1\n")
        body = generate_latest(self.registry)
        return (CONTENT_TYPE_LATEST, body)
