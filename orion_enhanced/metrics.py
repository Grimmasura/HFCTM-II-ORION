import os
try:
    from prometheus_client import (
        CollectorRegistry, Counter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST, REGISTRY, multiprocess
    )
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False


class OrionMetrics:
    """Prometheus-backed metrics with graceful no-op fallback and idempotent registration."""

    def __init__(self):
        self.available = PROM_AVAILABLE
        if self.available:
            if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
                self.registry = REGISTRY
                try:
                    multiprocess.MultiProcessCollector(self.registry)
                except Exception:
                    pass
            else:
                self.registry = CollectorRegistry()

            def _names():
                return getattr(self.registry, "_names_to_collectors", {})

            def _gauge(name, doc):
                names = _names()
                return names.get(name) or Gauge(name, doc, registry=self.registry)

            def _counter(name, doc):
                names = _names()
                return names.get(name) or Counter(name, doc, registry=self.registry)

            def _histogram(name, doc, **kw):
                names = _names()
                return names.get(name) or Histogram(name, doc, registry=self.registry, **kw)

            self.recursions_active = _gauge("orion_recursions_active", "Active recursion tasks")
            self.recursion_depth = _histogram(
                "orion_recursion_depth_histogram", "Observed recursion depth",
                buckets=(1, 2, 3, 5, 8, 13, 21, 34)
            )
            self.egregore_anomaly = _gauge("egregore_anomaly_score", "Aggregated anomaly score")
            self.quarantine_events = _counter("policy_quarantine_events_total", "Quarantine events")
            self.tasks_inflight = _gauge("orion_tasks_inflight", "Tasks in flight across agents")
            self.quantum_unavailable = _gauge("quantum_sync_unavailable", "Quantum sync unavailable flag")
        else:
            self.recursions_active = self._noop()
            self.recursion_depth = self._noop()
            self.egregore_anomaly = self._noop()
            self.quarantine_events = self._noop()
            self.tasks_inflight = self._noop()
            self.quantum_unavailable = self._noop()

    def _noop(self):
        class NoOp:
            def __getattr__(self, name):
                return lambda *a, **k: None
        return NoOp()

    def render(self) -> tuple[str, bytes]:
        if not self.available:
            return ("text/plain", b"prometheus_unavailable 1\n")
        body = generate_latest(self.registry)
        return (CONTENT_TYPE_LATEST, body)
