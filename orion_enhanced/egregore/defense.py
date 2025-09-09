from __future__ import annotations
from typing import Optional, Sequence
from collections import Counter
import math

try:
    import numpy as np
    NUMPY_OK = True
except Exception:  # pragma: no cover
    NUMPY_OK = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np  # ensure available
    _EMB = SentenceTransformer("all-MiniLM-L6-v2")
    EMB_OK = True
except Exception:  # pragma: no cover
    EMB_OK = False
    _EMB = None


class Quarantine(Exception):
    pass


def _kl(p: Counter, q: Counter) -> float:
    # simple add-k smoothing
    k = 1.0
    vocab = set(p) | set(q)
    P = [(p[t] + k) for t in vocab]
    Q = [(q[t] + k) for t in vocab]
    Zp = sum(P); Zq = sum(Q)
    return sum((pi/Zp) * math.log((pi/Zp) / (qi/Zq)) for pi, qi in zip(P, Q))


def ngram_kl(current: str, baseline: str, n: int = 3) -> Optional[float]:
    def grams(s: str) -> Counter:
        toks = s.split()
        return Counter(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))
    try:
        return _kl(grams(current), grams(baseline))
    except Exception:  # pragma: no cover
        return None


def embedding_shift(current: str, baseline: str) -> Optional[float]:
    if not EMB_OK:
        return None
    a = _EMB.encode([current, baseline])
    # cosine distance
    num = (a[0] * a[1]).sum()
    den = (np.linalg.norm(a[0]) * np.linalg.norm(a[1]) + 1e-9)
    return 1.0 - (num / den)


def behavior_signal(err_rate_delta: float, tool_mix_delta: float) -> float:
    # normalized [0,1] surrogate; tune as needed
    return max(0.0, min(1.0, 0.5 * abs(err_rate_delta) + 0.5 * abs(tool_mix_delta)))


def aggregate_score(values: Sequence[Optional[float]]) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


class EgregoreDefense:
    def __init__(self, metrics=None, thresholds=None):
        self.metrics = metrics
        self.th = {"kl": 1.2, "emb": 0.28, "beh": 0.35, "final": 0.4}
        if thresholds:
            self.th.update(thresholds)

    def score(self, current: str, baseline: str, err_rate_delta: float, tool_mix_delta: float) -> float:
        s1 = ngram_kl(current, baseline)
        s2 = embedding_shift(current, baseline)
        s3 = behavior_signal(err_rate_delta, tool_mix_delta)
        # require concordance across at least 2 signals above their thresholds
        flags = [
            (s1 is not None and s1 >= self.th["kl"]),
            (s2 is not None and s2 >= self.th["emb"]),
            (s3 is not None and s3 >= self.th["beh"]),
        ]
        final = aggregate_score([s1, s2, s3]) if sum(flags) >= 2 else 0.0
        if self.metrics and hasattr(self.metrics, "egregore_anomaly"):
            try:
                self.metrics.egregore_anomaly.set(final)
            except Exception:
                pass
        return final

    def guard(self, *args, **kwargs):
        val = self.score(*args, **kwargs)
        if val >= self.th["final"]:
            if self.metrics and hasattr(self.metrics, "quarantine_events"):
                try:
                    self.metrics.quarantine_events.inc()
                except Exception:
                    pass
            raise Quarantine(f"Quarantined due to egregore anomaly={val:.3f}")
        return val
