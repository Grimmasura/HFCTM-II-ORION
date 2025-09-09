import os
import math
import numpy as np
from collections import Counter
from functools import lru_cache
import time

LAMBDA1_THRESHOLD = float(os.getenv("ORION_EGREGORE_LAMBDA1_THRESHOLD", "3.2"))

class EgregoreDefense:
    def __init__(self, metrics=None):
        self.metrics = metrics
        self.thresholds = {
            "kl": 1.2,
            "embedding": 0.28,
            "behavior": 0.35,
            "final": 0.4,
        }
        self._emb_model = None

    def _kl_divergence(self, current: str, baseline: str, n: int = 3) -> float:
        def ngrams(text: str):
            toks = text.split()
            return Counter(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))
        p = ngrams(current)
        q = ngrams(baseline)
        k = 1.0
        vocab = set(p) | set(q)
        P = [(p[t] + k) for t in vocab]
        Q = [(q[t] + k) for t in vocab]
        Zp, Zq = sum(P), sum(Q)
        return sum((pi/Zp) * math.log((pi/Zp) / (qi/Zq)) for pi, qi in zip(P, Q))

    @lru_cache(maxsize=2048)
    def _embed_once(self, text: str):
        if self._emb_model is None:
            from sentence_transformers import SentenceTransformer
            self._emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._emb_model.encode([text])[0]

    def _embedding_shift(self, current: str, baseline: str):
        try:
            v1 = self._embed_once(current)
            v2 = self._embed_once(baseline)
            dot = float((v1 * v2).sum())
            den = float(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            return 1.0 - (dot / den)
        except Exception:
            return None

    def score_anomaly(self, current: str, baseline: str, err_rate_delta: float = 0.0, tool_mix_delta: float = 0.0) -> float:
        t0 = time.time()
        kl_score = self._kl_divergence(current, baseline)
        t1 = time.time()
        emb_score = self._embedding_shift(current, baseline)
        t2 = time.time()
        behavior_score = 0.5 * abs(err_rate_delta) + 0.5 * abs(tool_mix_delta)

        flags = [
            (kl_score is not None and kl_score >= self.thresholds["kl"]),
            (emb_score is not None and emb_score >= self.thresholds["embedding"]),
            (behavior_score >= self.thresholds["behavior"]),
        ]

        if sum(flags) >= 2:
            vals = [v for v in (kl_score, emb_score, behavior_score) if isinstance(v, (int, float))]
            final_score = float(sum(vals) / len(vals)) if vals else 0.0
        else:
            final_score = 0.0

        if self.metrics:
            try:
                self.metrics.egregore_anomaly.set(final_score)
                if hasattr(self.metrics, "recursion_depth"):
                    self.metrics.recursion_depth.observe(max(1, int((t1 - t0) * 1000)))
                    self.metrics.recursion_depth.observe(max(1, int((t2 - t1) * 1000)))
            except Exception:
                pass
        return final_score

    def should_quarantine(self, *args, **kwargs) -> bool:
        score = self.score_anomaly(*args, **kwargs)
        should_gate = score >= self.thresholds["final"]
        if should_gate and self.metrics:
            try:
                self.metrics.quarantine_events.inc()
            except Exception:
                pass
        return should_gate
