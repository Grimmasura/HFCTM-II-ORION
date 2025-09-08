"""Chiral inversion trainer with numerical stability guards."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


eps = 1e-6


def _safe_probs(p: torch.Tensor) -> torch.Tensor:
    """Ensure probabilities are valid and normalized."""
    p = p + eps
    return p / p.sum(dim=-1, keepdim=True)


@dataclass
class ChiralTrainer:
    """Minimal trainer computing JS divergence between policies."""

    def js_divergence(self, policy_orig: torch.Tensor, policy_chiral_mapped: torch.Tensor) -> torch.Tensor:
        p1 = _safe_probs(F.softmax(policy_orig, dim=-1))
        p2 = _safe_probs(F.softmax(policy_chiral_mapped, dim=-1))
        m = 0.5 * (p1 + p2)
        js = 0.5 * (
            F.kl_div(p1.log(), m, reduction="batchmean")
            + F.kl_div(p2.log(), m, reduction="batchmean")
        )
        return js


__all__ = ["ChiralTrainer"]
