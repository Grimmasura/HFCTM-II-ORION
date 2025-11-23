"""
Lightweight E8 invariant checks used by CI and notebooks.

Validates root counts, norms, inner products, and adjacency degree regularity.
The functions are intentionally deterministic and avoid expensive graph metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from models.e8_topology import E8RootSystem


@dataclass
class E8InvariantReport:
    num_roots: int
    unique_norms: Set[float]
    inner_products: Set[float]
    degrees: List[int]
    degree_distribution: Dict[int, int]
    adjacency_density: float

    def summary(self) -> Dict[str, object]:
        """Convert to a JSON-friendly summary."""
        return {
            "num_roots": self.num_roots,
            "unique_norms": sorted(self.unique_norms),
            "inner_products": sorted(self.inner_products),
            "degrees_unique": sorted(set(self.degrees)),
            "degree_distribution": self.degree_distribution,
            "adjacency_density": self.adjacency_density,
        }


def _safe_unique(values: Iterable[float], atol: float = 1e-6) -> Set[float]:
    """Return a set of unique floats rounded to tolerance to avoid fp drift."""
    return set(round(float(v), 6) for v in values)


def compute_e8_invariants(root_system: Optional[E8RootSystem] = None) -> E8InvariantReport:
    """
    Compute core invariants of the E8 root system.

    Returns counts, norm set, inner product spectrum (off-diagonal),
    and adjacency degree distribution (<r_i, r_j> = 1).
    """
    rs = root_system or E8RootSystem()
    vectors = np.stack([r.vector for r in rs.roots])

    # Norms and inner products
    norms = np.einsum("ij,ij->i", vectors, vectors)
    gram = vectors @ vectors.T

    # Remove diagonal from inner products to only inspect distinct root pairs
    off_diag = gram - np.eye(gram.shape[0]) * np.diag(gram)
    inner_products = _safe_unique(off_diag[np.triu_indices_from(off_diag, k=1)])

    # Adjacency defined by <r_i, r_j> = 1
    adjacency = np.isclose(off_diag, 1.0, atol=1e-6).astype(np.int8)
    degrees = adjacency.sum(axis=1).tolist()
    degree_distribution: Dict[int, int] = {}
    for d in degrees:
        degree_distribution[d] = degree_distribution.get(d, 0) + 1

    total_edges = adjacency.sum() // 2  # undirected
    n = len(rs.roots)
    adjacency_density = total_edges / (n * (n - 1) / 2)

    return E8InvariantReport(
        num_roots=n,
        unique_norms=_safe_unique(norms),
        inner_products=inner_products,
        degrees=degrees,
        degree_distribution=degree_distribution,
        adjacency_density=float(adjacency_density),
    )


def verify_e8_invariants(
    report: Optional[E8InvariantReport] = None,
    expected_degree: int = 56,
) -> Dict[str, bool]:
    """
    Validate basic E8 invariants with crisp booleans for CI/tests.

    - 240 roots
    - Norms all 2
    - Off-diagonal inner products in {-2, -1, 0, 1}
    - Degree regularity at 56 when adjacency is defined by <r_i, r_j> = 1
    """
    r = report or compute_e8_invariants()
    inner_expected = {-2.0, -1.0, 0.0, 1.0}

    return {
        "root_count": r.num_roots == 240,
        "norms": r.unique_norms == {2.0},
        "inner_products": r.inner_products.issubset(inner_expected)
        and not inner_expected.isdisjoint(r.inner_products),
        "degree_regular": set(r.degrees) == {expected_degree},
    }
