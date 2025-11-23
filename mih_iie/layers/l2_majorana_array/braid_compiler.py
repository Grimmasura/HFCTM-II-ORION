"""
Minimal braid compiler for E8 simple reflections.

Provides a deterministic way to generate and normalize braid words over the
E8 Coxeter presentation. Intended for software-level checks and examples,
not hardware pulse synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from models.e8_topology import E8RootSystem


def build_e8_coxeter_matrix() -> Dict[Tuple[int, int], int]:
    """
    Build Coxeter matrix m_ij from the standard simple roots in models.e8_topology.

    m_ij = 3 when roots are connected (inner product = -1), else 2 when orthogonal.
    Indexing is 1-based to align with braid word notation (s1..s8).
    """
    rs = E8RootSystem()
    simples = rs.get_simple_roots()
    matrix: Dict[Tuple[int, int], int] = {}

    for i, alpha_i in enumerate(simples, start=1):
        for j, alpha_j in enumerate(simples, start=1):
            if i == j:
                continue
            ip = float(np.dot(alpha_i.vector, alpha_j.vector))
            if np.isclose(ip, -1.0, atol=1e-6):
                matrix[(i, j)] = 3
            elif np.isclose(ip, 0.0, atol=1e-6):
                matrix[(i, j)] = 2
            else:
                raise ValueError(f"Unexpected inner product between simple roots {i},{j}: {ip}")
    return matrix


@dataclass(frozen=True)
class BraidWord:
    """Immutable braid word over generators s1..s8."""

    generators: Tuple[int, ...]

    def __str__(self) -> str:
        if not self.generators:
            return "identity"
        return " ".join(f"s{g}" for g in self.generators)

    def __len__(self) -> int:
        return len(self.generators)

    def multiply(self, other: "BraidWord") -> "BraidWord":
        """Concatenate two braid words."""
        return BraidWord(self.generators + other.generators)

    def normalize(self, coxeter: Dict[Tuple[int, int], int]) -> "BraidWord":
        """
        Apply local braid moves to reach a deterministic normal form.

        - m_ij = 2: commute generators (swap to ascending order)
        - m_ij = 3: rewrite i j i ↔ j i j toward lexicographically smaller form
        This is sufficient to check standard relations in tests/examples.
        """
        word = list(self.generators)
        changed = True

        while changed:
            changed = False
            i = 0
            while i < len(word) - 1:
                a, b = word[i], word[i + 1]
                m = coxeter.get((a, b))

                # Commuting case
                if m == 2 and a > b:
                    word[i], word[i + 1] = b, a
                    changed = True
                    if i:
                        i -= 1
                    continue

                # Braid move for m=3
                if m == 3 and i + 2 < len(word):
                    triplet = word[i : i + 3]
                    pattern_a = [a, b, a]
                    pattern_b = [b, a, b]
                    if triplet == pattern_a or triplet == pattern_b:
                        replacement = pattern_a if tuple(pattern_a) < tuple(pattern_b) else pattern_b
                        if tuple(triplet) != tuple(replacement):
                            word[i : i + 3] = replacement
                            changed = True
                            if i:
                                i -= 1
                            continue
                i += 1

        return BraidWord(tuple(word))


def simple_reflection_basis() -> List[BraidWord]:
    """Return the eight simple reflections as single-generator words."""
    return [BraidWord((i,)) for i in range(1, 9)]


def braid_word_from_generators(generators: Iterable[int]) -> BraidWord:
    """Create a braid word from a sequence of generator indices."""
    return BraidWord(tuple(int(g) for g in generators))


def compile_reflection_sequence(
    generators: Sequence[int],
    coxeter: Dict[Tuple[int, int], int] | None = None,
) -> BraidWord:
    """
    Build and normalize a braid word for the given simple reflection sequence.

    Args:
        generators: Iterable of generator indices (1-based, s1..s8).
        coxeter: Optional Coxeter matrix; defaults to computed E8 matrix.
    """
    coxeter = coxeter or build_e8_coxeter_matrix()
    word = braid_word_from_generators(generators)
    return word.normalize(coxeter)
