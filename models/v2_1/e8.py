# orion/e8.py
"""
E8 Math Engine for MIH-IIE v2.0
Complete implementation of E8 root system, adjacency, Weyl actions, and graph utilities.
Implements Algorithms 1-2 from the MIH-IIE specification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict, Set, Optional
import itertools
import numpy as np

Vector = np.ndarray
RootTuple = Tuple[int, ...]


def _normalize_int_vec(v: Vector, scale: int = 2) -> RootTuple:
    """
    Convert a float/half-int vector in R^8 to an integer tuple by scaling.
    E8 roots are either in Z^8 or (Z+1/2)^8 with even sum.
    We scale by 2 to represent half-integers as ints.
    """
    w = np.rint(v * scale).astype(int)
    return tuple(int(x) for x in w)


def _dot_scaled(a: RootTuple, b: RootTuple) -> int:
    """Dot product in scaled integer space (scale=2)."""
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class E8:
    """
    E8 root system engine.

    Representation:
      - Roots stored as integer tuples in scaled space (scale=2).
        So actual root vector r = t / 2.
      - Uses standard E8 root construction:
          Type A (112): permutations of (±1, ±1, 0,...,0)
          Type B (128): all 8-vectors with entries ±1/2 and even number of minus signs
    """
    roots_scaled: Tuple[RootTuple, ...]
    scale: int = 2

    @staticmethod
    def generate_roots() -> "E8":
        """
        Algorithm 1: Generate E8 Root Vectors
        
        Type I (112 vectors): All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        Type II (128 vectors): All vectors (±1/2)^8 with even number of minus signs
        """
        roots: Set[RootTuple] = set()

        # Type A: (±1, ±1, 0^6) - 112 vectors
        for i, j in itertools.combinations(range(8), 2):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    v = np.zeros(8, dtype=float)
                    v[i] = s1
                    v[j] = s2
                    roots.add(_normalize_int_vec(v, 2))

        # Type B: (±1/2)^8 with even number of minus signs - 128 vectors
        for signs in itertools.product((-1, 1), repeat=8):
            # Count minus signs
            minus_count = sum(1 for s in signs if s == -1)
            if minus_count % 2 == 0:  # even number of minus signs
                v = np.array(signs, dtype=float) * 0.5
                roots.add(_normalize_int_vec(v, 2))

        roots_list = tuple(sorted(roots))
        if len(roots_list) != 240:
            raise ValueError(f"E8 root generation failed: got {len(roots_list)} roots, expected 240")
        return E8(roots_scaled=roots_list)

    def roots(self) -> List[Vector]:
        """Return roots in unscaled float space."""
        return [np.array(r, dtype=float) / self.scale for r in self.roots_scaled]

    def root_vectors(self) -> np.ndarray:
        """Return all roots as (240, 8) array."""
        return np.array([np.array(r, dtype=float) / self.scale for r in self.roots_scaled])

    def is_root(self, v: Sequence[float], tol: float = 1e-9) -> bool:
        """Check membership by scaling to integer tuple."""
        t = _normalize_int_vec(np.array(v, dtype=float), self.scale)
        return t in set(self.roots_scaled)

    def adjacency_matrix(self, inner_product: float = 1.0) -> np.ndarray:
        """
        Algorithm 2: Build E8 Adjacency Matrix
        
        Build adjacency where <r_i, r_j> = inner_product.
        In scaled space: <t_i/2, t_j/2> = inner_product ==> dot_scaled = 4*inner_product.
        For inner_product=1, dot_scaled target = 4.
        
        Returns:
            A: (240, 240) symmetric adjacency matrix with A[i,j]=1 iff <r_i,r_j>=inner_product
        """
        target = int(round((self.scale ** 2) * inner_product))
        n = len(self.roots_scaled)
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            ri = self.roots_scaled[i]
            for j in range(i + 1, n):
                rj = self.roots_scaled[j]
                if _dot_scaled(ri, rj) == target:
                    A[i, j] = 1
                    A[j, i] = 1
        return A

    def adjacency_list(self, inner_product: float = 1.0) -> List[List[int]]:
        """Return adjacency as list of neighbor indices."""
        A = self.adjacency_matrix(inner_product=inner_product)
        return [list(np.where(A[i] == 1)[0]) for i in range(A.shape[0])]

    def degree_sequence(self, inner_product: float = 1.0) -> np.ndarray:
        """Return degree of each node."""
        A = self.adjacency_matrix(inner_product=inner_product)
        return A.sum(axis=1)

    # ----- Simple Roots and Weyl Group -----

    @staticmethod
    def simple_roots() -> List[Vector]:
        """
        Return the 8 simple roots of E8 (Definition 6.1).
        These generate the Weyl group through reflections.
        """
        alpha = [
            np.array([1, -1, 0, 0, 0, 0, 0, 0], dtype=float),  # α1
            np.array([0, 1, -1, 0, 0, 0, 0, 0], dtype=float),  # α2
            np.array([0, 0, 1, -1, 0, 0, 0, 0], dtype=float),  # α3
            np.array([0, 0, 0, 1, -1, 0, 0, 0], dtype=float),  # α4
            np.array([0, 0, 0, 0, 1, -1, 0, 0], dtype=float),  # α5
            np.array([0, 0, 0, 0, 0, 1, -1, 0], dtype=float),  # α6
            np.array([0, 0, 0, 0, 0, 1, 1, 0], dtype=float),   # α7
            np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5], dtype=float),  # α8
        ]
        return alpha

    def reflection_matrix(self, alpha_scaled: RootTuple) -> np.ndarray:
        """
        Definition 6.2: Weyl reflection s_alpha
        
        s_α(v) = v - 2*(v·α)/(α·α) * α
        
        For E8 roots, α·α = 2.
        Returns the 8x8 reflection matrix.
        """
        alpha = np.array(alpha_scaled, dtype=float) / self.scale
        denom = float(np.dot(alpha, alpha))
        if abs(denom - 2.0) > 1e-6:
            raise ValueError(f"alpha not a root; norm² = {denom}, expected 2.0")
        I = np.eye(8)
        outer = np.outer(alpha, alpha)
        S = I - 2.0 * outer / denom
        return S

    def apply_weyl_reflection(
        self,
        v: Sequence[float],
        alpha_scaled: RootTuple
    ) -> Vector:
        """Apply Weyl reflection s_α to vector v."""
        S = self.reflection_matrix(alpha_scaled)
        v = np.array(v, dtype=float)
        return S @ v

    def verify_weyl_action_on_roots(
        self,
        S: np.ndarray,
        tol: float = 1e-7
    ) -> bool:
        """
        Algorithm 5: Verify Symmetry Preservation
        
        Verify that matrix S preserves root system: S*r ∈ ±Roots for all r.
        """
        roots = self.roots()
        root_set = set(self.roots_scaled)
        for r in roots:
            r2 = S @ r
            t2 = _normalize_int_vec(r2, self.scale)
            t2_neg = tuple(-x for x in t2)
            if t2 not in root_set and t2_neg not in root_set:
                return False
        return True

    def compose_reflections(self, alphas_scaled: List[RootTuple]) -> np.ndarray:
        """Compose multiple Weyl reflections into single matrix."""
        result = np.eye(8)
        for alpha in alphas_scaled:
            S = self.reflection_matrix(alpha)
            result = S @ result
        return result

    # ----- Graph Utilities -----

    def find_k_cliques(
        self,
        k: int = 4,
        inner_product: float = 1.0,
        limit: Optional[int] = None
    ) -> List[Tuple[int, ...]]:
        """
        Algorithm 8: Construct E8 Stabilizers (partial)
        
        Enumerate k-cliques (complete subgraphs of size k).
        For k=4, these form the stabilizer generators S_ijkl = γ_i γ_j γ_k γ_l.
        """
        adj = self.adjacency_list(inner_product=inner_product)
        n = len(adj)
        cliques: List[Tuple[int, ...]] = []
        
        if k == 4:
            # Optimized 4-clique enumeration
            for a in range(n):
                na = set(adj[a])
                for b in na:
                    if b <= a:
                        continue
                    nab = na.intersection(adj[b])
                    for c in nab:
                        if c <= b:
                            continue
                        nabc = nab.intersection(adj[c])
                        for d in nabc:
                            if d <= c:
                                continue
                            cliques.append((a, b, c, d))
                            if limit is not None and len(cliques) >= limit:
                                return cliques
        else:
            # General k-clique finder using Bron-Kerbosch
            cliques = self._bron_kerbosch_k(adj, k, limit)
        
        return cliques

    def _bron_kerbosch_k(
        self,
        adj: List[List[int]],
        k: int,
        limit: Optional[int]
    ) -> List[Tuple[int, ...]]:
        """Bron-Kerbosch algorithm for k-cliques."""
        results: List[Tuple[int, ...]] = []
        n = len(adj)
        adj_set = [set(neighbors) for neighbors in adj]
        
        def bk(R: Set[int], P: Set[int], X: Set[int]):
            if limit and len(results) >= limit:
                return
            if len(R) == k:
                results.append(tuple(sorted(R)))
                return
            if len(R) + len(P) < k:
                return
            for v in list(P):
                new_R = R | {v}
                new_P = P & adj_set[v]
                new_X = X & adj_set[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return results

    def graph_diameter(self, inner_product: float = 1.0) -> int:
        """Compute diameter via BFS."""
        adj = self.adjacency_list(inner_product=inner_product)
        n = len(adj)

        def bfs(src: int) -> int:
            dist = [-1] * n
            dist[src] = 0
            q = [src]
            for u in q:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            return max(dist)

        return max(bfs(i) for i in range(n))

    def find_opposite_nodes(self, nodes: List[int]) -> List[int]:
        """
        Find antipodal nodes in E8 (negative roots).
        Used in Algorithm 13 for inference computation.
        """
        root_set = {r: i for i, r in enumerate(self.roots_scaled)}
        opposites = []
        for node in nodes:
            neg_root = tuple(-x for x in self.roots_scaled[node])
            if neg_root in root_set:
                opposites.append(root_set[neg_root])
        return opposites

    def find_closest_nodes(self, query_vector: Vector, n_closest: int = 8) -> List[int]:
        """
        Find nodes closest to query vector in E8 space.
        Used for query encoding in inference.
        """
        roots = self.root_vectors()
        # Cosine similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm < 1e-10:
            return list(range(n_closest))
        
        similarities = roots @ query_vector / (np.linalg.norm(roots, axis=1) * query_norm + 1e-10)
        return list(np.argsort(similarities)[-n_closest:])

    # ----- Substructure Extraction -----

    def extract_substructure(
        self,
        n_nodes: int,
        method: str = "spectral"
    ) -> Tuple[List[int], np.ndarray]:
        """
        Algorithm 4: Extract E8 Substructure
        
        For implementations with fewer than 240 nodes, extract maximally
        symmetric subgraphs using spectral clustering.
        """
        A = self.adjacency_matrix()
        
        if method == "spectral":
            # Compute principal eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(A.astype(float))
            # Take top 8 eigenvectors
            V = eigenvectors[:, -8:]
            
            # K-means clustering
            from scipy.cluster.vq import kmeans2
            n_clusters = max(1, n_nodes // 8)
            centroids, labels = kmeans2(V, n_clusters, minit='points')
            
            # Select highest internal degree node from each cluster
            selected = []
            for c in range(n_clusters):
                cluster_nodes = np.where(labels == c)[0]
                if len(cluster_nodes) == 0:
                    continue
                # Compute internal degree
                internal_degrees = A[np.ix_(cluster_nodes, cluster_nodes)].sum(axis=1)
                best = cluster_nodes[np.argmax(internal_degrees)]
                selected.append(best)
            
            # Pad if needed
            while len(selected) < n_nodes:
                remaining = set(range(240)) - set(selected)
                if not remaining:
                    break
                selected.append(min(remaining))
            
            selected = selected[:n_nodes]
        else:
            # Simple: take first n_nodes by index
            selected = list(range(n_nodes))
        
        # Extract induced subgraph
        selected_arr = np.array(selected)
        subgraph = A[np.ix_(selected_arr, selected_arr)]
        
        return selected, subgraph


# Module-level convenience functions
def generate_e8() -> E8:
    """Generate the E8 root system."""
    return E8.generate_roots()


if __name__ == "__main__":
    e8 = E8.generate_roots()
    deg = e8.degree_sequence()
    print(f"E8 Root System Generated:")
    print(f"  Roots: {len(e8.roots_scaled)}")
    print(f"  Degree (unique): {set(deg.tolist())}")
    print(f"  Diameter: {e8.graph_diameter()}")
    print(f"  4-cliques (first 10): {len(e8.find_k_cliques(limit=10))}")
    print(f"  Simple roots count: {len(e8.simple_roots())}")
