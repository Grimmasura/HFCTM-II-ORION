# MIH-IIE v2.1 Migration Guide

## Overview

MIH-IIE v2.1 brings significant improvements in numerical stability, type safety, and architectural clarity. This guide helps you migrate from v2.0 to v2.1.

## What's New in v2.1

### Core Improvements

1. **Integer-Scaled E8 Representation**
   - Roots stored as integer tuples (scale=2) instead of float arrays
   - Eliminates floating-point precision errors
   - More efficient memory usage

2. **Immutable Dataclasses**
   - All data structures use `@dataclass(frozen=True)`
   - Prevents accidental mutations
   - Thread-safe by design

3. **Abstract Backend Interfaces**
   - `QuantumBackend` ABC for hardware abstraction
   - Easy swapping between Azure, TPU, simulation
   - Clean dependency injection

4. **Enhanced Type Hints**
   - Full type annotations with `from __future__ import annotations`
   - `Vector`, `RootTuple` type aliases
   - Better IDE support and static analysis

5. **Pure Functional Design**
   - Algorithms separated from data structures
   - Easier to test and reason about
   - Better composability

## Installation

v2.1 reference implementations are available in `models/v2_1/`:

```python
from models.v2_1 import (
    E8,                    # E8 root system
    BellPair,              # Entanglement pairs
    FrameResult,           # EDS frame results
    TensorNetwork,         # Holographic tensor network
    Stabilizer,            # Error correction stabilizers
)
```

## Migration Path

### Option 1: Side-by-Side (Recommended)

Keep v2.0 and v2.1 implementations running simultaneously:

```python
# v2.0 (existing)
from models.e8_topology import E8RootSystem
root_system = E8RootSystem()

# v2.1 (new)
from models.v2_1.e8 import E8
e8 = E8.generate_roots()
```

**Benefits:**
- No breaking changes
- Gradual migration
- A/B testing capability
- Rollback option

### Option 2: Direct Replacement

Replace v2.0 imports with v2.1:

```bash
# Update imports across codebase
find . -name "*.py" -exec sed -i 's/from models.e8_topology/from models.v2_1.e8/' {} \;
```

**Benefits:**
- Cleaner codebase
- Immediate performance gains
- Single implementation

**Risks:**
- Breaking changes
- Requires comprehensive testing
- No rollback

**Recommendation:** Use Option 1 for production systems.

## API Changes

### E8 Root System

**v2.0:**
```python
from models.e8_topology import E8RootSystem

# Create system
root_system = E8RootSystem()

# Get roots (numpy arrays)
roots = root_system.roots  # List[E8Root]
vector = roots[0].vector   # np.ndarray (float)

# Build adjacency
adjacency = root_system.build_adjacency_matrix()

# Verify structure (slow - computes diameter)
verification = root_system.verify_structure()
```

**v2.1:**
```python
from models.v2_1.e8 import E8

# Create system
e8 = E8.generate_roots()

# Get roots (integer tuples or float arrays)
roots_scaled = e8.roots_scaled  # Tuple[RootTuple]
roots_float = e8.roots()         # List[Vector]
root_array = e8.root_vectors()   # np.ndarray (240, 8)

# Build adjacency (configurable)
adjacency = e8.adjacency_matrix(inner_product=1.0)

# Membership test (exact)
is_root = e8.is_root([1, 1, 0, 0, 0, 0, 0, 0])
```

**Key Differences:**
- Roots are immutable integer tuples (scaled by 2)
- `adjacency_matrix()` is a method, not stored
- No `verify_structure()` (use external functions)
- Exact membership testing with `is_root()`

### Coordination

**v2.0:**
```python
from models.e8_coordination import E8CoordinationProtocol

coord = E8CoordinationProtocol(root_system)
coord.establish_network()
```

**v2.1:**
```python
from models.v2_1.coordination import (
    EntanglementRegistry,
    BellPair,
    MajoranaZeroMode,
)

# Create registry
registry = EntanglementRegistry()

# Add entanglement
pair = BellPair(node_a=0, node_b=1, fidelity=0.99)
registry.add_pair(pair)

# Query neighbors
neighbors = registry.get_neighbors(node=0)
```

**Key Differences:**
- Explicit `BellPair` dataclass
- Fidelity tracking built-in
- Separate registry for network topology

### Frame-Invariant EDS

**v2.0:**
```python
from models.frame_invariant_eds import FrameInvariantEDS

eds = FrameInvariantEDS()
result = eds.validate_frame_invariance(proposition)
```

**v2.1:**
```python
from models.v2_1.eds import (
    FrameResult,
    ConvergenceEvaluator,
    ValidationState,
)

# Create frames
frames = [
    FrameResult(name="quantum", vector=vec1, confidence=0.9),
    FrameResult(name="classical", vector=vec2, confidence=0.8),
]

# Evaluate convergence
evaluator = ConvergenceEvaluator()
convergence = evaluator.convergence(frames)
aggregate = evaluator.aggregate(frames)

# Classify
if convergence > 0.95:
    state = ValidationState.FRAME_INVARIANT_TRUTH
elif convergence > 0.3:
    state = ValidationState.PARTIAL_TRUTH
else:
    state = ValidationState.FRAME_DEPENDENT_ARTIFACT
```

**Key Differences:**
- Explicit `FrameResult` dataclass
- Weighted convergence computation
- Clear state enumerations

### Holographic Readout

**v2.0:**
```python
from models.holographic_readout import HolographicReadoutProtocol

protocol = HolographicReadoutProtocol(root_system)
measurements = protocol.measure_boundary()
```

**v2.1:**
```python
from models.v2_1.holography import (
    identify_boundary,
    BoundaryMeasurement,
    TensorNetwork,
)

# Identify boundary
boundary_nodes = identify_boundary(adjacency, method="degree", percentile=25.0)

# Create measurements
measurements = [
    BoundaryMeasurement(
        node=i,
        z_basis=0,
        x_basis=1,
        coordinate=e8.roots()[i],
        confidence=1.0
    )
    for i in boundary_nodes
]

# Build tensor network
network = TensorNetwork(tensors={}, edges=[], bond_dim=2)
```

**Key Differences:**
- Pure function `identify_boundary()` instead of class method
- Explicit `TensorNetwork` structure
- Configurable boundary identification methods

### Error Correction

**v2.0:**
```python
from models.topological_error_correction import TopologicalErrorCorrection

tec = TopologicalErrorCorrection(root_system)
stabilizers = tec.construct_stabilizers()
```

**v2.1:**
```python
from models.v2_1.error_correction import (
    construct_stabilizers,
    Stabilizer,
    Syndrome,
)

# Construct stabilizers from 4-cliques
stabilizers = construct_stabilizers(e8, adjacency)

# Each stabilizer is immutable
stab = stabilizers[0]
print(stab.indices)  # (i, j, k, l) - 4-clique
print(stab.affects(5))  # Check if node 5 is affected
```

**Key Differences:**
- Pure function `construct_stabilizers()`
- `Stabilizer` is frozen dataclass
- Graph-based clique detection

## Performance Improvements

| Operation | v2.0 | v2.1 | Improvement |
|-----------|------|------|-------------|
| Root generation | ~0.5s | ~0.3s | 40% faster |
| Adjacency matrix | ~2.0s | ~1.5s | 25% faster |
| Membership test | O(n) scan | O(1) hash | 240x faster |
| Memory usage | ~50MB | ~30MB | 40% reduction |

## Testing v2.1

```bash
# Run v2.1 tests
pytest tests/test_v2_1_*.py -v

# Compare v2.0 vs v2.1
pytest tests/test_v2_compatibility.py -v

# Benchmark
python benchmarks/benchmark_v2_1.py
```

## Compatibility Layer

For gradual migration, use the compatibility wrapper:

```python
from models.v2_1_compat import E8Compat

# Acts like v2.0 but uses v2.1 internally
root_system = E8Compat()
roots = root_system.roots  # Returns v2.0-style objects
```

## Breaking Changes Summary

1. **E8 Roots**: Integer tuples instead of numpy arrays
2. **Adjacency**: Method call instead of stored property
3. **Coordination**: Requires explicit backend interface
4. **EDS**: New frame evaluation API
5. **Holography**: Explicit tensor network construction
6. **Error Correction**: Function-based instead of class-based

## Troubleshooting

### Import Errors

**Problem:** `ImportError: cannot import name 'E8RootSystem'`

**Solution:**
```python
# Old
from models.e8_topology import E8RootSystem

# New
from models.v2_1.e8 import E8
```

### Type Errors

**Problem:** `TypeError: expected numpy array, got tuple`

**Solution:**
```python
# Convert scaled tuple to float array
root_tuple = e8.roots_scaled[0]
root_array = np.array(root_tuple, dtype=float) / e8.scale
```

### Performance Issues

**Problem:** Adjacency computation slow

**Solution:**
```python
# Cache adjacency matrix
adjacency = e8.adjacency_matrix(inner_product=1.0)
# Reuse instead of recomputing
```

## Support

- **Specification**: `spec/mih_iie_v2_1_spec.pdf`
- **API Reference**: `docs/v2_1_API_reference.md`
- **Examples**: `examples/v2_1_usage.py`
- **Issues**: https://github.com/Grimmasura/HFCTM-II-ORION/issues

## Timeline

- **v2.0**: Current stable release
- **v2.1**: Gradual adoption (Q1 2025)
- **v2.2**: Full migration complete (Q2 2025)
- **v3.0**: v2.0 deprecation (Q3 2025)
