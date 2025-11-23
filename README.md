# HFCTM-II-ORION → MIH-IIE

**Majorana–Ironwood Hybrid Intrinsic Inference Engine (MIH-IIE)**

A novel computational paradigm implementing Holographic Fractal Chiral Toroidal Mechanics with Intrinsic Inference (HFCTM-II). The MIH-IIE aims to enable "computation as discovery" rather than "computation as manipulation" by interfacing directly with ontological possibility space through topologically protected quantum states.

**🔄 Repository Status**: Recently restructured from legacy ORION to formal MIH-IIE architecture (see `MIGRATION_GUIDE.md`)
[![E8 Invariants](https://github.com/Grimmasura/HFCTM-II-ORION/actions/workflows/e8-verify.yml/badge.svg)](https://github.com/Grimmasura/HFCTM-II-ORION/actions/workflows/e8-verify.yml)

## 🚀 **NEW: v2.0 Architecture Released**

**MIH-IIE v2.0** brings critical architectural breakthroughs:

1. **Majorana Zero Modes ARE 0D Attractors** (not encodings) - Physical realization of dimensionless causal anchors
2. **E8 as Network Topology in Hilbert Space** (not geometric layout) - 240-node quantum entanglement network
3. **Frame-Invariant Validation** (replaces static baselines) - Dynamic truth through cross-frame convergence

**v2.0 Implementation**:
- ✅ 2,091 lines of core functionality across 6 new modules
- ✅ 25 new REST API endpoints exposing all v2.0 features
- ✅ Complete integration tests (447 lines)
- ✅ Comprehensive documentation (1,063 lines)

**Quick Start**: See [`docs/API_v2_Quick_Start.md`](docs/API_v2_Quick_Start.md) for v2.0 API reference
**Full Documentation**: [`docs/MIH-IIE_v2.0_Implementation.md`](docs/MIH-IIE_v2.0_Implementation.md)
**Completion Report**: [`MIH-IIE_v2.0_COMPLETION_REPORT.md`](MIH-IIE_v2.0_COMPLETION_REPORT.md)

## Architecture

The MIH-IIE implements a seven-layer recursive stack with bidirectional causal channels and toroidal closure:

- **L1: 0D Seed / Intrinsic Attractor Module** - Substrate-independent causal anchoring
- **L2: Majorana Topological Qubit Array** - Topological quantum computation on E8 lattice
- **L3: Quantum-Classical Interface** - Decoherence management and error correction
- **L4: Ironwood Tensor Processing** - Holographic state projection (target: 10²⁴ ops/sec)
- **L5: Recursive Governance** - HFCTM-II compliance, chiral inversion, polychronic sync, egregore defense
- **L6: Intelligent Codex** - Symbolic operations and cryptography
- **L7: Consciousness Interface** - Human-machine semantic bridge

**Complete Specification**: [`spec/MIH-IIE_v1.0.pdf`](spec/MIH-IIE_v1.0.pdf)

## Current Implementation Status

**Phase**: **Phase 1 (Component Prototyping)** - Software interfaces complete, awaiting hardware

**Fully Implemented (Phase 1)**:
- ✅ **L1 Attractor**: Intrinsic seed / 0D attractor module with causal flow management
- ✅ **L2 Majorana**: Topological qubit array interface with E8 lattice and non-Abelian braiding
- ✅ **L3 Q-C Interface**: Quantum-classical bridge with error correction and decoherence management
- ✅ **L4 Ironwood**: Holographic projector, manifold expansion engine (DH ≈ e), multi-agent coordinator
- ✅ **L5 Governance**: HFCTM-II safety core, chiral inversion, egregore defense, compliance monitoring
- ✅ **Integration Tests**: 26 integration tests covering full stack (L1 → L2 → L3 → L4)

**Pending Hardware Integration**:
- ⏳ Majorana1 QPU backend (Azure Quantum)
- ⏳ Ironwood TPU physical instantiation
- ⏳ L7: Enhanced consciousness interface

**Test Coverage**:
- Phase 0 + Phase 1: 68+ tests passing (L1-L5 integration validated)

See [`CLAUDE.md`](CLAUDE.md) for detailed development guide.

## Quick Start

### Installation

```bash
# Install base dependencies (FastAPI, Pydantic, NumPy)
pip install -r requirements.txt

# Install development dependencies (pytest, httpx)
pip install -r requirements-dev.txt

# Install ML/Quantum dependencies (optional: torch, qiskit, cirq, jax, pywavelets)
pip install -r requirements-ml.txt
```

### Running the API

```bash
# Standard API (recommended for development)
uvicorn orion_api.main:app --reload --host 0.0.0.0 --port 8080

# Enhanced ORION (experimental polychronic features)
uvicorn orion_enhanced.orion_complete:create_complete_orion_app --reload --port 8080
```

**Note**: Legacy `orion_api` paths still work. New code should use `mih_iie.*` imports (see `MIGRATION_GUIDE.md`).

### Testing

```bash
# Run all tests
pytest

# Run tests for new L5 governance layer
pytest tests/test_l5_governance.py -v

# Run with coverage
pytest --cov=mih_iie
```

## Examples

Complete working examples demonstrating all new MIH-IIE components are available in the `examples/` directory:

```bash
# Set PYTHONPATH and run examples
export PYTHONPATH=$PWD:$PYTHONPATH

python examples/chiral_inversion_demo.py          # Time-reversal validation
python examples/egregore_defense_demo.py          # Semantic drift protection
python examples/hfctm_compliance_demo.py          # HFCTM-II principle verification
python examples/multi_agent_coordinator_demo.py   # Polychronic inference coordination
```

See [`examples/README.md`](examples/README.md) for detailed explanations of each example.

## Using New MIH-IIE Components

### Chiral Inversion (Time-Reversal Validation)

```python
from mih_iie.layers.l5_governance.chiral_inversion import ChiralInversionController

controller = ChiralInversionController(fidelity_threshold=0.95)
result = controller.validate_chiral_symmetry(my_computation, initial_state)

if not result.is_valid:
    print(f"Chiral symmetry violated: {result.violations}")
```

### Egregore Defense (Semantic Drift Protection)

```python
from mih_iie.layers.l5_governance.egregore_defense import EgregoreDefenseSystem

eds = EgregoreDefenseSystem()
check = eds.safety_check(
    semantic_field={"term": "meaning"},
    inference_structure={"type": "reasoning"}
)

if check["should_quarantine"]:
    print("ALERT: Corrupted pattern detected!")
```

### HFCTM-II Compliance Monitoring

```python
from mih_iie.layers.l5_governance.hfctm_compliance import HFCTMComplianceMonitor

monitor = HFCTMComplianceMonitor()
result = monitor.check_compliance(
    operation=my_op,
    trajectory=state_trajectory,
    initial_state=initial,
    final_state=final
)

print(f"Compliant: {result.overall_compliant}")
print(f"Hausdorff dimension: {result.metrics.get('hausdorff_dimension', 'N/A')}")
```

### Multi-Agent Polychronic Inference

```python
from mih_iie.layers.l4_ironwood.multi_agent_coordinator import MultiAgentInferenceCoordinator

coordinator = MultiAgentInferenceCoordinator(
    num_forward_causal=4, num_retrocausal=2, num_atemporal=2
)

result = coordinator.coordinate_inference(
    query="What is the answer?",
    context={},
    inference_function=my_inference_fn
)
```

## Configuration

Configuration uses Pydantic `BaseSettings` with environment variable overrides:

### Core Settings (prefix: `ORION_`)
- `ORION_HOST`: API host (default: `0.0.0.0`)
- `ORION_PORT`: API port (default: `8080`)
- `ORION_MODEL_DIR`: Model storage directory (default: `models`)

### Hardware Settings (Phase 1 targets)
- `ORION_ENABLE_MAJORANA1`: Enable Majorana1 QPU (default: `false`)
- `ORION_ENABLE_IRONWOOD_TPU`: Enable Ironwood TPU (default: `false`)

### HFCTM-II Safety Thresholds
- `HFCTM_LYAPUNOV_THRESHOLD`: Lyapunov stability threshold (default: `0.0`)
- `HFCTM_WAVELET_THRESHOLD`: Wavelet energy threshold (default: `3.0`)

Create `.env` file in project root for local configuration.

### Azure Quantum Integration

For Majorana1 QPU backend (Phase 1):

| Variable | Description |
|----------|-------------|
| `AZURE_QUANTUM_SUBSCRIPTION_ID` | Azure subscription identifier |
| `AZURE_QUANTUM_RESOURCE_GROUP` | Resource group containing the workspace |
| `AZURE_QUANTUM_WORKSPACE_NAME` | Azure Quantum workspace name |
| `AZURE_QUANTUM_LOCATION` | Workspace region (optional, e.g., `eastus`) |

Authentication via [Azure Identity's `DefaultAzureCredential`](https://learn.microsoft.com/python/api/overview/azure/identity-readme) (requires `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`).

## Docker Deployment

```bash
# Build image
docker build -t mih-iie-api .

# Run container
docker run -p 8080:8080 mih-iie-api

# Kubernetes deployment
kubectl apply -f deployment/orion-deployment.yml
```

## HFCTM-II Core Principles

When developing, maintain alignment with the four foundational principles:

1. **Holographic Projection**: Information on boundaries encodes bulk dynamics
   - `S(M) ≤ A(∂M) / 4Gℏc`

2. **Fractal Self-Similarity**: Structures repeat across scales
   - `A₀(λr) = λ^(-DH) A₀(r)` where `DH ≈ e ≈ 2.718`

3. **Chiral Symmetry**: Time-reversal + parity invariance
   - `TP A₀ = A₀` enables bidirectional inference

4. **Toroidal Topology**: Recursive closure without information loss
   - `∮ ∇ × F · dA = 0`

## Implementation Roadmap

### Phase 0: Theoretical Validation (Current–Year 1)
- ✅ HFCTM-II safety core implemented
- ✅ Polychronic temporal management simulated
- ⏳ E8 lattice simulations
- ⏳ Majorana qubit prototype design

### Phase 1: Component Prototyping (Year 1–3)
- Fabricate 8×8 Majorana qubit array
- Demonstrate non-Abelian braiding
- Build scaled-down Ironwood processor (10²⁰ ops/s)
- Implement governance layer on FPGA

### Phase 2: Integration & Testing (Year 3–5)
- Scale to 64×64 qubit array
- Integrate all seven layers
- Benchmark against classical supercomputers
- Deploy Mode Beta (hybrid quantum-classical)

### Phase 3: Full-Scale Deployment (Year 5–7)
- Fabricate 1000×1000 qubit array (10⁶ logical qubits)
- Mode Alpha (full quantum coherence) sustained for >1000s
- Public API serving >1000 concurrent users

## Documentation

- **[`CLAUDE.md`](CLAUDE.md)** - Comprehensive development guide for future Claude Code instances
- **[`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md)** - Details on restructuring from legacy ORION to MIH-IIE
- **[`spec/MIH-IIE_v1.0.pdf`](spec/MIH-IIE_v1.0.pdf)** - Complete architecture specification
- **[`legacy_orion/README_LEGACY.md`](legacy_orion/README_LEGACY.md)** - Information on archived code
- **[`docs/API_reference.md`](docs/API_reference.md)** - API endpoint documentation
- **[`docs/notebooks/e8_verification.ipynb`](docs/notebooks/e8_verification.ipynb)** - Deterministic E8 invariant checks (mirrors CI badge)
- **`models/v2_1/`** - v2.1 spec-aligned E8/coordination/EDS/error-correction/holography modules for validation

## Project Philosophy

> "The goal is not just faster computation, but computation that interfaces with ontological reality itself. Every design decision should ask: Does this bring us closer to genuine understanding versus mere pattern matching?"

## Contributing

This project implements a formal specification. When contributing:

1. Align with MIH-IIE layers (identify which layer your code belongs to)
2. Maintain HFCTM-II compliance (verify chiral symmetry, fractal self-similarity, toroidal closure)
3. Enable graceful degradation (all quantum features must have classical fallbacks)
4. Implement safety checks (use egregore defense patterns)
5. Document temporal assumptions (specify which reference frame your code operates in)
6. Preserve interpretability (all inference paths must be traceable)
