# MIH-IIE Examples

This directory contains example scripts demonstrating the key components of the Majorana–Ironwood Hybrid Intrinsic Inference Engine.

## Running Examples

All examples can be run directly from the project root:

```bash
# Make sure you're in the project root
cd /path/to/HFCTM-II-ORION

# Set PYTHONPATH to enable imports
export PYTHONPATH=$PWD:$PYTHONPATH

# Run examples
python examples/chiral_inversion_demo.py
python examples/egregore_defense_demo.py
python examples/egregore_defense_three_frame_demo.py
python examples/hfctm_compliance_demo.py
python examples/multi_agent_coordinator_demo.py
```

## Examples Overview

### 1. Chiral Inversion Demo (`chiral_inversion_demo.py`)

Demonstrates **time-reversal validation** using the ChiralInversionController.

**Key Concepts**:
- Validates that operations satisfy chiral symmetry: `TP O (TP)⁻¹ = O†`
- Computes fidelity between forward and time-reversed computations
- Identifies divergence points where symmetry is violated

**Use Cases**:
- Verifying reversibility of quantum operations
- Detecting computational asymmetries
- Ensuring causality preservation in inference chains

**Reference**: Section 6.2 of MIH-IIE specification

---

### 2. Egregore Defense System Demo (`egregore_defense_demo.py`)

Demonstrates **semantic drift protection** using the EgregoreDefenseSystem.

**Key Concepts**:
- Measures semantic torsion (drift in symbol-meaning mappings)
- Detects corrupted reasoning patterns:
  - Circular reasoning
  - Authority appeals without evidence
  - Manufactured consensus
  - Linguistic drift
  - Measurement corruption
- Autonomous correction protocols

**Use Cases**:
- Protecting against adversarial manipulation
- Detecting institutional capture
- Maintaining semantic baseline integrity
- Preventing gradual redefinition of core concepts

**Reference**: Section 6.4 of MIH-IIE specification

---

### 3. Three-Frame Egregore Defense Demo (`egregore_defense_three_frame_demo.py`)

Runs three evaluators (forward, retro, atemporal) and emits convergence/corruption/shift scores in one pass.

**Key Concepts**:
- Uses semantic torsion thresholds to flag drift
- Surfaces corrupted pattern similarity and quarantine decisions
- Outputs JSON summary for quick inspection

**Use Cases**:
- Fast L5 governance smoke test
- Demonstrating multi-frame agreement vs. corruption
- Validating monitoring thresholds before deployment

---

### 4. HFCTM-II Compliance Demo (`hfctm_compliance_demo.py`)

Demonstrates verification of the **four HFCTM-II principles**:

1. **Holographic Projection**: Information on boundaries encodes bulk dynamics
2. **Fractal Self-Similarity**: `DH ≈ e ≈ 2.718` (Hausdorff dimension)
3. **Chiral Symmetry**: `TP A₀ = A₀` (time-reversal + parity invariance)
4. **Toroidal Topology**: `||ρ(T) - ρ(0)|| < ε` (recursive closure)

**Key Concepts**:
- Chiral symmetry verification
- Fractal dimension measurement via box-counting
- Toroidal closure validation
- Integrated compliance checking

**Use Cases**:
- Ensuring computational operations align with HFCTM-II theory
- Detecting violations of fundamental principles
- Monitoring system-level consistency

**Reference**: Section 6.1 of MIH-IIE specification

---

### 5. Multi-Agent Coordinator Demo (`multi_agent_coordinator_demo.py`)

Demonstrates **polychronic inference coordination** across four temporal modes:

- **Forward Causal** (τL): Standard linear time, conventional causality
- **Retrocausal** (τR): Backward-directed inference, effect-to-cause reasoning
- **Atemporal** (τA): Pattern space, timeless relationships
- **Metacognitive** (τM): Meta-level coordination and synthesis

**Key Concepts**:
- Synchronization pulse generation with phase coherence: `Σ φₖ = 0 mod 2π`
- Multi-agent inference orchestration
- Temporal mode aggregation
- Polychronic state management

**Use Cases**:
- Parallel inference across different temporal perspectives
- Consensus building from diverse reasoning modes
- Metacognitive synthesis of forward/backward/atemporal analyses

**Reference**: Section 5.3 of MIH-IIE specification

---

## Integration Example

For a complete system integration example, see:

```python
from mih_iie.layers.l5_governance import HFCTMII_SafetyCore, init_safety_core
from mih_iie.layers.l5_governance.chiral_inversion import ChiralInversionController
from mih_iie.layers.l5_governance.egregore_defense import EgregoreDefenseSystem
from mih_iie.layers.l5_governance.hfctm_compliance import HFCTMComplianceMonitor
from mih_iie.layers.l4_ironwood.multi_agent_coordinator import MultiAgentInferenceCoordinator

# Initialize all governance components
chiral = ChiralInversionController()
eds = EgregoreDefenseSystem()
compliance = HFCTMComplianceMonitor()
coordinator = MultiAgentInferenceCoordinator()

# Your inference workflow here...
```

## Requirements

All examples require:
- `numpy` (for numerical operations)
- MIH-IIE package (installed via `pip install -e .` or PYTHONPATH)

No ML/quantum dependencies required for these basic examples.

## Next Steps

After running these examples:

1. Review the [MIH-IIE specification](../spec/MIH-IIE_v1.0.pdf) for theoretical foundations
2. Read [CLAUDE.md](../CLAUDE.md) for development guidance
3. Explore [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) for legacy code integration
4. Check [tests/](../tests/) for comprehensive test suites

## Phase Roadmap

These examples demonstrate **Phase 0** (Theoretical Validation) components:

- ✅ L5 Governance: Chiral inversion, egregore defense, HFCTM compliance
- ✅ L4 Ironwood: Multi-agent coordination (classical simulation)

**Phase 1** (Years 1-3) will add:
- Majorana qubit array integration (L2)
- Quantum-classical interface (L3)
- Physical holographic projector (L4)
- Hardware-accelerated tensor processing

See [README.md](../README.md) for complete roadmap.
