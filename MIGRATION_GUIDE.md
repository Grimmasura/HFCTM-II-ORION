# MIH-IIE Restructuring Migration Guide

This guide explains the repository restructuring from the original ORION implementation to the formal MIH-IIE seven-layer architecture.

## Overview

**Date**: 2025-11-22
**Trigger**: Alignment with formal specification in `spec/MIH-IIE_v1.0.pdf`
**Approach**: Restructure existing code while preserving functionality

## Directory Structure Changes

### Before (Legacy ORION)
```
HFCTM-II-ORION/
├── orion_api/
│   ├── main.py
│   ├── hfctm_safety.py
│   ├── hardware_interfaces.py
│   └── routers/
├── orion_enhanced/
│   └── orion_complete.py
├── models/
│   └── stability_core.py
└── The_Majorana_Ironwood_Hybrid_Intrinsic_Inference_Engine.pdf
```

### After (MIH-IIE Architecture)
```
HFCTM-II-ORION/
├── spec/
│   └── MIH-IIE_v1.0.pdf
├── mih_iie/
│   ├── layers/
│   │   ├── l1_attractor/
│   │   ├── l2_majorana_array/
│   │   ├── l3_qc_interface/
│   │   ├── l4_ironwood/
│   │   │   └── multi_agent_coordinator.py
│   │   ├── l5_governance/
│   │   │   ├── hfctm_safety.py
│   │   │   ├── hfctm_compliance.py
│   │   │   ├── chiral_inversion.py
│   │   │   ├── egregore_defense.py
│   │   │   └── polychronic_sync.py
│   │   ├── l6_codex/
│   │   └── l7_interface/
│   ├── hardware/
│   │   └── hardware_interfaces.py
│   └── core/
│       └── stability_core.py
├── legacy_orion/
│   ├── orion_api/
│   ├── orion_enhanced/
│   └── README_LEGACY.md
└── CLAUDE.md (updated with MIH-IIE architecture)
```

## Migration Map

### Layer 5 (L5): Recursive Governance

| Legacy Location | New Location | Status |
|----------------|--------------|--------|
| `orion_api/hfctm_safety.py` | `mih_iie/layers/l5_governance/hfctm_safety.py` | ✅ Migrated |
| `orion_enhanced/orion_complete.py` | `mih_iie/layers/l5_governance/polychronic_sync.py` | ✅ Migrated |
| N/A (new) | `mih_iie/layers/l5_governance/chiral_inversion.py` | ✅ Created |
| N/A (new) | `mih_iie/layers/l5_governance/egregore_defense.py` | ✅ Created |
| N/A (new) | `mih_iie/layers/l5_governance/hfctm_compliance.py` | ✅ Created |

**New Components Created**:
- **ChiralInversionController**: Time-reversal validation (Section 6.2 of spec)
- **EgregoreDefenseSystem**: Semantic drift protection (Section 6.4 of spec)
- **HFCTMComplianceMonitor**: Verifies HFCTM-II principles (Section 6.1 of spec)

### Layer 4 (L4): Ironwood Tensor Processing

| Legacy Location | New Location | Status |
|----------------|--------------|--------|
| `orion_enhanced/orion_complete.py` (partial) | `mih_iie/layers/l4_ironwood/multi_agent_coordinator.py` | ✅ Created |
| N/A (Phase 1 target) | `mih_iie/layers/l4_ironwood/holographic_projector.py` | ⏳ Pending |
| N/A (Phase 1 target) | `mih_iie/layers/l4_ironwood/manifold_expansion.py` | ⏳ Pending |

**New Components Created**:
- **MultiAgentInferenceCoordinator**: Orchestrates parallel inference across 4 temporal modes

### Hardware Abstraction

| Legacy Location | New Location | Status |
|----------------|--------------|--------|
| `orion_api/hardware_interfaces.py` | `mih_iie/hardware/hardware_interfaces.py` | ✅ Migrated |

### Core Modules

| Legacy Location | New Location | Status |
|----------------|--------------|--------|
| `models/stability_core.py` | `mih_iie/core/stability_core.py` | ✅ Migrated |

### Documentation

| Legacy Location | New Location | Status |
|----------------|--------------|--------|
| `The_Majorana_Ironwood_Hybrid_Intrinsic_Inference_Engine.pdf` | `spec/MIH-IIE_v1.0.pdf` | ✅ Moved |
| `CLAUDE.md` | `CLAUDE.md` (updated) | ✅ Updated |
| N/A | `MIGRATION_GUIDE.md` (this file) | ✅ Created |
| N/A | `legacy_orion/README_LEGACY.md` | ✅ Created |

## Code Changes Required

### Import Updates

**Old imports**:
```python
from orion_api.hfctm_safety import HFCTMII_SafetyCore, init_safety_core
from orion_enhanced.orion_complete import PolychronicTemporalManager
from models.stability_core import stability_core
```

**New imports**:
```python
from mih_iie.layers.l5_governance import HFCTMII_SafetyCore, init_safety_core
from mih_iie.layers.l5_governance.polychronic_sync import PolychronicTemporalManager
from mih_iie.core.stability_core import stability_core
```

### Using New Components

#### Chiral Inversion
```python
from mih_iie.layers.l5_governance.chiral_inversion import ChiralInversionController

controller = ChiralInversionController(fidelity_threshold=0.95)

# Validate computation
result = controller.validate_chiral_symmetry(
    computation=my_function,
    initial_state=state,
    duration=100
)

if result.is_valid:
    print(f"Chiral symmetry preserved (fidelity: {result.fidelity:.4f})")
else:
    print(f"Violation detected at points: {result.divergence_points}")
```

#### Egregore Defense
```python
from mih_iie.layers.l5_governance.egregore_defense import EgregoreDefenseSystem

eds = EgregoreDefenseSystem()

# Safety check
result = eds.safety_check(
    semantic_field={"concept1": "meaning1", "concept2": "meaning2"},
    inference_structure={"type": "reasoning", "evidence_required": True}
)

if result["should_quarantine"]:
    print("ALERT: Corrupted pattern detected!")
    corrected = eds.autonomous_correction(result)
```

#### HFCTM Compliance
```python
from mih_iie.layers.l5_governance.hfctm_compliance import HFCTMComplianceMonitor

monitor = HFCTMComplianceMonitor()

# Check compliance
result = monitor.check_compliance(
    operation=my_operation,
    state=current_state,
    trajectory=state_trajectory,
    initial_state=initial,
    final_state=final
)

if not result.overall_compliant:
    print(f"Violations: {result.violations}")
```

#### Multi-Agent Coordinator
```python
from mih_iie.layers.l4_ironwood.multi_agent_coordinator import (
    MultiAgentInferenceCoordinator,
    TemporalMode
)

coordinator = MultiAgentInferenceCoordinator(
    num_forward_causal=4,
    num_retrocausal=2,
    num_atemporal=2,
    num_metacognitive=1
)

# Coordinate inference
result = coordinator.coordinate_inference(
    query="What is the answer?",
    context={"domain": "physics"},
    inference_function=my_inference_fn
)

print(f"Aggregated result: {result['aggregated_result']}")
```

## Testing Migration

### Running Tests
Tests should continue to work with legacy imports until fully migrated:

```bash
# Run all tests
pytest

# Run tests for new L5 governance
pytest tests/test_l5_governance.py

# Run tests for new L4 Ironwood
pytest tests/test_l4_ironwood.py
```

### Creating Tests for New Components

```python
# tests/test_l5_governance.py
from mih_iie.layers.l5_governance.chiral_inversion import ChiralInversionController
from mih_iie.layers.l5_governance.egregore_defense import EgregoreDefenseSystem
from mih_iie.layers.l5_governance.hfctm_compliance import HFCTMComplianceMonitor

def test_chiral_inversion():
    controller = ChiralInversionController()
    # ... test logic
```

## Gradual Migration Strategy

For production systems, migrate gradually:

### Phase 1: Dual Imports (Current)
```python
# Support both legacy and new imports
try:
    from mih_iie.layers.l5_governance import HFCTMII_SafetyCore
except ImportError:
    from orion_api.hfctm_safety import HFCTMII_SafetyCore
```

### Phase 2: Deprecation Warnings
```python
import warnings

from orion_api.hfctm_safety import HFCTMII_SafetyCore

warnings.warn(
    "orion_api.hfctm_safety is deprecated. "
    "Use mih_iie.layers.l5_governance instead.",
    DeprecationWarning
)
```

### Phase 3: Full Migration
Remove legacy code entirely, use only `mih_iie.*` imports.

## Benefits of Restructuring

1. **Clear Layer Separation**: Each layer has distinct responsibility per MIH-IIE spec
2. **Modular Development**: Layers can be developed/tested independently
3. **Alignment with Spec**: Direct mapping to `spec/MIH-IIE_v1.0.pdf`
4. **Future-Proof**: Structure supports Phase 1-4 roadmap
5. **Better Documentation**: Each layer documents its spec reference
6. **Testability**: Easier to write layer-specific tests

## Next Steps

### Immediate (Phase 0 completion)
- [ ] Update `orion_api/main.py` to use new imports
- [ ] Create tests for new L5 governance components
- [ ] Update README.md with new structure

### Short-term (Phase 1 preparation)
- [ ] Implement `l4_ironwood/holographic_projector.py`
- [ ] Implement `l4_ironwood/manifold_expansion.py`
- [ ] Design L2 Majorana array interface
- [ ] Design L3 quantum-classical interface

### Medium-term (Phase 1-2)
- [ ] Integrate L7 interface with existing FastAPI routers
- [ ] Implement L6 codex layer
- [ ] Connect all seven layers with bidirectional channels
- [ ] Verify toroidal closure: T₁→₂ ∘ T₂→₃ ∘ ... ∘ T₇→₁ = I

## Questions?

See:
- `CLAUDE.md` - Development guide with MIH-IIE architecture
- `spec/MIH-IIE_v1.0.pdf` - Complete specification
- `legacy_orion/README_LEGACY.md` - Legacy code reference
