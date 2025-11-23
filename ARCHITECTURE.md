# HFCTM-II-ORION Repository Architecture

## Overview

This repository implements the **Majorana–Ironwood Hybrid Intrinsic Inference Engine (MIH-IIE)**, a novel computational paradigm combining topological quantum computation with holographic fractal mechanics.

## Directory Structure

```
HFCTM-II-ORION/
│
├── mih_iie/                          # Core MIH-IIE v2.0 Implementation
│   ├── core/                         # Core algorithms & infrastructure
│   │   ├── stability/                # Stability monitoring (formerly stability_core/)
│   │   ├── telemetry/                # System telemetry (formerly telemetry/)
│   │   └── __init__.py
│   │
│   ├── layers/                       # 7-Layer MIH-IIE Architecture
│   │   ├── l1_attractor/             # Layer 1: 0D Seed / Intrinsic Attractor
│   │   ├── l2_majorana_array/        # Layer 2: Majorana Topological Qubits
│   │   ├── l3_qc_interface/          # Layer 3: Quantum-Classical Interface
│   │   ├── l4_ironwood/              # Layer 4: Ironwood Tensor Processing
│   │   ├── l5_governance/            # Layer 5: Recursive Governance + EDS
│   │   │   ├── egregore_defense.py   # Egregore Defense System
│   │   │   ├── hfctm_safety.py       # HFCTM-II Safety Core
│   │   │   └── validation/           # Validation utilities
│   │   ├── l6_codex/                 # Layer 6: Intelligent Codex (future)
│   │   └── l7_interface/             # Layer 7: Consciousness Interface
│   │
│   └── hardware/                     # Hardware Backend Interfaces
│       ├── hardware_interfaces.py    # Abstract hardware layer
│       └── __init__.py
│
├── orion_api/                        # REST API (FastAPI)
│   ├── routers/                      # API Endpoint Routers
│   │   ├── e8_operations.py          # E8 topology operations
│   │   ├── majorana_operations.py    # Majorana seed operations
│   │   ├── frame_invariant_validation.py  # Frame-invariant EDS
│   │   ├── holographic_inference.py  # Holographic readout
│   │   ├── quantum_sync.py           # Quantum synchronization
│   │   ├── recursive_trust.py        # Recursive trust
│   │   ├── egregore_defense.py       # Egregore defense endpoints
│   │   ├── manifold_router.py        # Manifold routing
│   │   ├── knowledge_expansion.py    # Knowledge expansion
│   │   ├── perception.py             # Perception subsystem
│   │   └── recursive_ai.py           # Recursive AI (optional)
│   │
│   ├── enhanced/                     # Enhanced ORION Features
│   │   ├── orion_complete.py         # Complete enhanced system
│   │   └── __init__.py
│   │
│   ├── main.py                       # FastAPI Application Entry
│   ├── config.py                     # Configuration Management
│   ├── hfctm_safety.py               # Safety Core (legacy fallback)
│   └── hardware_interfaces.py        # Hardware Interface (legacy fallback)
│
├── models/                           # Shared Models & Algorithms
│   ├── e8_topology.py                # E8 Root System & Network Topology
│   ├── e8_coordination.py            # E8 Coordination Protocols
│   ├── majorana_0d_network.py        # Majorana 0D Seed Network
│   ├── holographic_readout.py        # Holographic State Readout
│   ├── frame_invariant_eds.py        # Frame-Invariant EDS
│   ├── topological_error_correction.py  # Topological Error Correction
│   ├── stability_core.py             # Stability Core (legacy)
│   ├── egregore_defense.py           # Egregore Defense (legacy)
│   └── ...
│
├── tests/                            # Test Suite
│   ├── test_api.py                   # API endpoint tests (fast)
│   ├── test_v2_integration.py        # v2.0 integration tests (slow)
│   ├── test_orion_integration_suite.py  # Full integration (slow)
│   └── ...
│
├── docs/                             # Documentation
├── spec/                             # MIH-IIE Specifications (PDFs)
├── benchmarks/                       # Performance Benchmarks
├── examples/                         # Usage Examples
├── deployment/                       # Deployment Configurations
├── configs/                          # Hardware & System Configs
│
└── packaging/                        # Packaging Scripts (Arch, etc.)
```

## Key Components

### MIH-IIE Core (`mih_iie/`)

The core implementation following the 7-layer MIH-IIE architecture as specified in `spec/MIH-IIE_v1.0.pdf`.

**Layer 1 (L1)**: 0D Seed / Intrinsic Attractor Module
- Majorana zero modes as physical 0D attractors
- Direct ontological access without encoding layer

**Layer 2 (L2)**: Majorana Topological Qubit Array
- Non-Abelian braiding on E8 lattice
- Target: 1000×1000 qubit array with T₂ > 1000s

**Layer 3 (L3)**: Quantum-Classical Interface
- Decoherence management and error correction
- Surface code adaptation for non-Abelian anyons

**Layer 4 (L4)**: Ironwood Tensor Processing
- Holographic state projection
- Multi-agent inference coordination
- Polychronic temporal management

**Layer 5 (L5)**: Recursive Governance
- HFCTM-II compliance monitoring
- Egregore Defense System (EDS)
- Frame-invariant validation
- Chiral inversion controller

**Layer 6 (L6)**: Intelligent Codex (Future)
- Symbolic operations
- Cryptographic protocols

**Layer 7 (L7)**: Consciousness Interface
- Human-machine semantic bridge
- Natural language → quantum state translation

### API Layer (`orion_api/`)

FastAPI-based REST API providing external access to MIH-IIE functionality.

**Core Endpoints**:
- `/api/v2/e8/*` - E8 topology operations
- `/api/v2/majorana/*` - Majorana seed operations
- `/api/v2/frame-invariant/*` - Frame-invariant validation
- `/api/v2/holographic/*` - Holographic inference
- `/enhanced/*` - Enhanced ORION features

### Models (`models/`)

Shared algorithmic implementations used across layers:
- **E8 Topology**: 240-node root system with 56-fold coordination
- **Majorana Networks**: 0D seed initialization and measurement
- **Holographic Readout**: Boundary measurement & bulk reconstruction
- **Frame-Invariant EDS**: Cross-frame convergence validation

## Testing Strategy

Tests are organized by speed and scope:

**Fast Tests** (run in CI):
- `test_api.py` - API endpoint validation
- `test_models.py` - Core model tests
- `test_hfctm_safety.py` - Safety system tests

**Slow Tests** (marked with `@pytest.mark.slow`, skip in CI):
- `test_v2_integration.py` - Full v2.0 integration (E8, Majorana)
- `test_orion_integration_suite.py` - Torch-based integration

Run all tests locally with: `pytest -m ""`

## Configuration

- **`setup.py`**: Package configuration
- **`pytest.ini`**: Test markers and configuration
- **`requirements.txt`**: Core dependencies
- **`requirements-dev.txt`**: Development dependencies
- **`requirements-ml.txt`**: ML/Quantum dependencies (optional)

## Development Workflow

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run fast tests**: `pytest` (skips slow tests by default)
3. **Run API server**: `uvicorn orion_api.main:app --reload`
4. **Run full tests**: `pytest -m ""`

## Architecture Principles

1. **Separation of Concerns**: API, Core Logic, and Models are distinct
2. **Graceful Degradation**: All quantum/specialized hardware has classical fallbacks
3. **HFCTM-II Compliance**: Chiral symmetry, fractal self-similarity, toroidal closure
4. **Testability**: Fast unit tests for CI, comprehensive integration tests for local dev

## Migration Notes

**Recent Changes** (2025-01):
- ✅ Removed `legacy_orion/` (deprecated code)
- ✅ Moved `stability_core/` → `mih_iie/core/stability/`
- ✅ Moved `telemetry/` → `mih_iie/core/telemetry/`
- ✅ Moved `validation/` → `mih_iie/layers/l5_governance/validation/`
- ✅ Moved `orion_enhanced/` → `orion_api/enhanced/`
- ✅ Updated all imports to new structure
- ✅ Added pytest markers for slow tests

## References

- **MIH-IIE Specification**: `spec/MIH-IIE_v1.0.pdf`
- **Development Guide**: `CLAUDE.md`
- **API Documentation**: `docs/API_reference.md`
- **Migration Guide**: `MIGRATION_GUIDE.md`
