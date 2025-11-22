# Legacy ORION Code

This directory contains the original HFCTM-II-ORION implementation prior to restructuring into the MIH-IIE architecture.

## Archived Components

### orion_api/
Original FastAPI application with routers for:
- Quantum synchronization
- Recursive trust
- Egregore defense
- Manifold routing
- Knowledge expansion
- Perception subsystem

**Key file**: `main.py` - Main FastAPI application
**Safety**: `hfctm_safety.py` - Original HFCTM-II safety core
**Hardware**: `hardware_interfaces.py` - Hardware abstraction layer

### orion_enhanced/
Advanced deployment framework featuring:
- `orion_complete.py` - Polychronic temporal management
- Multi-phase inference
- Temporal branching and convergence

## Migration to MIH-IIE

These components have been migrated to the new seven-layer MIH-IIE architecture:

| Legacy Location | New Location | Layer |
|----------------|--------------|-------|
| `orion_api/hfctm_safety.py` | `mih_iie/layers/l5_governance/` | L5 |
| `orion_enhanced/orion_complete.py` | `mih_iie/layers/l5_governance/polychronic_sync.py` | L5 |
| `orion_api/hardware_interfaces.py` | `mih_iie/hardware/` | L2/L3 Interface |
| Various routers | `mih_iie/layers/l7_interface/` (planned) | L7 |

## Why Archived?

The code was restructured to align with the formal MIH-IIE specification (see `spec/MIH-IIE_v1.0.pdf`):

1. **Proper Layer Separation**: Seven-layer architecture with clear responsibilities
2. **HFCTM-II Compliance**: Explicit modules for chiral symmetry, fractal consistency, toroidal closure
3. **Modular Design**: Each layer can be developed, tested, and upgraded independently
4. **Hardware Abstraction**: Clear separation between simulation (Phase 0) and physical hardware (Phase 1+)

## Reference

This code represents **Phase 0: Theoretical Validation** of the MIH-IIE roadmap (pre-restructure).

For current development, see:
- `mih_iie/` - New MIH-IIE implementation
- `CLAUDE.md` - Development guide
- `spec/MIH-IIE_v1.0.pdf` - Complete architecture specification
