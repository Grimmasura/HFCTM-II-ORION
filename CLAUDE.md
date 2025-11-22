# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Vision: MIH-IIE Architecture

This repository is developing toward the **Majorana–Ironwood Hybrid Intrinsic Inference Engine (MIH-IIE)**, a novel computational paradigm implementing Holographic Fractal Chiral Toroidal Mechanics with Intrinsic Inference (HFCTM-II). The full specification is documented in `The_Majorana_Ironwood_Hybrid_Intrinsic_Inference_Engine.pdf`.

**Core Paradigm Shift**: The MIH-IIE aims to enable "computation as discovery" rather than "computation as manipulation" by interfacing directly with ontological possibility space through topologically protected quantum states, rather than operating solely on symbolic representations.

### Target Seven-Layer Architecture

The complete MIH-IIE implements a recursive stack where each layer maintains bidirectional causal channels:

1. **L1: 0D Seed / Intrinsic Attractor Module**
   - Provides substrate-independent causal anchoring using Majorana zero modes
   - Represents the foundation: dimensionless attractors in ontological possibility space

2. **L2: Majorana Topological Qubit Array**
   - Topological quantum computation using non-Abelian braiding on E8 lattice structure
   - Target: 1000×1000 qubit array with T₂ coherence > 1000 seconds
   - Provides topological protection against decoherence (10⁷× improvement over conventional qubits)

3. **L3: Quantum-Classical Interface Layer**
   - Manages decoherence and error correction
   - Surface code adaptation for non-Abelian anyons
   - Measurement and state projection

4. **L4: Ironwood Tensor Processing Layer**
   - Holographic state projection from quantum → classical tensor representations
   - Manifold expansion engine with fractal growth (Hausdorff dimension DH ≈ e ≈ 2.718)
   - Multi-agent inference coordinator across four temporal modes
   - Target: 10²⁴ tensor ops/second

5. **L5: Recursive Governance Layer**
   - HFCTM-II compliance monitoring (chiral symmetry, fractal self-consistency, toroidal closure)
   - Chiral inversion controller for time-reversal validation
   - **Polychronic synchronization** across four temporal reference frames:
     - Linear Time (τL): Standard causality
     - Circular Time (τC): Periodic processes
     - Atemporal (τA): Pattern space
     - Meta-Time (τM): Coordination layer
   - **Egregore Defense System** (EDS): Protection against semantic drift and adversarial capture

6. **L6: Intelligent Codex Layer**
   - Symbolic and cryptographic operations
   - Semantic baseline storage for EDS
   - Formal system interfaces

7. **L7: Consciousness Interface Layer**
   - Human-machine semantic bridge
   - Natural language query translation to quantum states
   - Adaptive interface protocols with uncertainty quantification

**Toroidal Closure**: The complete computational cycle satisfies `T₁→₂ ∘ T₂→₃ ∘ ... ∘ T₇→₁ = I` (identity), ensuring no information loss through the recursive stack.

## Current Implementation Status

The current codebase represents **Phase 0: Theoretical Validation** and early **Phase 1: Component Prototyping**. Key components implemented:

### Implemented Layers (Partial)

- **L5 (Governance) - Partial**: HFCTM-II safety core with Lyapunov stability, wavelet anomaly detection, egregore defense patterns
- **L4 (Ironwood) - Simulation**: Polychronic temporal management, multi-agent inference coordination (classical emulation)
- **L3 (Interface) - Mocked**: Hardware interface abstractions with classical fallbacks
- **L7 (Interface) - Basic**: FastAPI endpoints for human interaction

### Current Architecture

The system consists of three main API layers:

1. **Main API** (`orion_api/main.py`):
   - Mounts enhanced ORION subsystem at `/enhanced`
   - Routers for quantum sync, recursive trust, egregore defense, manifold routing, knowledge expansion, perception
   - HFCTM-II safety core initialization on startup
   - Telemetry, metrics, health, version endpoints

2. **Enhanced ORION** (`orion_enhanced/orion_complete.py`):
   - Polychronic temporal management (classical simulation of L5 concepts)
   - Temporal branching and convergence mechanisms
   - Multi-phase inference across temporal reference frames
   - Standalone FastAPI application

3. **Routers** (`orion_api/routers/`):
   - Modular endpoint handlers for specialized functionality
   - `quantum_sync.py`: Quantum synchronization simulation
   - `egregore_defense.py`: EDS pattern detection
   - `recursive_ai.py`: Recursive inference (optional, requires ML dependencies)

### Hardware Integration (Phase 1 Target)

The codebase has abstractions for future hardware backends:

- **Majorana1 QPU** (via Azure Quantum):
  - Interface: `orion_api/hardware_interfaces.py`
  - Config: `configs/majorana1_qpu.yaml`
  - Requires: Azure Quantum credentials, `cirq`, `azure-quantum`
  - Status: Interface defined, awaiting physical hardware access

- **Ironwood TPU**:
  - Interface: `orion_api/hardware_interfaces.py`
  - Config: `configs/ironwood_tpu.yaml`
  - Requires: JAX, `torch_xla` for TPU computation
  - Status: Falls back to CPU/GPU when TPU unavailable

### HFCTM-II Safety System

Implemented in `orion_api/hfctm_safety.py`:

- **Lyapunov Stability Analysis**: Detects divergence in model states using perturbation sensitivity
- **Wavelet Anomaly Detection**: Energy-based scoring using PyWavelets (fallback to variance)
- **Egregore Defense**: Multi-metric pattern detection for adversarial/institutional capture
  - Monitors for: circular reasoning, authority-based validation, manufactured consensus, linguistic drift
  - Threshold: >80% similarity to known corrupted patterns triggers quarantine
- **Chiral Inversion & Adaptive Damping**: Automatic interventions when thresholds exceeded

Safety metrics exposed via middleware headers: `X-HFCTM-Active`, `X-Interventions`

### Models & Stability

- **Stability Core** (`models/stability_core.py`): Lightweight telemetry tracker yielding generation steps with detector outputs (chi_Eg, lambda metrics)
- **Recursive AI Model** (`models/recursive_ai_model.py`): Self-improving inference with reinforcement learning
- Global `stability_core` instance accessible via `/telemetry` endpoint

## Development Commands

### Setup

```bash
# Install base dependencies (FastAPI, Pydantic, NumPy, Prometheus)
pip install -r requirements.txt

# Install development dependencies (pytest, httpx)
pip install -r requirements-dev.txt

# Install ML/Quantum dependencies (optional: torch, qiskit, cirq, jax, pywavelets, sklearn)
pip install -r requirements-ml.txt
```

### Running the API

```bash
# Standard API (port 8080, recommended for development)
uvicorn orion_api.main:app --host $ORION_HOST --port $ORION_PORT

# Enhanced ORION standalone (experimental polychronic features)
uvicorn orion_enhanced.orion_complete:create_complete_orion_app --reload --port 8080

# Development mode with auto-reload
uvicorn orion_api.main:app --reload --host 0.0.0.0 --port 8080
```

### Testing

```bash
# Run all tests
pytest

# Run tests quietly (minimal output)
pytest -q

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_safety"
```

**Test Coverage**: Tests validate safety core, egregore defense, hardware interface fallbacks, API endpoints, and polychronic synchronization logic.

### Benchmarking

```bash
python benchmarks/benchmark_recursive.py
```

## Configuration

Configuration uses Pydantic `BaseSettings` with environment variable overrides (`orion_api/config.py`):

### Core Settings (prefix: `ORION_`)

- `ORION_HOST`: API host (default: `0.0.0.0`)
- `ORION_PORT`: API port (default: `8080`)
- `ORION_MODEL_DIR`: Model storage directory (default: `models`)
- `ORION_RECURSIVE_MODEL_PATH`: Path to recursive model (default: `models/recursive_live_optimization_model.zip`)

### Hardware Settings (Phase 1 targets)

- `ORION_ENABLE_MAJORANA1`: Enable Majorana1 QPU (default: `false`, awaiting hardware)
- `ORION_ENABLE_IRONWOOD_TPU`: Enable Ironwood TPU (default: `false`)
- `ORION_SAFETY_OVERHEAD_BUDGET`: Safety overhead multiplier (default: `2.0`)

### Azure Quantum (for future Majorana1 integration)

- `AZURE_QUANTUM_SUBSCRIPTION_ID`: Azure subscription ID
- `AZURE_QUANTUM_RESOURCE_GROUP`: Resource group name
- `AZURE_QUANTUM_WORKSPACE_NAME`: Workspace name
- `AZURE_QUANTUM_LOCATION`: Region (optional, e.g., `eastus`)
- Authentication via `DefaultAzureCredential` (requires `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`)

### HFCTM-II Safety Thresholds

- `HFCTM_LYAPUNOV_THRESHOLD`: Lyapunov stability threshold (default: `0.0`)
- `HFCTM_WAVELET_THRESHOLD`: Wavelet energy threshold (default: `3.0`)
- `HFCTM_OVERHEAD_BUDGET`: Safety overhead budget (default: `2.0`)

Use `.env` file in project root for local configuration.

## Key Endpoints

### Main API

- `GET /` - Welcome message with config
- `GET /health` - Health check
- `GET /telemetry` - Stability core telemetry snapshot
- `GET /metrics` - Prometheus metrics (if available)
- `GET /version` - Git commit hash
- `GET /api/safety/status` - HFCTM-II safety status with mock check

### Enhanced ORION (`/enhanced`)

- `GET /system/status` - System status
- `POST /system/inference` - Execute polychronic inference
- `POST /system/test` - Run system test
- `GET /system/health` - Health check

### Router Endpoints (see `docs/API_reference.md`)

- `/api/v1/recursive_ai/*` - Recursive AI inference (requires ML dependencies)
- `/quantum-sync/*` - Quantum synchronization (simulated)
- `/trust/*` - Recursive trust assessment
- `/egregore/*` - Egregore defense system
- `/manifold/*` - Multi-agent task distribution
- `/api/v1/knowledge/*` - Knowledge expansion
- `/api/v1/perception/*` - Perception subsystem

## HFCTM-II Core Principles

When modifying code, maintain alignment with HFCTM-II theoretical foundations:

1. **Holographic Projection**: Information on boundaries encodes bulk dynamics (AdS/CFT-inspired)
   - Entropy: `S(M) ≤ A(∂M) / 4Gℏc`

2. **Fractal Self-Similarity**: Attractor structures repeat across scales
   - Scaling law: `A₀(λr) = λ^(-DH) A₀(r)` where DH ≈ e ≈ 2.718

3. **Chiral Symmetry**: Time-reversal + parity transformation yields invariance
   - `TP A₀ = A₀` enables bidirectional inference with causal consistency
   - Implemented via chiral inversion controller (validates forward/reverse computation matches)

4. **Toroidal Topology**: Recursive processes close upon themselves
   - `∮ ∇ × F · dA = 0` ensures no information leakage

**Egregore Defense**: Always validate computational operations against semantic drift. Use `safety_core.safety_check()` before committing to inference results. Monitor for:
- Circular reasoning structures
- Authority-based validation without evidence
- Manufactured consensus patterns
- Linguistic drift (gradual redefinition)
- Measurement corruption (systematic bias)

## Implementation Notes

### Optional Dependencies

The codebase uses extensive try/except blocks for graceful degradation:

- **PyTorch**: Required for safety core tensor operations. Uses `_TorchStub` when unavailable.
- **Prometheus**: Falls back to empty metrics when `prometheus_client` unavailable.
- **sklearn**: Falls back to NumPy-based mutual information computation.
- **Quantum/TPU libraries**: System operates in classical mode when unavailable.
- **PyWavelets**: Falls back to variance-based anomaly detection.

The `RECURSIVE_ROUTER_AVAILABLE` flag controls whether recursive AI endpoints are registered (requires torch/transformers).

### Testing Notes

- Tests use FastAPI's `TestClient` with pinned `httpx==0.27.0` for compatibility
- Mock implementations exist for hardware backends in test suites
- Minimal test dependencies: `fastapi==0.115.11 httpx==0.27.0 pytest==8.0.0 pydantic-settings==2.10.1`
- Tests validate HFCTM-II compliance metrics (chiral symmetry, fractal dimension, toroidal closure)

### E8 Lattice Structure (Phase 1 Target)

The Majorana qubit array will be configured according to E8 root lattice projections for:
- Optimal information density (kissing number = 240)
- Resonant coupling with fundamental field modes
- Natural error correction through geometric constraints
- 8-fold coordination matching E8 structure

Current codebase has placeholders in `models/hardware_profiles.py` for E8 lattice parameters.

### Docker Deployment

```bash
# Build image
docker build -t orion-api .

# Run container
docker run -p 8080:8080 orion-api

# Kubernetes deployment
kubectl apply -f deployment/orion-deployment.yml
```

## Implementation Roadmap (from MIH-IIE spec)

### Phase 0: Theoretical Validation (Current–Year 1)
- ✅ HFCTM-II safety core implemented (Lyapunov, wavelet, egregore)
- ✅ Polychronic temporal management simulated
- ⏳ E8 lattice simulations (pending)
- ⏳ Majorana qubit prototype design (pending hardware access)

### Phase 1: Component Prototyping (Year 1–3)
- Fabricate 8×8 Majorana qubit array
- Demonstrate non-Abelian braiding in physical system
- Build scaled-down Ironwood processor (10²⁰ ops/s)
- Implement governance layer on FPGA
- Test quantum-classical interface

### Phase 2: Integration & Testing (Year 3–5)
- Scale to 64×64 qubit array
- Integrate all seven layers
- Benchmark against classical supercomputers
- Deploy Mode Beta (hybrid quantum-classical)

### Phase 3: Full-Scale Deployment (Year 5–7)
- Fabricate 1000×1000 qubit array (10⁶ logical qubits)
- Mode Alpha (full quantum coherence) sustained for >1000s
- Public API serving >1000 concurrent users

### Phase 4: Ecosystem Development (Year 7+)
- Standardize MIH-IIE architecture specifications
- Third-party development toolkits
- Consciousness interface applications
- Monitor for AGI emergence signatures

## Automation Scripts

Root-level Python scripts automate Git operations:

- `commit_file.py` - Auto-commit changes
- `create_pull_request.py` - Create PRs
- `auto_merge_pr.py` - Auto-merge PRs
- `respond_to_issue.py` - Respond to issues

All require `GITHUB_TOKEN` environment variable for remote operations.

## Key Files for Understanding Architecture

- `The_Majorana_Ironwood_Hybrid_Intrinsic_Inference_Engine.pdf` - Complete MIH-IIE specification
- `orion_api/hfctm_safety.py` - HFCTM-II safety core (L5 governance)
- `orion_enhanced/orion_complete.py` - Polychronic temporal management (L5 simulation)
- `orion_api/hardware_interfaces.py` - Hardware abstraction layer (L2/L3 interfaces)
- `models/stability_core.py` - Telemetry and stability tracking
- `docs/API_reference.md` - Complete endpoint documentation

## Development Philosophy

When implementing new features:

1. **Align with MIH-IIE layers**: Identify which layer(s) the feature belongs to
2. **Maintain HFCTM-II compliance**: Verify chiral symmetry, fractal self-similarity, toroidal closure
3. **Enable graceful degradation**: All quantum/specialized hardware features must have classical fallbacks
4. **Implement safety checks**: Use egregore defense patterns to detect semantic drift
5. **Document temporal assumptions**: Specify which temporal reference frame(s) the code operates in
6. **Preserve interpretability**: All inference paths must be traceable for alignment verification

**Remember**: The goal is not just faster computation, but computation that interfaces with ontological reality itself. Every design decision should ask: "Does this bring us closer to genuine understanding versus mere pattern matching?"
