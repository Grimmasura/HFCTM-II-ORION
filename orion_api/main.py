from fastapi import FastAPI, Response
from orion_enhanced.orion_complete import create_complete_orion_app
from orion_api.routers import (
    quantum_sync,
    recursive_trust,
    egregore_defense,
    manifold_router,
    knowledge_expansion,
    perception,
)
try:  # pragma: no cover - optional router
    from orion_api.routers import recursive_ai
    RECURSIVE_ROUTER_AVAILABLE = True
except Exception:  # pragma: no cover - missing deps
    RECURSIVE_ROUTER_AVAILABLE = False

# MIH-IIE v2.0 routers
try:
    from orion_api.routers import (
        e8_operations,
        majorana_operations,
        frame_invariant_validation,
        holographic_inference
    )
    V2_ROUTERS_AVAILABLE = True
except Exception:
    V2_ROUTERS_AVAILABLE = False

# MIH-IIE imports (new structure)
try:
    from mih_iie.core.stability_core import stability_core
    from mih_iie.layers.l5_governance import init_safety_core, safety_core, SafetyConfig
    MIH_IIE_AVAILABLE = True
except ImportError:
    # Fallback to legacy imports
    from models.stability_core import stability_core
    from orion_api.hfctm_safety import init_safety_core, safety_core, SafetyConfig
    MIH_IIE_AVAILABLE = False

from orion_api.config import settings
try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - import error handling
    TORCH_AVAILABLE = False

    class _TorchStub:
        """Minimal stub used when PyTorch is unavailable."""

        def __getattr__(self, name):  # pragma: no cover - defensive
            raise RuntimeError("PyTorch is not installed")

    torch = _TorchStub()  # type: ignore
from pathlib import Path
import subprocess
try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover - import error handling
    PROMETHEUS_AVAILABLE = False

    CONTENT_TYPE_LATEST = "text/plain"

    def generate_latest() -> bytes:  # type: ignore
        """Fallback when prometheus_client is unavailable."""
        return b""

app_title = "MIH-IIE API" if MIH_IIE_AVAILABLE else "O.R.I.O.N. ∞ API (Legacy)"
app = FastAPI(
    title=app_title,
    description="Majorana–Ironwood Hybrid Intrinsic Inference Engine" if MIH_IIE_AVAILABLE else "Omniversal Recursive Intelligence for Ontological Navigation"
)
app.mount("/enhanced", create_complete_orion_app())

# Initialize safety core on startup
@app.on_event("startup")
async def startup_event():
    config = SafetyConfig()
    init_safety_core(config)

# Safety middleware
@app.middleware("http")
async def safety_middleware(request, call_next):
    response = await call_next(request)
    response.headers["X-HFCTM-Active"] = "true"
    if safety_core:
        response.headers["X-Interventions"] = str(safety_core.intervention_count)
    return response

# Include routers from orion_api
if RECURSIVE_ROUTER_AVAILABLE:  # pragma: no cover - optional router
    app.include_router(
        recursive_ai.router,
        prefix="/api/v1/recursive_ai",
        tags=["Recursive AI"],
    )
app.include_router(quantum_sync.router, prefix="/quantum-sync", tags=["Quantum Sync"])
app.include_router(recursive_trust.router, prefix="/trust", tags=["Recursive Trust"])
app.include_router(egregore_defense.router, prefix="/egregore", tags=["Egregore Defense"])
app.include_router(manifold_router.router, prefix="/manifold", tags=["Manifold Routing"])
app.include_router(knowledge_expansion.router, prefix="/api/v1/knowledge", tags=["Knowledge Expansion"])
app.include_router(perception.router, prefix="/api/v1/perception", tags=["Perception"])

# Include v2.0 routers if available
if V2_ROUTERS_AVAILABLE:
    app.include_router(e8_operations.router)
    app.include_router(majorana_operations.router)
    app.include_router(frame_invariant_validation.router)
    app.include_router(holographic_inference.router)

@app.get("/")
async def root():
    version = "v2.0" if V2_ROUTERS_AVAILABLE else ("v0.1.0-alpha" if MIH_IIE_AVAILABLE else "Legacy")

    response = {
        "message": f"Welcome to {'MIH-IIE' if MIH_IIE_AVAILABLE else 'O.R.I.O.N. ∞'} API",
        "architecture": f"MIH-IIE {version}",
        "host": settings.host,
        "port": settings.port,
        "docs": "/docs",
    }

    if V2_ROUTERS_AVAILABLE:
        response["v2_features"] = {
            "e8_topology": "/api/v2/e8",
            "majorana_0d_seeds": "/api/v2/majorana",
            "frame_invariant_validation": "/api/v2/frame-invariant",
            "holographic_inference": "/api/v2/holographic"
        }

    return response


@app.get("/architecture")
async def architecture_info():
    """Get information about the current architecture."""
    if MIH_IIE_AVAILABLE:
        return {
            "name": "Majorana–Ironwood Hybrid Intrinsic Inference Engine (MIH-IIE)",
            "version": "0.1.0-alpha",
            "spec_version": "1.0",
            "phase": "Phase 0 (Theoretical Validation) → Early Phase 1 (Component Prototyping)",
            "layers": {
                "l1": "0D Seed / Intrinsic Attractor Module (Phase 1 target)",
                "l2": "Majorana Topological Qubit Array (Phase 1 target)",
                "l3": "Quantum-Classical Interface (Phase 1 target)",
                "l4": "Ironwood Tensor Processing (Partial - Multi-agent coordinator)",
                "l5": "Recursive Governance (Implemented - Safety core, compliance, egregore defense)",
                "l6": "Intelligent Codex (Phase 2 target)",
                "l7": "Consciousness Interface (Basic - FastAPI endpoints)"
            },
            "hfctm_principles": [
                "Holographic Projection",
                "Fractal Self-Similarity (DH ≈ e ≈ 2.718)",
                "Chiral Symmetry (TP A₀ = A₀)",
                "Toroidal Topology"
            ],
            "documentation": {
                "spec": "spec/MIH-IIE_v1.0.pdf",
                "dev_guide": "CLAUDE.md",
                "migration": "MIGRATION_GUIDE.md"
            }
        }
    else:
        return {
            "name": "O.R.I.O.N. ∞ (Legacy)",
            "message": "Using legacy imports. MIH-IIE structure not available.",
            "recommendation": "Install mih_iie package or check import paths"
        }


@app.get("/health")
async def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/telemetry")
async def telemetry() -> dict:
    """Expose a snapshot of the StabilityCore telemetry."""
    return {"telemetry": stability_core.snapshot()}


@app.get("/version")
async def version() -> dict:
    repo_dir = Path(__file__).resolve().parent.parent
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir)
            .decode()
            .strip()
        )
    except Exception:
        commit_hash = "unknown"
    return {"version": commit_hash}


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics if available."""
    if not PROMETHEUS_AVAILABLE:
        return Response("Prometheus metrics unavailable", media_type="text/plain")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Add safety endpoint
@app.get("/api/safety/status")
async def safety_status():
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not installed"}
    if not safety_core:
        return {"error": "Safety core not initialized"}

    # Mock safety check
    mock_state = torch.randn(10, 10)
    result = await safety_core.safety_check(mock_state)

    return {
        "safety_active": True,
        "interventions_total": safety_core.intervention_count,
        "last_check": result,
    }
