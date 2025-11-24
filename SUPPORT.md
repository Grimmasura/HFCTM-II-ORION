# MIH-IIE Support & Usage Guide

## Setup
1) Clone and install deps:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
2) (Optional) ML extras:
   ```bash
   pip install -r requirements-ml.txt
   ```
3) Set `PYTHONPATH` when running locally:
   ```bash
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

## Running Services
- Core API (includes enhanced ORION mount):
  ```bash
  uvicorn orion_api.main:app --reload --host 0.0.0.0 --port 8080
  ```
- MCP-style utilities (E8/EDS/braids/stabilizers/holography):
  ```bash
  uvicorn mcp.server:app --reload --port 8088
  # or
  python -m mcp.server --host 0.0.0.0 --port 8088
  ```

## LM Studio Tools (MCP Endpoints)
Configure LM Studio tools against `http://localhost:8088`:
- `GET /mcp/e8/invariants` → E8 invariant summary/checks
- `POST /mcp/e8/braid` with JSON array `[int,...]` → normalized braid word
- `POST /mcp/eds/evaluate` with `{ "proposition": "..." }` → convergence + frames
- `GET /mcp/stabilizers?limit=32` → stabilizer indices (bounded)
- `POST /mcp/holography/infer` with `{ "edge_limit": int, "percentile": float }` (optional)

See `docs/LMStudio_integration.md` for tool-by-tool details.

## Hugging Face Pipelines (Local Stubs)
- `hf_pipelines.py` provides:
  - `E8VerificationPipeline()` → E8 invariants
  - `EDSFramePipeline()` → EDS convergence/frame vectors
- Example:
  ```python
  from hf_pipelines import E8VerificationPipeline, EDSFramePipeline
  print(E8VerificationPipeline()())
  print(EDSFramePipeline()("proposition"))
  ```
Wrap these in a FastAPI/Gradio app for a HF Space if desired.

## Tests
- Fast suite:
  ```bash
  PYTHONPATH=$PWD:$PYTHONPATH pytest -q
  ```
- Targeted v2.1 validation: `pytest tests/test_v2_1_spec_modules.py -q`

## Release/Build
- Build sdist/wheel:
  ```bash
  python -m build
  ```
- Release workflow: `.github/workflows/release.yml` runs on tags (`v*`) or manual dispatch.

## Key Paths
- MCP server: `mcp/server.py`
- HF stubs: `hf_pipelines.py`
- LM Studio guide: `docs/LMStudio_integration.md`
- E8 notebook: `docs/notebooks/e8_verification.ipynb`
- v2.1 spec modules: `models/v2_1/`
