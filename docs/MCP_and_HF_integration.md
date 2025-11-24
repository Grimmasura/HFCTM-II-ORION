### MCP-Style Server

- Module: `mcp/server.py`
- Start locally: `uvicorn mcp.server:app --reload --port 8088`
- Endpoints:
  - `GET /mcp/e8/invariants` — returns E8 invariant summary + checks.
  - `POST /mcp/e8/braid` — body: JSON array of generators; returns normalized braid word.
  - `POST /mcp/eds/evaluate` — body: `{ "proposition": "..." }`; returns convergence and frame vectors.
  - `GET /mcp/stabilizers?limit=32` — returns bounded stabilizer indices.
  - `POST /mcp/holography/infer` — params: `edge_limit`, `percentile`; returns boundary size and bulk shape.
- Add auth if exposing externally; defaults are for local/CI use.

### Hugging Face Pipelines (Local Stubs)

- Module: `hf_pipelines.py`
  - `E8VerificationPipeline()` → summary/checks for E8 invariants.
  - `EDSFramePipeline()` → convergence + frame vectors for a proposition.
- Usage:
```python
from hf_pipelines import E8VerificationPipeline, EDSFramePipeline

e8 = E8VerificationPipeline()
print(e8())

eds = EDSFramePipeline()
print(eds("some proposition"))
```
- To serve via HF Space, wrap these calls in a small FastAPI/Gradio app and point the pipeline to your endpoint if remote execution is preferred.
