# LM Studio Integration Guide (Local Tools)

This repo ships an MCP-style server exposing MIH-IIE utilities. Configure LM Studio tools to call these local endpoints.

## Start the server

```bash
PYTHONPATH=$PWD:$PYTHONPATH uvicorn mcp.server:app --reload --port 8088
# or
python -m mcp.server --host 0.0.0.0 --port 8088
```

Requirements: `pip install -r requirements.txt -r requirements-dev.txt` (dev covers pytest/httpx; FastAPI/uvicorn are in base).

## Tool endpoints to configure

- `GET http://localhost:8088/mcp/e8/invariants`  
  Returns `{ summary: {...}, checks: {...} }`

- `POST http://localhost:8088/mcp/e8/braid`  
  Body: JSON array of ints, e.g. `[1,2,1]` → `{ normalized_word: [..], string: "s1 s2 s1" }`

- `POST http://localhost:8088/mcp/eds/evaluate`  
  Body: `{ "proposition": "..." }` → convergence + per-frame vectors/confidence

- `GET http://localhost:8088/mcp/stabilizers?limit=32`  
  Returns stabilizer indices (bounded)

- `POST http://localhost:8088/mcp/holography/infer`  
  Body (optional): `{ "edge_limit": 8, "percentile": 2.0 }` → boundary size + bulk shape

## LM Studio tool examples

Define each tool in LM Studio with the corresponding method, URL, and a simple schema:

- Name: `e8_invariants`  
  Method: GET  
  URL: `http://localhost:8088/mcp/e8/invariants`  
  Params: none

- Name: `compile_braid`  
  Method: POST  
  URL: `http://localhost:8088/mcp/e8/braid`  
  JSON: `{ "sequence": [int, ...] }` (LM Studio can pass an array directly)

- Name: `eds_eval`  
  Method: POST  
  URL: `http://localhost:8088/mcp/eds/evaluate`  
  JSON: `{ "proposition": "string" }`

- Name: `stabilizers`  
  Method: GET  
  URL: `http://localhost:8088/mcp/stabilizers`  
  Query: `limit` (int, optional)

- Name: `holography_infer`  
  Method: POST  
  URL: `http://localhost:8088/mcp/holography/infer`  
  JSON: `{ "edge_limit": int, "percentile": float }` (both optional)

## Notes

- All endpoints are JSON and self-contained; no auth by default—keep to localhost or add a proxy with auth if needed.
- Heavy operations are bounded (`limit`, `edge_limit`) to stay fast for interactive use.
- If colocated with the LLM runtime, you can skip HTTP and call Python directly (e.g., `compute_e8_invariants()`, `compile_reflection_sequence([...])`, `EDSFramePipeline()(proposition)`).
