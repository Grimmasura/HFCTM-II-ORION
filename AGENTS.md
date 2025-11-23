# Repository Guidelines

This guide orients contributors to the MIH-IIE codebase and keeps changes aligned with the HFCTM-II architecture and safety expectations.

## Project Structure & Module Organization
- `mih_iie/layers/`: L1–L7 implementation modules (attractor, Majorana array, QC interface, Ironwood, governance, codex, interface); keep new logic inside the appropriate layer.
- `mih_iie/core/`: Stability and telemetry primitives used across layers.
- `orion_api/`: FastAPI service, routers, and deployment assets (`orion_api/main.py`, `orion_api/routers/`).
- `examples/`: Runnable demos; use them as reference implementations.
- `docs/`, `MIGRATION_GUIDE.md`, `ARCHITECTURE.md`: Architectural intent and migration notes—update when behavior changes.
- `tests/`: Integration and regression coverage; mirror the package layout (`tests/test_l5_governance.py`, etc.).

## Build, Test, and Development Commands
- Install: `pip install -r requirements.txt` (base), `pip install -r requirements-dev.txt` (pytest/httpx), `pip install -r requirements-ml.txt` (optional quantum/ML).
- Set paths before running examples or tests: `export PYTHONPATH=$PWD:$PYTHONPATH`.
- Run API locally: `uvicorn orion_api.main:app --reload --host 0.0.0.0 --port 8080`.
- Execute demos: `python examples/chiral_inversion_demo.py` (and other scripts under `examples/`).
- Tests: `pytest` for full suite; `pytest --cov=mih_iie` for coverage; target failing tests before adding new ones.

## Coding Style & Naming Conventions
- Python 3, PEP 8, 4-space indents; prefer type hints and explicit returns.
- `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants and env keys (`ORION_*`, `HFCTM_*`).
- Keep imports layered (L1→L7) and avoid circular references; prefer dependency injection over globals.
- Add docstrings for new public classes/functions; keep comments minimal and purposeful.

## Testing Guidelines
- Place new tests under `tests/` matching the module path; name files `test_*.py` and functions `test_*`.
- When adding behavior, pair a positive path test with at least one failure/edge case; use `pytest -k <pattern>` for focused runs.
- For API changes, exercise routers with `httpx.AsyncClient` in integration-style tests.

## Commit & Pull Request Guidelines
- Follow the existing log style: short, imperative subject lines (e.g., `Add MIH-IIE v2.1 reference implementations`); include the scope of change when possible.
- PRs should summarize affected layers/modules, list commands run (`pytest`, coverage, demos), and call out config or contract changes.
- Link related issues and include sample outputs or payloads for API-affecting changes; update docs when public behavior shifts.

## Security & Configuration Tips
- Do not commit secrets; prefer `.env` for local overrides (`ORION_HOST`, `ORION_PORT`, `ORION_MODEL_DIR`, Azure Quantum credentials).
- Keep defaults minimal; document any new env vars in `README.md` or `docs/`.
