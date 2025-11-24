# Changelog

## [2.0.0] - Unreleased
- Add E8 invariant CI workflow, badge, deterministic notebook (`docs/notebooks/e8_verification.ipynb`), and tests.
- Introduce braid compiler and export APIs; add three-frame EDS demo and examples docs.
- Import v2.1 spec modules (E8, coordination, EDS, error correction, holography) with deterministic frames, improved holography contraction, bounded stabilizers, and validation tests.
- Add MCP-style FastAPI server exposing E8 invariants, braids, EDS frames, stabilizers, and holography helpers.
- Add Hugging Face pipeline stubs for E8 verification and EDS evaluation.
- Refine FastAPI lifespan handling; fix enhanced ORION inference serialization; stabilize async tests.
- Use timezone-aware UTC for telemetry and clean Pydantic configs; full test suite passing (105).

[Unreleased]: https://github.com/Grimmasura/HFCTM-II-ORION/compare/main...feature/mih-iie-v2.0-implementation
