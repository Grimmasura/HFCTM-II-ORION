"""
ASGI adapter for HFCTM-II-ORION.
- Tries to build the real FastAPI via create_enhanced_orion_api()
- If it fails (missing deps, import error), exposes a fallback app with diagnostics
"""
import logging

try:
    from orion_enhanced_extensions import create_enhanced_orion_api
    app = create_enhanced_orion_api()
except Exception as e:  # pragma: no cover - fallback path
    logging.getLogger("uvicorn.error").exception("Failed to build ORION app", exc_info=e)
    try:
        from fastapi import FastAPI

        app = FastAPI(title="ORION (fallback)")

        @app.get("/healthz")
        def healthz():
            return {"status": "fallback", "error": str(e)}

        @app.get("/")
        def root():
            return {"ok": True, "mode": "fallback"}
    except Exception:  # pragma: no cover - if even FastAPI is missing
        raise e
