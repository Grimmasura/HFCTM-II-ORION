"""
ASGI adapter for HFCTM-II-ORION.
Exports a top-level `app` for Uvicorn by building it via the enhanced factory.
"""
from orion_enhanced_extensions import create_enhanced_orion_api
app = create_enhanced_orion_api()
