import pytest
pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("pydantic_settings", reason="pydantic-settings not installed")

import numpy as np

from orion_enhanced.orion_complete import create_complete_orion_app


def test_subsystems_smoke():
    """Test quantum stabilizer with proper state dimensions."""
    app = create_complete_orion_app()
    stabilizer = app.state.quantum_stabilizer

    dim = 16
    rho = np.eye(dim, dtype=complex) / dim
    rho_flat = rho.flatten()

    result = stabilizer.solve({"rho": rho_flat})
    assert "result" in result
