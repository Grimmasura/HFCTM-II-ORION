import pytest

from mih_iie.layers.l2_majorana_array import compute_e8_invariants, verify_e8_invariants


def test_e8_invariants_core_properties():
    report = compute_e8_invariants()
    checks = verify_e8_invariants(report)

    assert checks["root_count"], "E8 must contain 240 roots"
    assert checks["norms"], f"Unexpected norms: {report.unique_norms}"
    assert checks["inner_products"], f"Inner products off expected set: {report.inner_products}"
    assert checks["degree_regular"], f"Degree irregularity: {report.degree_distribution}"


def test_e8_adjacency_density_reasonable():
    report = compute_e8_invariants()
    # For a 56-regular graph with 240 nodes: edges = n*deg/2
    expected_density = (240 * 56 / 2) / (240 * 239 / 2)
    assert abs(report.adjacency_density - expected_density) < 1e-6


@pytest.mark.parametrize(
    "expected_degree",
    [56],
)
def test_e8_degree_distribution(expected_degree: int):
    report = compute_e8_invariants()
    assert set(report.degrees) == {expected_degree}
    assert report.degree_distribution.get(expected_degree) == 240
