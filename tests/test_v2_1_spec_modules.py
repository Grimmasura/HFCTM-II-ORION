import numpy as np

from models.v2_1 import (
    E8,
    _normalize_int_vec,
    SimulatedBackend,
    initialize_0d_seed_array,
    establish_e8_network,
    FrameResult,
    ConvergenceEvaluator,
    FrameInvariantEDS,
    Stabilizer,
    MinimumWeightDecoder,
    error_correction_cycle,
    CorrectionResult,
    identify_boundary,
    BoundaryMeasurement,
    TensorNetwork,
    build_e8_tensor_network,
    reconstruct_bulk,
    find_independent_operations,
    schedule_braiding_sequence,
    establish_synchronization,
    construct_stabilizers,
    optimize_contraction_order,
    get_default_frames,
)


def test_v21_e8_invariants_and_weyl_action():
    e8 = E8.generate_roots()
    adjacency = e8.adjacency_matrix()

    assert len(e8.roots_scaled) == 240
    assert set(e8.degree_sequence()) == {56}
    assert e8.graph_diameter() == 3

    simple = E8.simple_roots()[0]
    alpha_scaled = _normalize_int_vec(simple)
    reflection = e8.reflection_matrix(alpha_scaled)
    assert e8.verify_weyl_action_on_roots(reflection)


def test_v21_coordination_establishes_entanglement_registry():
    e8 = E8.generate_roots()
    adjacency = e8.adjacency_matrix()
    backend = SimulatedBackend(e8)
    seeds = initialize_0d_seed_array(240, e8, backend)

    registry = establish_e8_network(seeds, adjacency, backend)
    expected_edges = int(240 * 56 / 2)

    assert len(registry.pairs) == expected_edges
    assert len(registry.get_neighbors(0)) == 56


def test_v21_coordination_parallel_and_sync():
    e8 = E8.generate_roots()
    adjacency = e8.adjacency_matrix()
    backend = SimulatedBackend(e8)
    seeds = initialize_0d_seed_array(16, e8, backend)
    registry = establish_e8_network(seeds, adjacency, backend)

    ops = find_independent_operations(
        [type("op", (), {"nodes": (0, 1)}), type("op", (), {"nodes": (2, 3)}), type("op", (), {"nodes": (1, 2)})]
    )
    assert ops and len(ops) >= 1

    batches = schedule_braiding_sequence([(0, 1), (2, 3)], e8)
    assert batches and all(isinstance(batch, list) for batch in batches)

    sync_state = establish_synchronization(registry, e8, backend)
    assert len(sync_state.temporal_map) == 240
    assert len(sync_state.ghz_anchors) == 1
    assert len(sync_state.ghz_anchors[0].anchor_nodes) == 8


def test_v21_frame_invariant_detection_tracks_convergence():
    frames = [
        FrameResult("f1", np.ones(8), confidence=0.95),
        FrameResult("f2", np.ones(8) * 0.9, confidence=0.90),
    ]
    ev = ConvergenceEvaluator()
    convergence = ev.convergence(frames)
    assert convergence > 0.9

    eds = FrameInvariantEDS(frames=[], baseline_convergence=0.9)
    result = eds.evaluate("proposition")
    assert "state" in result and "convergence" in result


def test_v21_frame_invariant_default_frames_deterministic():
    frames = get_default_frames()
    eds = FrameInvariantEDS(frames=frames, baseline_convergence=0.9)
    first = eds.evaluate("same proposition")
    second = eds.evaluate("same proposition")
    assert abs(first["convergence"] - second["convergence"]) < 1e-9


def test_v21_error_correction_cycle_runs_with_minimal_stabilizer():
    e8 = E8.generate_roots()
    backend = SimulatedBackend(e8)
    seeds = initialize_0d_seed_array(8, e8, backend)

    stabilizers = [Stabilizer((0, 1, 2, 3))]
    decoder = MinimumWeightDecoder(e8, stabilizers)
    result = error_correction_cycle(stabilizers, seeds, backend, decoder)

    assert result in {
        CorrectionResult.NO_ERROR,
        CorrectionResult.CORRECTED,
        CorrectionResult.UNCORRECTABLE,
    }


def test_v21_holography_boundary_and_bulk_reconstruction():
    e8 = E8.generate_roots()
    adjacency = e8.adjacency_matrix()
    boundary = identify_boundary(adjacency, method="random", percentile=2.0)
    assert boundary, "Boundary selection should yield nodes"

    # Build a compressed tensor network and generate boundary measurements
    network = build_e8_tensor_network(e8, adjacency, bond_dim=2, physical_dim=2, max_tensor_rank=2)
    measurements = {
        node: BoundaryMeasurement(node=node, z_basis=0, x_basis=0, coordinate=e8.root_vectors()[node])
        for node in boundary[:4]
    }
    assert isinstance(network, TensorNetwork)
    assert measurements

    bulk = reconstruct_bulk(measurements, network, boundary[:4], edge_limit=8)
    assert np.asarray(bulk).size > 0

    order = optimize_contraction_order(network, set(boundary[:4]), max_edges=4)
    assert order and all(len(edge) == 2 for edge in order)


def test_v21_stabilizer_construction_limited():
    e8 = E8.generate_roots()
    stabs = construct_stabilizers(e8, limit=16)
    assert stabs and len(stabs) <= 16
