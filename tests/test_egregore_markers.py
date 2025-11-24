import numpy as np

from mih_iie.layers.l5_governance.egregore_defense import EgregoreDefenseSystem


def test_empirical_markers_qualify_when_signals_high():
    eds = EgregoreDefenseSystem(marker_threshold=0.5)

    beliefs = np.array([
        [1, 1, 0],
        [1, 1, 0.1],
        [0.9, 1, 0],
    ])
    interactions = np.ones((3, 3)) - np.eye(3)
    retention = np.array([10, 9, 9])
    behaviors = np.array([
        [1, 0, 1],
        [0.9, 0.1, 1],
        [1, 0.2, 0.9],
    ])
    semantics = np.array([
        [0.2, 0.8],
        [0.21, 0.79],
        [0.19, 0.81],
    ])

    markers = eds.assess_empirical_egregore(
        belief_matrix=beliefs,
        interaction_matrix=interactions,
        retention_curve=retention,
        outcome_coherence=0.9,
        behavior_matrix=behaviors,
        semantic_embeddings=semantics,
    )

    assert markers.qualifies
    assert markers.distributed_representation > 0.5
    assert markers.feedback_loops > 0.5
    assert markers.semantic_topology > 0.5


def test_empirical_markers_fail_when_signals_low():
    eds = EgregoreDefenseSystem(marker_threshold=0.6)
    beliefs = np.array([[1, 0], [0, 1]])  # low correlation
    markers = eds.assess_empirical_egregore(belief_matrix=beliefs)
    assert not markers.qualifies
