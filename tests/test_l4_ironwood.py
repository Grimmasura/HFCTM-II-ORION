"""
Tests for L4: Ironwood Tensor Processing Layer

Tests multi-agent inference coordination and polychronic synchronization.
"""

import pytest
import numpy as np
from mih_iie.layers.l4_ironwood.multi_agent_coordinator import (
    MultiAgentInferenceCoordinator,
    TemporalMode,
    InferenceAgent,
    SynchronizationPulse
)


class TestMultiAgentInferenceCoordinator:
    """Test multi-agent polychronic inference coordination."""

    def test_initialization(self):
        """Test coordinator initialization."""
        coordinator = MultiAgentInferenceCoordinator(
            num_forward_causal=4,
            num_retrocausal=2,
            num_atemporal=2,
            num_metacognitive=1
        )

        assert len(coordinator.agents) == 4 + 2 + 2 + 1
        assert len(coordinator.sync_pulse_history) == 0
        assert len(coordinator.inference_results) == 0

    def test_agent_initialization_counts(self):
        """Test that correct number of agents are created per mode."""
        coordinator = MultiAgentInferenceCoordinator(
            num_forward_causal=3,
            num_retrocausal=2,
            num_atemporal=1,
            num_metacognitive=1
        )

        # Count agents by mode
        mode_counts = {}
        for agent in coordinator.agents.values():
            mode = agent.temporal_mode
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        assert mode_counts[TemporalMode.FORWARD_CAUSAL] == 3
        assert mode_counts[TemporalMode.RETROCAUSAL] == 2
        assert mode_counts[TemporalMode.ATEMPORAL] == 1
        assert mode_counts[TemporalMode.METACOGNITIVE] == 1

    def test_agent_ids(self):
        """Test that agent IDs are correctly assigned."""
        coordinator = MultiAgentInferenceCoordinator(
            num_forward_causal=2,
            num_retrocausal=1,
            num_atemporal=1,
            num_metacognitive=1
        )

        agent_ids = list(coordinator.agents.keys())

        # Check for expected ID patterns
        assert any(aid.startswith("fc_") for aid in agent_ids)  # forward causal
        assert any(aid.startswith("rc_") for aid in agent_ids)  # retrocausal
        assert any(aid.startswith("at_") for aid in agent_ids)  # atemporal
        assert any(aid.startswith("mc_") for aid in agent_ids)  # metacognitive

    def test_sync_pulse_generation(self):
        """Test synchronization pulse generation."""
        coordinator = MultiAgentInferenceCoordinator()

        pulse = coordinator.generate_sync_pulse(timestamp=1.0)

        assert isinstance(pulse, SynchronizationPulse)
        assert pulse.timestamp == 1.0
        assert len(pulse.amplitudes) == len(coordinator.agents)
        assert len(pulse.frequencies) == len(coordinator.agents)
        assert len(pulse.phases) == len(coordinator.agents)

        # Verify phase coherence: sum of phases should be 0 mod 2π
        phase_sum = sum(pulse.phases)
        assert np.isclose(phase_sum % (2 * np.pi), 0.0, atol=1e-6)

    def test_sync_pulse_phase_coherence(self):
        """Test that sync pulses maintain phase coherence."""
        coordinator = MultiAgentInferenceCoordinator()

        # Generate multiple pulses
        pulses = [coordinator.generate_sync_pulse(timestamp=float(i)) for i in range(5)]

        for pulse in pulses:
            phase_sum = sum(pulse.phases)
            # Sum should be 0 mod 2π (within numerical tolerance)
            assert np.isclose(phase_sum % (2 * np.pi), 0.0, atol=1e-6)

    def test_agent_synchronization(self):
        """Test that agents receive synchronization pulses."""
        coordinator = MultiAgentInferenceCoordinator(num_forward_causal=3)

        pulse = coordinator.generate_sync_pulse(timestamp=1.0)
        coordinator.synchronize_agents(pulse)

        # Check that agents have received phase updates
        for agent in coordinator.agents.values():
            assert agent.phase in pulse.phases

    def test_coordinate_inference_basic(self):
        """Test basic inference coordination."""
        coordinator = MultiAgentInferenceCoordinator(
            num_forward_causal=2,
            num_retrocausal=1,
            num_atemporal=1,
            num_metacognitive=1
        )

        # Simple inference function that returns mode name
        def test_inference(query, context, mode):
            return {
                "query": query,
                "mode": mode.value,
                "result": f"Result from {mode.value}"
            }

        result = coordinator.coordinate_inference(
            query="test query",
            context={"key": "value"},
            inference_function=test_inference
        )

        assert "aggregated_result" in result
        assert "results_by_mode" in result
        assert "sync_pulse" in result
        assert "num_agents" in result

        # Check that we got results from all modes
        results_by_mode = result["results_by_mode"]
        assert TemporalMode.FORWARD_CAUSAL in results_by_mode
        assert TemporalMode.RETROCAUSAL in results_by_mode
        assert TemporalMode.ATEMPORAL in results_by_mode
        assert TemporalMode.METACOGNITIVE in results_by_mode

    def test_coordinate_inference_result_counts(self):
        """Test that correct number of results are collected per mode."""
        coordinator = MultiAgentInferenceCoordinator(
            num_forward_causal=3,
            num_retrocausal=2,
            num_atemporal=1,
            num_metacognitive=1
        )

        def simple_inference(query, context, mode):
            return {"mode": mode.value}

        result = coordinator.coordinate_inference(
            query="test",
            context={},
            inference_function=simple_inference
        )

        results_by_mode = result["results_by_mode"]

        # Check counts match agent counts
        assert len(results_by_mode[TemporalMode.FORWARD_CAUSAL]) == 3
        assert len(results_by_mode[TemporalMode.RETROCAUSAL]) == 2
        assert len(results_by_mode[TemporalMode.ATEMPORAL]) == 1
        assert len(results_by_mode[TemporalMode.METACOGNITIVE]) == 1

    def test_coordinate_inference_aggregation(self):
        """Test metacognitive aggregation of results."""
        coordinator = MultiAgentInferenceCoordinator()

        def inference_with_value(query, context, mode):
            return f"result_{mode.value}"

        result = coordinator.coordinate_inference(
            query="test",
            context={},
            inference_function=inference_with_value
        )

        # Should have aggregated result
        assert result["aggregated_result"] is not None

    def test_coordinate_inference_none_handling(self):
        """Test handling of None results in aggregation."""
        coordinator = MultiAgentInferenceCoordinator(num_forward_causal=2)

        call_count = [0]

        def inference_returns_none(query, context, mode):
            # Return None for first calls, then a value
            call_count[0] += 1
            if call_count[0] <= 1:
                return None
            return "valid_result"

        result = coordinator.coordinate_inference(
            query="test",
            context={},
            inference_function=inference_returns_none
        )

        # Should still have an aggregated result (either valid or no_consensus)
        assert "aggregated_result" in result

    def test_agent_history_tracking(self):
        """Test that agent inference history is tracked."""
        coordinator = MultiAgentInferenceCoordinator(num_forward_causal=1)

        def simple_inference(query, context, mode):
            return {"iteration": query}

        # Run multiple inferences
        for i in range(3):
            coordinator.coordinate_inference(
                query=str(i),
                context={},
                inference_function=simple_inference
            )

        # Check that agents have history
        for agent in coordinator.agents.values():
            assert len(agent.inference_history) == 3

    def test_agent_history_maxlen(self):
        """Test that agent history respects maxlen."""
        coordinator = MultiAgentInferenceCoordinator(num_forward_causal=1)

        def simple_inference(query, context, mode):
            return {"value": query}

        # Run more than maxlen (100) inferences
        for i in range(150):
            coordinator.coordinate_inference(
                query=str(i),
                context={},
                inference_function=simple_inference
            )

        # Check that history is capped at 100
        for agent in coordinator.agents.values():
            assert len(agent.inference_history) == 100

    def test_get_agent_statistics(self):
        """Test agent statistics retrieval."""
        coordinator = MultiAgentInferenceCoordinator(
            num_forward_causal=4,
            num_retrocausal=2,
            num_atemporal=2,
            num_metacognitive=1
        )

        def simple_inference(query, context, mode):
            return "result"

        # Run some inferences
        for _ in range(3):
            coordinator.coordinate_inference(
                query="test",
                context={},
                inference_function=simple_inference
            )

        stats = coordinator.get_agent_statistics()

        assert stats["total_agents"] == 9
        assert "stats_by_mode" in stats
        assert "total_inferences" in stats
        assert "sync_pulses_generated" in stats

        # Check mode-specific stats
        mode_stats = stats["stats_by_mode"]
        assert TemporalMode.FORWARD_CAUSAL.value in mode_stats
        assert TemporalMode.RETROCAUSAL.value in mode_stats

        # Check counts
        fc_stats = mode_stats[TemporalMode.FORWARD_CAUSAL.value]
        assert fc_stats["count"] == 4
        assert "avg_history_length" in fc_stats
        assert fc_stats["avg_history_length"] == 3.0

    def test_multiple_inference_rounds(self):
        """Test running multiple rounds of inference."""
        coordinator = MultiAgentInferenceCoordinator()

        def query_inference(query, context, mode):
            return {"query": query, "mode": mode.value}

        # Run multiple rounds
        results = []
        for i in range(5):
            result = coordinator.coordinate_inference(
                query=f"query_{i}",
                context={},
                inference_function=query_inference
            )
            results.append(result)

        # Check that we have 5 results
        assert len(results) == 5
        assert len(coordinator.sync_pulse_history) == 5
        assert len(coordinator.inference_results) == 5

    def test_temporal_mode_enum(self):
        """Test TemporalMode enum values."""
        assert TemporalMode.FORWARD_CAUSAL.value == "forward_causal"
        assert TemporalMode.RETROCAUSAL.value == "retrocausal"
        assert TemporalMode.ATEMPORAL.value == "atemporal"
        assert TemporalMode.METACOGNITIVE.value == "metacognitive"

    def test_inference_agent_structure(self):
        """Test InferenceAgent dataclass structure."""
        agent = InferenceAgent(
            agent_id="test_agent",
            temporal_mode=TemporalMode.FORWARD_CAUSAL,
            state={"key": "value"},
            inference_history=[],
            phase=0.5
        )

        assert agent.agent_id == "test_agent"
        assert agent.temporal_mode == TemporalMode.FORWARD_CAUSAL
        assert agent.state == {"key": "value"}
        assert isinstance(agent.inference_history, list)
        assert agent.phase == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
