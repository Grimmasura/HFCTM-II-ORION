"""
Multi-Agent Polychronic Inference Demo

Demonstrates coordinated inference across four temporal modes:
- Forward Causal
- Retrocausal
- Atemporal
- Metacognitive

Reference: Section 5.3 of MIH-IIE specification
"""

import numpy as np
from mih_iie.layers.l4_ironwood.multi_agent_coordinator import (
    MultiAgentInferenceCoordinator,
    TemporalMode
)


def main():
    print("=== Multi-Agent Polychronic Inference Demo ===\n")

    # Initialize coordinator with custom agent counts
    coordinator = MultiAgentInferenceCoordinator(
        num_forward_causal=4,
        num_retrocausal=2,
        num_atemporal=2,
        num_metacognitive=1
    )

    print(f"Initialized coordinator with {len(coordinator.agents)} agents:")
    print(f"  Forward Causal: 4")
    print(f"  Retrocausal: 2")
    print(f"  Atemporal: 2")
    print(f"  Metacognitive: 1\n")

    # Define a simple inference function
    def temporal_inference(query, context, mode):
        """
        Example inference function that responds differently based on temporal mode.

        In a real system, this would interface with actual inference engines
        that operate in different temporal reference frames.
        """
        responses = {
            TemporalMode.FORWARD_CAUSAL: f"Forward analysis: {query}",
            TemporalMode.RETROCAUSAL: f"Backward analysis: {query}",
            TemporalMode.ATEMPORAL: f"Pattern analysis: {query}",
            TemporalMode.METACOGNITIVE: f"Meta-analysis: {query}"
        }

        return {
            "query": query,
            "mode": mode.value,
            "response": responses.get(mode, "Unknown mode"),
            "context": context
        }

    # Example 1: Basic coordinated inference
    print("Example 1: Basic coordinated inference")

    result = coordinator.coordinate_inference(
        query="What is the optimal path?",
        context={"domain": "navigation", "constraints": ["time", "energy"]},
        inference_function=temporal_inference
    )

    print(f"  Num agents: {result['num_agents']}")
    print(f"  Aggregated result: {result['aggregated_result']}")
    print(f"  Results by mode:")
    for mode, results in result['results_by_mode'].items():
        print(f"    {mode.value}: {len(results)} result(s)")
    print()

    # Example 2: Synchronization pulse
    print("Example 2: Examining synchronization pulse")

    pulse = result['sync_pulse']
    print(f"  Timestamp: {pulse.timestamp}")
    print(f"  Number of agents: {len(pulse.amplitudes)}")
    print(f"  Amplitudes (first 5): {pulse.amplitudes[:5]}")
    print(f"  Frequencies (first 5): {pulse.frequencies[:5]}")
    print(f"  Phases (first 5): {pulse.phases[:5]}")

    # Verify phase coherence: sum of phases should be 0 mod 2π
    phase_sum = sum(pulse.phases)
    coherence = phase_sum % (2 * np.pi)
    print(f"  Phase coherence (should be ≈0): {coherence:.8f}\n")

    # Example 3: Multiple inference rounds
    print("Example 3: Running multiple inference rounds")

    queries = [
        "Analyze current state",
        "Predict future trajectory",
        "Identify patterns",
        "Synthesize insights"
    ]

    for i, query in enumerate(queries, 1):
        result = coordinator.coordinate_inference(
            query=query,
            context={"iteration": i},
            inference_function=temporal_inference
        )
        print(f"  Round {i}: {len(result['results_by_mode'])} temporal modes engaged")
    print()

    # Example 4: Agent statistics
    print("Example 4: Agent statistics")

    stats = coordinator.get_agent_statistics()
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Total inferences: {stats['total_inferences']}")
    print(f"  Sync pulses generated: {stats['sync_pulses_generated']}")
    print(f"  Stats by mode:")
    for mode_name, mode_stats in stats['stats_by_mode'].items():
        print(f"    {mode_name}:")
        print(f"      Count: {mode_stats['count']}")
        print(f"      Avg history length: {mode_stats['avg_history_length']:.1f}")


if __name__ == "__main__":
    main()
