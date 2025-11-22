"""
Multi-Agent Inference Coordinator (MAIC)

Orchestrates parallel inference across four agent classes operating in distinct
temporal modes as specified in Section 5.3 of the MIH-IIE specification.

Agent Types:
1. Forward Causal (τL): Linear future - Standard inference
2. Retrocausal (τL reversed): Linear past - Constraint satisfaction
3. Atemporal (τA): Pattern space - Invariant detection
4. Metacognitive (τM): Meta-time - Coordination

Synchronization via chiral pulse trains ensures phase coherence across
all temporal reference frames.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque


class TemporalMode(Enum):
    """Temporal reference frames for agent operation."""
    FORWARD_CAUSAL = "forward_causal"     # Linear future (τL)
    RETROCAUSAL = "retrocausal"           # Linear past (τL reversed)
    ATEMPORAL = "atemporal"               # Pattern space (τA)
    METACOGNITIVE = "metacognitive"       # Meta-time (τM)


@dataclass
class InferenceAgent:
    """Agent operating in a specific temporal mode."""
    agent_id: str
    temporal_mode: TemporalMode
    state: Dict[str, Any]
    inference_history: deque
    phase: float  # For synchronization


@dataclass
class SynchronizationPulse:
    """Chiral pulse for multi-agent synchronization."""
    timestamp: float
    amplitudes: List[float]
    frequencies: List[float]
    phases: List[float]


class MultiAgentInferenceCoordinator:
    """
    Coordinates parallel inference across multiple temporal reference frames.

    Synchronization Protocol:
    Sync(t) = Σ_k A_k e^{i(ω_k t + φ_k)} with Σ_k φ_k = 0 mod 2π
    """

    def __init__(
        self,
        num_forward_causal: int = 4,
        num_retrocausal: int = 2,
        num_atemporal: int = 2,
        num_metacognitive: int = 1
    ):
        """
        Initialize Multi-Agent Inference Coordinator.

        Args:
            num_forward_causal: Number of forward causal agents
            num_retrocausal: Number of retrocausal agents
            num_atemporal: Number of atemporal agents
            num_metacognitive: Number of metacognitive agents
        """
        self.agents: Dict[str, InferenceAgent] = {}
        self.sync_pulse_history: List[SynchronizationPulse] = []
        self.inference_results: Dict[str, List[Any]] = {}

        # Initialize agent pools
        self._initialize_agents(
            num_forward_causal,
            num_retrocausal,
            num_atemporal,
            num_metacognitive
        )

    def _initialize_agents(
        self,
        nf: int, nr: int, na: int, nm: int
    ):
        """Initialize agent pools for each temporal mode."""

        # Forward causal agents
        for i in range(nf):
            agent_id = f"fc_{i}"
            self.agents[agent_id] = InferenceAgent(
                agent_id=agent_id,
                temporal_mode=TemporalMode.FORWARD_CAUSAL,
                state={},
                inference_history=deque(maxlen=100),
                phase=0.0
            )

        # Retrocausal agents
        for i in range(nr):
            agent_id = f"rc_{i}"
            self.agents[agent_id] = InferenceAgent(
                agent_id=agent_id,
                temporal_mode=TemporalMode.RETROCAUSAL,
                state={},
                inference_history=deque(maxlen=100),
                phase=0.0
            )

        # Atemporal agents
        for i in range(na):
            agent_id = f"at_{i}"
            self.agents[agent_id] = InferenceAgent(
                agent_id=agent_id,
                temporal_mode=TemporalMode.ATEMPORAL,
                state={},
                inference_history=deque(maxlen=100),
                phase=0.0
            )

        # Metacognitive agents
        for i in range(nm):
            agent_id = f"mc_{i}"
            self.agents[agent_id] = InferenceAgent(
                agent_id=agent_id,
                temporal_mode=TemporalMode.METACOGNITIVE,
                state={},
                inference_history=deque(maxlen=100),
                phase=0.0
            )

    def generate_sync_pulse(self, timestamp: float) -> SynchronizationPulse:
        """
        Generate chiral synchronization pulse.

        Ensures Σ_k φ_k = 0 mod 2π for phase coherence.

        Args:
            timestamp: Current time

        Returns:
            SynchronizationPulse
        """
        num_agents = len(self.agents)

        # Generate random frequencies and amplitudes
        amplitudes = np.random.uniform(0.5, 1.5, num_agents).tolist()
        frequencies = np.random.uniform(0.1, 1.0, num_agents).tolist()

        # Generate phases that sum to 0 mod 2π
        phases = np.random.uniform(0, 2*np.pi, num_agents - 1).tolist()
        # Last phase ensures sum = 0 mod 2π
        phase_sum = sum(phases)
        last_phase = (2*np.pi - (phase_sum % (2*np.pi))) % (2*np.pi)
        phases.append(last_phase)

        pulse = SynchronizationPulse(
            timestamp=timestamp,
            amplitudes=amplitudes,
            frequencies=frequencies,
            phases=phases
        )

        self.sync_pulse_history.append(pulse)
        return pulse

    def synchronize_agents(self, pulse: SynchronizationPulse):
        """
        Apply synchronization pulse to all agents.

        Args:
            pulse: Synchronization pulse to apply
        """
        for idx, (agent_id, agent) in enumerate(self.agents.items()):
            # Update agent phase based on pulse
            agent.phase = pulse.phases[idx % len(pulse.phases)]

    def coordinate_inference(
        self,
        query: str,
        context: Dict[str, Any],
        inference_function: Callable[[str, Dict[str, Any], TemporalMode], Any]
    ) -> Dict[str, Any]:
        """
        Coordinate parallel inference across all temporal modes.

        Args:
            query: Input query
            context: Inference context
            inference_function: Function(query, context, mode) -> result

        Returns:
            Aggregated inference results from all temporal modes
        """
        # Generate synchronization pulse
        pulse = self.generate_sync_pulse(timestamp=len(self.inference_results))
        self.synchronize_agents(pulse)

        # Collect results by temporal mode
        results_by_mode: Dict[TemporalMode, List[Any]] = {
            mode: [] for mode in TemporalMode
        }

        # Execute inference in parallel (simulated as sequential)
        for agent_id, agent in self.agents.items():
            result = inference_function(query, context, agent.temporal_mode)

            # Store in agent history
            agent.inference_history.append(result)

            # Collect by mode
            results_by_mode[agent.temporal_mode].append(result)

        # Metacognitive aggregation
        aggregated = self._metacognitive_aggregation(results_by_mode)

        # Store results
        result_id = f"inference_{len(self.inference_results)}"
        self.inference_results[result_id] = results_by_mode

        return {
            "aggregated_result": aggregated,
            "results_by_mode": results_by_mode,
            "sync_pulse": pulse,
            "num_agents": len(self.agents)
        }

    def _metacognitive_aggregation(
        self,
        results_by_mode: Dict[TemporalMode, List[Any]]
    ) -> Any:
        """
        Aggregate results from different temporal modes using metacognitive synthesis.

        Args:
            results_by_mode: Results organized by temporal mode

        Returns:
            Synthesized result
        """
        # Simple aggregation: collect all results
        all_results = []
        for mode, mode_results in results_by_mode.items():
            all_results.extend(mode_results)

        # Return first non-None result or default message
        for result in all_results:
            if result is not None:
                return result

        return {
            "status": "no_consensus",
            "message": "Agents could not reach consensus",
            "modes_consulted": list(results_by_mode.keys())
        }

    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on agent performance.

        Returns:
            Dictionary with agent statistics
        """
        stats_by_mode = {}

        for mode in TemporalMode:
            mode_agents = [a for a in self.agents.values() if a.temporal_mode == mode]

            if mode_agents:
                avg_history_len = np.mean([len(a.inference_history) for a in mode_agents])

                stats_by_mode[mode.value] = {
                    "count": len(mode_agents),
                    "avg_history_length": float(avg_history_len),
                    "agent_ids": [a.agent_id for a in mode_agents]
                }

        return {
            "total_agents": len(self.agents),
            "stats_by_mode": stats_by_mode,
            "total_inferences": len(self.inference_results),
            "sync_pulses_generated": len(self.sync_pulse_history)
        }
