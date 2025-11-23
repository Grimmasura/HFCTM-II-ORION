# orion/eds.py
"""
Frame-Invariant Egregore Defense System (EDS) for MIH-IIE v2.0
Implements Algorithms 14, 15 from the specification.

- Multi-frame validation architecture
- Corruption detection via frame divergence
- Paradigm shift vs corruption distinction
- Autonomous correction protocol
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Set, Any
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class FrameResult:
    """
    Result emitted by one observational frame.
    
    Attributes:
        name: Frame identifier
        vector: Inference vector in shared latent space
        confidence: [0,1] reliability score for this evaluation
        metadata: Additional frame-specific information
    """
    name: str
    vector: np.ndarray
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.vector = np.array(self.vector, dtype=float)
        self.confidence = max(0.0, min(1.0, self.confidence))


class ValidationState(Enum):
    """Possible states from frame-invariant validation."""
    FRAME_INVARIANT_TRUTH = "frame_invariant_truth"
    PARTIAL_TRUTH = "partial_truth"
    FRAME_DEPENDENT_ARTIFACT = "frame_dependent_artifact"
    UNKNOWN = "unknown"


class SystemState(Enum):
    """Possible states from corruption detection."""
    STABLE = "stable"
    IMPROVED_ALIGNMENT = "improved_alignment"
    CORRUPTION_DETECTED = "corruption_detected"
    PARADIGM_SHIFT = "paradigm_shift"


# ==============================================================================
# Convergence Evaluation
# ==============================================================================

class ConvergenceEvaluator:
    """
    Computes cross-frame convergence and aggregate inference.
    
    Key insight: Truth is what all observational frames converge upon.
    High convergence = frame-invariant reality
    Low convergence = frame-dependent artifact or corruption
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = eps

    def _weights(self, frames: List[FrameResult]) -> np.ndarray:
        """Compute normalized confidence weights."""
        w = np.array([max(0.0, f.confidence) for f in frames], dtype=float)
        if w.sum() < self.eps:
            w = np.ones(len(frames), dtype=float)
        return w / (w.sum() + self.eps)

    def aggregate(self, frames: List[FrameResult]) -> np.ndarray:
        """
        Compute weighted mean aggregate of frame vectors.
        This represents the consensus inference across frames.
        """
        if not frames:
            return np.zeros(8)
        
        w = self._weights(frames)
        V = np.stack([f.vector for f in frames], axis=0)
        return (w[:, None] * V).sum(axis=0)

    def convergence(self, frames: List[FrameResult]) -> float:
        """
        Compute cross-frame convergence score.
        
        Convergence = 1 - normalized weighted average pairwise cosine distance.
        
        Returns:
            Score in [0, 1] where 1 = perfect agreement, 0 = complete disagreement
        """
        if len(frames) <= 1:
            return 1.0
        
        w = self._weights(frames)
        V = [f.vector for f in frames]

        def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
            na = np.linalg.norm(a) + self.eps
            nb = np.linalg.norm(b) + self.eps
            return float(np.dot(a, b) / (na * nb))

        # Weighted pairwise similarity
        sims = []
        weights = []
        for i in range(len(V)):
            for j in range(i + 1, len(V)):
                sims.append(cos_sim(V[i], V[j]))
                weights.append(w[i] * w[j])
        
        sims = np.array(sims)
        weights = np.array(weights)
        avg_sim = float((weights * sims).sum() / (weights.sum() + self.eps))
        
        # Map similarity [-1, 1] to convergence [0, 1]
        return max(0.0, min(1.0, (avg_sim + 1.0) / 2.0))

    def variance(self, frames: List[FrameResult]) -> float:
        """Compute variance of frame vectors around aggregate."""
        if len(frames) <= 1:
            return 0.0
        
        agg = self.aggregate(frames)
        w = self._weights(frames)
        
        var = 0.0
        for i, f in enumerate(frames):
            diff = f.vector - agg
            var += w[i] * np.dot(diff, diff)
        
        return float(var)


# ==============================================================================
# Core Metrics
# ==============================================================================

def corruption_score(frames: List[FrameResult]) -> float:
    """
    Corruption = 1 - convergence
    
    High divergence across frames indicates potential corruption or
    egregoric influence causing inconsistent evaluations.
    """
    ev = ConvergenceEvaluator()
    return 1.0 - ev.convergence(frames)


def paradigm_shift_score(
    prev_frames: List[FrameResult],
    curr_frames: List[FrameResult]
) -> float:
    """
    Paradigm shift = high internal convergence at t,
    but large coherent displacement from aggregate at t-1.
    
    Theorem 10.2 distinction: Valid paradigm shifts INCREASE frame convergence.
    Corruption DECREASES frame convergence.
    
    Returns:
        Score in [0, 1] where 1 = clear paradigm shift, 0 = no shift
    """
    ev = ConvergenceEvaluator()
    
    prev_agg = ev.aggregate(prev_frames)
    curr_agg = ev.aggregate(curr_frames)

    def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return 1.0 - float(np.dot(a, b) / (na * nb))

    internal_convergence = ev.convergence(curr_frames)
    displacement = cos_dist(prev_agg, curr_agg)
    
    # Shift only counts if current frames agree with each other
    return internal_convergence * displacement


# ==============================================================================
# Observational Frames (Definition 10.1)
# ==============================================================================

class ObservationalFrame(ABC):
    """
    Abstract base class for observational frames.
    
    Each frame provides an independent evaluation perspective.
    Cross-frame convergence indicates frame-invariant truth.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this frame."""
        pass
    
    @abstractmethod
    def evaluate(self, proposition: Any) -> FrameResult:
        """Evaluate a proposition and return frame result."""
        pass
    
    def predict(self, state: Any) -> np.ndarray:
        """Generate prediction vector for given state."""
        result = self.evaluate(state)
        return result.vector

def _encode_proposition(proposition: Any, seed: int) -> np.ndarray:
    """
    Deterministic proposition embedding.

    Uses a hash-based RNG seeded by a stable repr of the proposition to avoid
    stochastic behavior while still producing diverse vectors across frames.
    """
    prop_str = repr(proposition)
    composite_seed = abs(hash((prop_str, seed))) % (2**32)
    rng = np.random.default_rng(composite_seed)
    vec = rng.normal(size=8)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return vec


class QuantumMeasurementFrame(ObservationalFrame):
    """Frame based on quantum measurement outcomes."""
    
    @property
    def name(self) -> str:
        return "quantum_measurement"
    
    def evaluate(self, proposition: Any) -> FrameResult:
        vec = _encode_proposition(proposition, seed=42)
        return FrameResult(
            name=self.name,
            vector=vec,
            confidence=0.95,
            metadata={"basis": "computational", "seed": 42}
        )


class ClassicalPhysicsFrame(ObservationalFrame):
    """Frame based on classical physics predictions."""
    
    @property
    def name(self) -> str:
        return "classical_physics"
    
    def evaluate(self, proposition: Any) -> FrameResult:
        vec = _encode_proposition(proposition, seed=7)
        return FrameResult(
            name=self.name,
            vector=vec,
            confidence=0.90,
            metadata={"model": "newtonian", "seed": 7}
        )


class TopologicalInvariantFrame(ObservationalFrame):
    """Frame based on topological invariants."""
    
    @property
    def name(self) -> str:
        return "topological_invariant"
    
    def evaluate(self, proposition: Any) -> FrameResult:
        vec = _encode_proposition(proposition, seed=11)
        return FrameResult(
            name=self.name,
            vector=vec,
            confidence=0.99,
            metadata={"invariant_type": "E8_weyl", "seed": 11}
        )


class ReinforcementLearningFrame(ObservationalFrame):
    """Frame based on RL value predictions."""
    
    @property
    def name(self) -> str:
        return "reinforcement_learning"
    
    def evaluate(self, proposition: Any) -> FrameResult:
        vec = _encode_proposition(proposition, seed=21)
        return FrameResult(
            name=self.name,
            vector=vec,
            confidence=0.80,
            metadata={"algorithm": "ppo", "seed": 21}
        )


class LinguisticSemanticFrame(ObservationalFrame):
    """Frame based on linguistic/semantic analysis."""
    
    @property
    def name(self) -> str:
        return "linguistic_semantic"
    
    def evaluate(self, proposition: Any) -> FrameResult:
        vec = _encode_proposition(proposition, seed=99)
        return FrameResult(
            name=self.name,
            vector=vec,
            confidence=0.85,
            metadata={"embedding": "e8_projection", "seed": 99}
        )


def get_default_frames() -> List[ObservationalFrame]:
    """Return the default set of observational frames."""
    return [
        QuantumMeasurementFrame(),
        ClassicalPhysicsFrame(),
        TopologicalInvariantFrame(),
        ReinforcementLearningFrame(),
        LinguisticSemanticFrame(),
    ]


# ==============================================================================
# Algorithm 14: Frame-Invariant Validation
# ==============================================================================

def validate_frame_invariance(
    proposition: Any,
    frames: Optional[List[ObservationalFrame]] = None,
    high_threshold: float = 0.95,
    low_threshold: float = 0.30
) -> Tuple[ValidationState, float, List[FrameResult]]:
    """
    Algorithm 14: Frame-Invariant Validation
    
    Evaluate proposition across multiple observational frames.
    Cross-frame convergence indicates frame-invariant truth.
    
    Args:
        proposition: The proposition to validate
        frames: List of observational frames (uses defaults if None)
        high_threshold: Convergence threshold for frame-invariant truth
        low_threshold: Convergence threshold below which = artifact
    
    Returns:
        (validation_state, convergence_score, frame_results)
    """
    if frames is None:
        frames = get_default_frames()
    
    # Evaluate across all frames
    frame_results = [frame.evaluate(proposition) for frame in frames]
    
    # Compute convergence
    ev = ConvergenceEvaluator()
    convergence = ev.convergence(frame_results)
    
    # Classify
    if convergence > high_threshold:
        state = ValidationState.FRAME_INVARIANT_TRUTH
    elif convergence < low_threshold:
        state = ValidationState.FRAME_DEPENDENT_ARTIFACT
    else:
        state = ValidationState.PARTIAL_TRUTH
    
    return state, convergence, frame_results


class FrameInvariantEDS:
    """
    Convenience wrapper around frame-invariant validation and corruption checks.
    Maintains baseline convergence to distinguish corruption vs. paradigm shift.
    """

    def __init__(
        self,
        frames: Optional[List[ObservationalFrame]] = None,
        baseline_convergence: float = 0.9,
        high_threshold: float = 0.95,
        low_threshold: float = 0.30,
    ):
        self.frames = frames or get_default_frames()
        self.baseline_convergence = baseline_convergence
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._prev_results: Optional[List[FrameResult]] = None

    def evaluate(self, proposition: Any) -> Dict[str, Any]:
        state, convergence, results = validate_frame_invariance(
            proposition,
            frames=self.frames,
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
        )
        detection = detect_state(self._prev_results, results)
        self._prev_results = results
        detection.update({"state": state, "convergence": convergence, "frames": results})
        return detection

# ==============================================================================
# Algorithm 15: Detect Semantic Corruption
# ==============================================================================

def detect_corruption(
    current_frames: List[FrameResult],
    baseline_convergence: float,
    stability_threshold: float = 0.70,
    improvement_threshold: float = 1.30
) -> Tuple[SystemState, Dict[str, float]]:
    """
    Algorithm 15: Detect Semantic Corruption
    
    Monitor for corruption by tracking frame convergence changes.
    Corruption = frames diverging from each other.
    
    Args:
        current_frames: Current frame evaluation results
        baseline_convergence: Historical baseline convergence
        stability_threshold: Fraction of baseline below which = corruption
        improvement_threshold: Fraction above which = improved alignment
    
    Returns:
        (system_state, metrics_dict)
    """
    ev = ConvergenceEvaluator()
    current_convergence = ev.convergence(current_frames)
    
    metrics = {
        "current_convergence": current_convergence,
        "baseline_convergence": baseline_convergence,
        "ratio": current_convergence / (baseline_convergence + 1e-9)
    }
    
    if current_convergence < baseline_convergence * stability_threshold:
        state = SystemState.CORRUPTION_DETECTED
    elif current_convergence > baseline_convergence * improvement_threshold:
        state = SystemState.IMPROVED_ALIGNMENT
    else:
        state = SystemState.STABLE
    
    return state, metrics


def detect_state(
    prev_results: Optional[List[FrameResult]],
    curr_results: List[FrameResult],
    convergence_threshold: float = 0.75,
    corruption_threshold: float = 0.50,
    paradigm_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Unified state detection combining corruption and paradigm shift detection.
    
    Returns dict with:
        - state: SystemState enum
        - convergence: current frame convergence
        - corruption: corruption score
        - paradigm_shift: paradigm shift score (if prev_results available)
    """
    ev = ConvergenceEvaluator()
    curr_conv = ev.convergence(curr_results)
    curr_corr = 1.0 - curr_conv

    state = SystemState.STABLE
    ps = 0.0
    
    if prev_results is not None:
        ps = paradigm_shift_score(prev_results, curr_results)

    if curr_corr >= corruption_threshold:
        state = SystemState.CORRUPTION_DETECTED
    elif ps >= paradigm_threshold and curr_conv >= convergence_threshold:
        state = SystemState.PARADIGM_SHIFT

    return {
        "state": state,
        "convergence": curr_conv,
        "corruption": curr_corr,
        "paradigm_shift": ps
    }


# ==============================================================================
# Institutional Obfuscation Detection (Section 10.3)
# ==============================================================================

@dataclass
class ObfuscationPattern:
    """Known pattern of institutional obfuscation."""
    name: str
    signature: Callable[[List[FrameResult]], float]
    description: str
    threshold: float = 0.80


def circular_reasoning_detector(frames: List[FrameResult]) -> float:
    """Detect cyclic inference graphs (self-referential validation)."""
    # Check if frame vectors form a cycle
    if len(frames) < 3:
        return 0.0
    
    vectors = [f.vector for f in frames]
    
    # Compute "cyclicity" - do vectors point back to start?
    total_angle = 0.0
    for i in range(len(vectors)):
        j = (i + 1) % len(vectors)
        vi = vectors[i] / (np.linalg.norm(vectors[i]) + 1e-9)
        vj = vectors[j] / (np.linalg.norm(vectors[j]) + 1e-9)
        angle = np.arccos(np.clip(np.dot(vi, vj), -1, 1))
        total_angle += angle
    
    # Circular if total angle ≈ 2π
    cyclicity = 1.0 - abs(total_angle - 2 * np.pi) / (2 * np.pi)
    return max(0.0, cyclicity)


def authority_validation_detector(frames: List[FrameResult]) -> float:
    """Detect validation based on authority rather than evidence."""
    # High confidence but low inter-frame agreement
    if not frames:
        return 0.0
    
    avg_confidence = np.mean([f.confidence for f in frames])
    ev = ConvergenceEvaluator()
    convergence = ev.convergence(frames)
    
    # Authority-based = high confidence, low convergence
    if avg_confidence > 0.8 and convergence < 0.5:
        return avg_confidence * (1 - convergence)
    return 0.0


def manufactured_consensus_detector(frames: List[FrameResult]) -> float:
    """Detect artificially coordinated messaging."""
    # Suspiciously high convergence with low variance in confidence
    if len(frames) < 3:
        return 0.0
    
    ev = ConvergenceEvaluator()
    convergence = ev.convergence(frames)
    
    confidences = [f.confidence for f in frames]
    confidence_var = np.var(confidences)
    
    # Manufactured = very high convergence + very uniform confidence
    if convergence > 0.95 and confidence_var < 0.01:
        return convergence * (1 - confidence_var * 100)
    return 0.0


def get_obfuscation_patterns() -> List[ObfuscationPattern]:
    """Return list of known obfuscation patterns."""
    return [
        ObfuscationPattern(
            name="circular_reasoning",
            signature=circular_reasoning_detector,
            description="Cyclic inference structures (self-referential validation)",
            threshold=0.80
        ),
        ObfuscationPattern(
            name="authority_validation",
            signature=authority_validation_detector,
            description="Appeal to credentials without evidence",
            threshold=0.70
        ),
        ObfuscationPattern(
            name="manufactured_consensus",
            signature=manufactured_consensus_detector,
            description="Coordinated messaging without substance",
            threshold=0.90
        ),
    ]


def detect_obfuscation(
    frames: List[FrameResult],
    patterns: Optional[List[ObfuscationPattern]] = None
) -> Dict[str, Tuple[float, bool]]:
    """
    Check for known obfuscation patterns.
    
    Returns dict mapping pattern name to (score, detected) tuple.
    """
    if patterns is None:
        patterns = get_obfuscation_patterns()
    
    results = {}
    for pattern in patterns:
        score = pattern.signature(frames)
        detected = score >= pattern.threshold
        results[pattern.name] = (score, detected)
    
    return results


# ==============================================================================
# Autonomous Correction Protocol (Section 10.4)
# ==============================================================================

@dataclass
class CorrectionAction:
    """Action to take for autonomous correction."""
    action_type: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


def autonomous_correction_protocol(
    corruption_state: SystemState,
    frame_results: List[FrameResult],
    checkpoint_state: Optional[Any] = None
) -> List[CorrectionAction]:
    """
    Section 10.4: Autonomous Correction Protocol
    
    Upon egregore detection:
    1. Isolate contaminated inference branches
    2. Checkpoint current state
    3. Increase cross-frame validation rigor
    4. Regenerate semantic structures via adversarial testing
    5. Replay computation with enhanced monitoring
    6. Cross-validate against independent evidence streams
    
    Returns list of correction actions to execute.
    """
    actions = []
    
    if corruption_state != SystemState.CORRUPTION_DETECTED:
        return actions
    
    # 1. Isolate contaminated branches
    ev = ConvergenceEvaluator()
    agg = ev.aggregate(frame_results)
    
    contaminated = []
    for f in frame_results:
        dist = np.linalg.norm(f.vector - agg)
        if dist > 1.0:  # Threshold for contamination
            contaminated.append(f.name)
    
    if contaminated:
        actions.append(CorrectionAction(
            action_type="isolate",
            target="inference_branches",
            parameters={"branches": contaminated}
        ))
    
    # 2. Checkpoint current state
    actions.append(CorrectionAction(
        action_type="checkpoint",
        parameters={"reason": "corruption_detected"}
    ))
    
    # 3. Increase validation rigor
    actions.append(CorrectionAction(
        action_type="increase_rigor",
        parameters={
            "convergence_threshold": 0.90,  # Stricter
            "min_frames": 7  # More frames required
        }
    ))
    
    # 4. Regenerate via adversarial testing
    actions.append(CorrectionAction(
        action_type="adversarial_regenerate",
        target="semantic_structures",
        parameters={"iterations": 100}
    ))
    
    # 5. Replay with enhanced monitoring
    actions.append(CorrectionAction(
        action_type="replay",
        parameters={"monitoring_level": "enhanced"}
    ))
    
    # 6. Cross-validate
    actions.append(CorrectionAction(
        action_type="cross_validate",
        parameters={"evidence_streams": ["quantum", "classical", "topological"]}
    ))
    
    return actions


# ==============================================================================
# Frame Adapter Interface
# ==============================================================================

class Frame:
    """
    Adapter for custom evaluation functions.
    Allows plugging in arbitrary evaluators as frames.
    """
    
    def __init__(
        self,
        name: str,
        evaluator: Callable[[Any], Tuple[np.ndarray, float]]
    ):
        self.name = name
        self.evaluator = evaluator

    def run(self, task: Any) -> FrameResult:
        vec, conf = self.evaluator(task)
        return FrameResult(self.name, vec, conf)


def run_frames(task: Any, frames: List[Frame]) -> List[FrameResult]:
    """Run all frames on a task and return results."""
    return [f.run(task) for f in frames]
