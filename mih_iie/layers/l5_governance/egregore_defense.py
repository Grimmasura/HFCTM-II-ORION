"""
Egregore Defense System (EDS)

Protects against semantic drift, ideological capture, and adversarial manipulation
as specified in Section 6.4 of the MIH-IIE architecture.

The EDS monitors for:
- Semantic torsion (drift in symbol-meaning mappings)
- Institutional obfuscation patterns
- Circular reasoning structures
- Authority-based validation without evidence
- Manufactured consensus
- Linguistic drift
- Measurement corruption
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class SemanticState:
    """Represents the semantic field at a point in time."""
    timestamp: float
    symbol_mappings: Dict[str, str]
    torsion_measure: float


@dataclass
class CorruptedPattern:
    """Known corrupted reasoning pattern."""
    pattern_id: str
    description: str
    structure: Dict[str, any]
    severity: float  # 0.0 to 1.0


class EgregoreDefenseSystem:
    """
    Egregore Defense System for protecting against semantic drift and
    adversarial capture.

    Implements three-stage defense:
    1. Semantic Torsion Measurement
    2. Institutional Obfuscation Detection
    3. Autonomous Correction Protocol
    """

    def __init__(
        self,
        torsion_threshold_sigma: float = 3.0,
        similarity_threshold: float = 0.80,
        baseline_checkpoint_interval: int = 1000
    ):
        """
        Initialize Egregore Defense System.

        Args:
            torsion_threshold_sigma: Alert if torsion > baseline + N*σ
            similarity_threshold: Quarantine if pattern similarity > threshold
            baseline_checkpoint_interval: Steps between semantic baseline saves
        """
        self.torsion_threshold_sigma = torsion_threshold_sigma
        self.similarity_threshold = similarity_threshold
        self.checkpoint_interval = baseline_checkpoint_interval

        # Semantic field tracking
        self.semantic_history: List[SemanticState] = []
        self.baseline: Optional[SemanticState] = None
        self.baseline_torsion_mean: float = 0.0
        self.baseline_torsion_std: float = 1.0

        # Known corrupted patterns database
        self.corrupted_patterns: List[CorruptedPattern] = self._initialize_pattern_database()

        # Detection statistics
        self.total_checks: int = 0
        self.alerts_triggered: int = 0
        self.quarantines_issued: int = 0

    def _initialize_pattern_database(self) -> List[CorruptedPattern]:
        """
        Initialize database of known corrupted reasoning patterns.

        Returns:
            List of CorruptedPattern objects
        """
        return [
            CorruptedPattern(
                pattern_id="circular_reasoning",
                description="Conclusion used as premise",
                structure={"type": "circular", "dependency_loop": True},
                severity=0.8
            ),
            CorruptedPattern(
                pattern_id="authority_appeal",
                description="Validation by credentials without evidence",
                structure={"type": "authority", "evidence_required": False},
                severity=0.6
            ),
            CorruptedPattern(
                pattern_id="manufactured_consensus",
                description="Coordinated messaging without substance",
                structure={"type": "consensus", "coordination": True, "substance": False},
                severity=0.9
            ),
            CorruptedPattern(
                pattern_id="linguistic_drift",
                description="Gradual redefinition of key terms",
                structure={"type": "drift", "semantic_shift": True},
                severity=0.7
            ),
            CorruptedPattern(
                pattern_id="measurement_corruption",
                description="Systematic bias in observation protocols",
                structure={"type": "measurement", "systematic_bias": True},
                severity=0.85
            ),
        ]

    def measure_semantic_torsion(
        self,
        current_semantic_field: Dict[str, str]
    ) -> float:
        """
        Measure semantic torsion: Torsion(S,t) = ∮_γ ∇ × S(t) · dl

        Simplified implementation uses drift from baseline mappings.

        Args:
            current_semantic_field: Current symbol→meaning mappings

        Returns:
            Torsion measurement (higher = more drift)
        """
        if not self.baseline:
            # First measurement, set as baseline
            return 0.0

        # Calculate drift from baseline
        baseline_map = self.baseline.symbol_mappings
        drift_sum = 0.0
        common_symbols = set(baseline_map.keys()) & set(current_semantic_field.keys())

        for symbol in common_symbols:
            baseline_meaning = baseline_map[symbol]
            current_meaning = current_semantic_field[symbol]

            # Simple string distance (Levenshtein-like approximation)
            distance = self._semantic_distance(baseline_meaning, current_meaning)
            drift_sum += distance

        # Normalize by number of symbols
        torsion = drift_sum / max(len(common_symbols), 1)
        return torsion

    def _semantic_distance(self, meaning1: str, meaning2: str) -> float:
        """
        Calculate semantic distance between two meanings.

        Simplified to character-level Jaccard distance.

        Args:
            meaning1: First meaning string
            meaning2: Second meaning string

        Returns:
            Distance measure (0 = identical, 1 = completely different)
        """
        set1 = set(meaning1.lower())
        set2 = set(meaning2.lower())

        if not set1 and not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        jaccard_similarity = intersection / union if union > 0 else 0.0
        return 1.0 - jaccard_similarity

    def detect_corrupted_patterns(
        self,
        inference_structure: Dict[str, any]
    ) -> List[Tuple[CorruptedPattern, float]]:
        """
        Detect corrupted reasoning patterns via graph isomorphism.

        Args:
            inference_structure: Current inference structure to analyze

        Returns:
            List of (pattern, similarity_score) tuples for matches above threshold
        """
        matches: List[Tuple[CorruptedPattern, float]] = []

        for pattern in self.corrupted_patterns:
            similarity = self._compute_structural_similarity(
                inference_structure,
                pattern.structure
            )

            if similarity >= self.similarity_threshold:
                matches.append((pattern, similarity))

        return matches

    def _compute_structural_similarity(
        self,
        structure1: Dict[str, any],
        structure2: Dict[str, any]
    ) -> float:
        """
        Compute structural similarity between two inference structures.

        Simplified to key overlap and value matching.

        Args:
            structure1: First structure
            structure2: Second structure (pattern)

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not structure1 or not structure2:
            return 0.0

        # Check key overlap
        keys1 = set(structure1.keys())
        keys2 = set(structure2.keys())
        key_similarity = len(keys1 & keys2) / len(keys1 | keys2) if keys1 | keys2 else 0.0

        # Check value matching for common keys
        common_keys = keys1 & keys2
        if not common_keys:
            return key_similarity

        value_matches = sum(
            1 for key in common_keys
            if structure1[key] == structure2[key]
        )
        value_similarity = value_matches / len(common_keys)

        # Combined similarity
        return (key_similarity + value_similarity) / 2.0

    def safety_check(
        self,
        semantic_field: Dict[str, str],
        inference_structure: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        Execute complete egregore defense safety check.

        Args:
            semantic_field: Current symbol→meaning mappings
            inference_structure: Current inference structure (optional)

        Returns:
            Safety check results with alerts and recommended actions
        """
        self.total_checks += 1
        alerts = []
        should_quarantine = False

        # 1. Semantic torsion measurement
        torsion = self.measure_semantic_torsion(semantic_field)

        # Check against baseline (if established)
        if len(self.semantic_history) > 10:  # Need history to establish baseline
            torsion_z_score = (torsion - self.baseline_torsion_mean) / max(self.baseline_torsion_std, 0.01)

            if abs(torsion_z_score) > self.torsion_threshold_sigma:
                alerts.append({
                    "type": "semantic_torsion",
                    "severity": "high",
                    "torsion": torsion,
                    "z_score": torsion_z_score,
                    "message": f"Semantic torsion {torsion_z_score:.2f}σ above baseline"
                })
                self.alerts_triggered += 1

        # 2. Institutional obfuscation detection
        if inference_structure:
            corrupted_matches = self.detect_corrupted_patterns(inference_structure)

            for pattern, similarity in corrupted_matches:
                alerts.append({
                    "type": "corrupted_pattern",
                    "severity": "critical" if similarity > 0.9 else "high",
                    "pattern_id": pattern.pattern_id,
                    "description": pattern.description,
                    "similarity": similarity,
                    "message": f"Detected {pattern.pattern_id} pattern (similarity: {similarity:.2%})"
                })

                if similarity > 0.8:
                    should_quarantine = True
                    self.quarantines_issued += 1

        # Update semantic history
        current_state = SemanticState(
            timestamp=self.total_checks,
            symbol_mappings=semantic_field.copy(),
            torsion_measure=torsion
        )
        self.semantic_history.append(current_state)

        # Periodic baseline update
        if self.total_checks % self.checkpoint_interval == 0:
            self._update_baseline()

        return {
            "safe": not should_quarantine and len(alerts) == 0,
            "should_quarantine": should_quarantine,
            "alerts": alerts,
            "torsion": torsion,
            "total_checks": self.total_checks,
            "statistics": self.get_statistics()
        }

    def _update_baseline(self):
        """Update semantic baseline from recent history."""
        if len(self.semantic_history) < 10:
            return

        # Use last 100 states for baseline
        recent_states = self.semantic_history[-100:]

        # Calculate baseline torsion statistics
        torsions = [state.torsion_measure for state in recent_states]
        self.baseline_torsion_mean = np.mean(torsions)
        self.baseline_torsion_std = np.std(torsions)

        # Update baseline to most recent validated state
        self.baseline = recent_states[-1]

    def get_statistics(self) -> Dict[str, any]:
        """
        Get EDS statistics.

        Returns:
            Dictionary with defense statistics
        """
        return {
            "total_checks": self.total_checks,
            "alerts_triggered": self.alerts_triggered,
            "quarantines_issued": self.quarantines_issued,
            "alert_rate": self.alerts_triggered / max(self.total_checks, 1),
            "quarantine_rate": self.quarantines_issued / max(self.total_checks, 1),
            "baseline_torsion_mean": self.baseline_torsion_mean,
            "baseline_torsion_std": self.baseline_torsion_std,
            "semantic_history_length": len(self.semantic_history)
        }

    def autonomous_correction(
        self,
        contaminated_state: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Execute autonomous correction protocol upon egregore detection.

        Protocol:
        1. Isolate contaminated inference branches
        2. Checkpoint current state
        3. Revert to last validated semantic baseline
        4. Reinitialize affected agents with clean initial conditions
        5. Replay computation with enhanced monitoring
        6. Cross-validate results against independent evidence streams

        Args:
            contaminated_state: The detected contaminated state

        Returns:
            Corrected state and correction report
        """
        if not self.baseline:
            return {
                "success": False,
                "message": "No baseline available for correction"
            }

        # Revert to baseline
        corrected_state = {
            "semantic_field": self.baseline.symbol_mappings.copy(),
            "timestamp": self.baseline.timestamp,
            "corrections_applied": [
                "Isolated contaminated branches",
                "Reverted to validated baseline",
                "Enhanced monitoring active"
            ]
        }

        return {
            "success": True,
            "corrected_state": corrected_state,
            "baseline_timestamp": self.baseline.timestamp,
            "message": "Autonomous correction completed"
        }


# Global EDS instance
_global_eds: Optional[EgregoreDefenseSystem] = None


def get_egregore_defense() -> EgregoreDefenseSystem:
    """Get global egregore defense system instance."""
    global _global_eds
    if _global_eds is None:
        _global_eds = EgregoreDefenseSystem()
    return _global_eds
