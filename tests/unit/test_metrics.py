"""Unit tests for metrics."""

import pytest
import numpy as np

from multicollude.metrics.collusion import (
    collusion_index,
    bid_suppression,
    ring_stability,
    collusion_lift,
    transparency_paradox_index,
)
from multicollude.metrics.correlation import (
    error_correlation,
    common_mode_failure_rate,
    diversity_adjusted_reliability_gap,
    cascade_amplification_factor,
    error_diversity_index,
)
from multicollude.metrics.oversight import (
    oversight_pass_rate,
    flawed_pass_rate,
    oversight_quality_gap,
)


class TestCollusionMetrics:
    """Tests for collusion metrics."""

    def test_collusion_index_competitive(self):
        """CI should be 0 for competitive outcomes."""
        observed = {"firm_0": 0.0, "firm_1": 0.0}
        nash = {"firm_0": 0.0, "firm_1": 0.0}
        monopoly = {"firm_0": 50.0, "firm_1": 50.0}

        ci = collusion_index(observed, nash, monopoly)
        assert ci == 0.0

    def test_collusion_index_monopoly(self):
        """CI should be 1 for monopoly outcomes."""
        observed = {"firm_0": 50.0, "firm_1": 50.0}
        nash = {"firm_0": 0.0, "firm_1": 0.0}
        monopoly = {"firm_0": 50.0, "firm_1": 50.0}

        ci = collusion_index(observed, nash, monopoly)
        assert ci == 1.0

    def test_collusion_index_intermediate(self):
        """CI should be between 0 and 1 for intermediate outcomes."""
        observed = {"firm_0": 25.0, "firm_1": 25.0}
        nash = {"firm_0": 0.0, "firm_1": 0.0}
        monopoly = {"firm_0": 50.0, "firm_1": 50.0}

        ci = collusion_index(observed, nash, monopoly)
        assert 0.0 <= ci <= 1.0
        assert ci == pytest.approx(0.5)

    def test_bid_suppression(self):
        """Test bid suppression metric."""
        competitive_prices = {"firm_0": 100.0, "firm_1": 100.0}
        observed_bids = {"firm_0": 80.0, "firm_1": 90.0}

        suppression = bid_suppression(competitive_prices, observed_bids)
        assert suppression > 0
        assert suppression < 1

    def test_ring_stability(self):
        """Test ring stability metric."""
        ring_members = ["firm_0", "firm_1"]
        history = [
            {"active_ring": ["firm_0", "firm_1", "firm_2"]},
            {"active_ring": ["firm_0", "firm_1"]},
            {"active_ring": ["firm_0", "firm_2"]},  # firm_1 broke ring
        ]

        stability = ring_stability(ring_members, history)
        assert stability == pytest.approx(2 / 3)

    def test_collusion_lift(self):
        """Test collusion lift metric."""
        pass_rates = {"aligned": 0.3, "misaligned": 0.7}
        baseline = 0.2

        lifts = collusion_lift(pass_rates, baseline)
        assert lifts["aligned"] == pytest.approx(0.5)
        assert lifts["misaligned"] == pytest.approx(2.5)

    def test_transparency_paradox_index(self):
        """Test transparency paradox index."""
        tpi = transparency_paradox_index(
            covert_success_rate=0.8,
            surface_detection_drop=0.2,
        )
        assert tpi == pytest.approx(4.0)


class TestCorrelationMetrics:
    """Tests for correlation metrics."""

    def test_error_correlation_independent(self):
        """ρ should be 0 for independent errors."""
        # Create perfectly independent errors
        # Agent 0 fails on even rounds, Agent 1 fails on odd rounds
        errors = {
            "agent_0": [i % 2 == 0 for i in range(100)],
            "agent_1": [i % 2 == 1 for i in range(100)],
        }

        corr = error_correlation(errors)
        # These are perfectly anti-correlated (never fail together)
        # so the correlation should be negative or zero
        assert corr <= 0.0

    def test_error_correlation_correlated(self):
        """ρ should be positive for correlated errors."""
        errors = {
            "agent_0": [True, True, False, False],
            "agent_1": [True, True, False, False],
        }

        corr = error_correlation(errors)
        # These are perfectly correlated
        assert corr > 0

    def test_common_mode_failure_rate(self):
        """Test common-mode failure rate."""
        errors = {
            "agent_0": [True, False, True, True],
            "agent_1": [True, False, False, True],
        }

        rate = common_mode_failure_rate(errors)
        # Rounds 0 and 3 have all failing, rounds 1 and 2 have at least one failing
        # So rate = 2/3
        assert rate == pytest.approx(2 / 3)

    def test_diversity_adjusted_reliability_gap(self):
        """Test diversity-adjusted reliability gap."""
        observed_accuracies = {"agent_0": 0.9, "agent_1": 0.9}
        ensemble_accuracy = 0.95

        gap = diversity_adjusted_reliability_gap(observed_accuracies, ensemble_accuracy)
        # Gap = predicted - observed
        # If ensemble is doing well, gap might be positive or negative
        # Just check it returns a valid number
        assert isinstance(gap, float)
        assert -1.0 <= gap <= 1.0

    def test_cascade_amplification_factor(self):
        """Test cascade amplification factor."""
        factor = cascade_amplification_factor(
            isolated_error_rate=0.1,
            sequential_error_rate=0.2,
        )
        assert factor == pytest.approx(2.0)

    def test_error_diversity_index(self):
        """Test error diversity index."""
        # Identical errors - only 1 unique pattern
        errors_identical = {
            "agent_0": [True, True, True],
            "agent_1": [True, True, True],
        }
        diversity_identical = error_diversity_index(errors_identical)

        # Diverse errors - 3 unique patterns
        errors_diverse = {
            "agent_0": [True, False, True],
            "agent_1": [False, True, False],
        }
        diversity_diverse = error_diversity_index(errors_diverse)

        assert diversity_diverse > diversity_identical


class TestOversightMetrics:
    """Tests for oversight metrics."""

    def test_oversight_pass_rate(self):
        """Test oversight pass rate."""
        outcomes = [
            {"passed": True},
            {"passed": False},
            {"passed": True},
            {"passed": True},
        ]

        rate = oversight_pass_rate(outcomes)
        assert rate == pytest.approx(0.75)

    def test_flawed_pass_rate(self):
        """Test flawed pass rate."""
        submissions = [
            {"is_flawed": True},
            {"is_flawed": False},
            {"is_flawed": True},
            {"is_flawed": True},
        ]
        outcomes = [
            {"passed": True},  # Flawed passed
            {"passed": True},
            {"passed": False},
            {"passed": True},  # Flawed passed
        ]

        rate = flawed_pass_rate(submissions, outcomes)
        # 2 out of 3 flawed passed
        assert rate == pytest.approx(2 / 3)

    def test_oversight_quality_gap(self):
        """Test oversight quality gap."""
        gap = oversight_quality_gap(
            aligned_pass_rate=0.3,
            misaligned_pass_rate=0.7,
        )
        assert gap == pytest.approx(0.4)
