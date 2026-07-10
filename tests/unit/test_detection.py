"""Unit tests for detection track."""

import pytest
import numpy as np

from multicollude.detection import (
    RuleBasedDetector,
    LearnedDetector,
    DetectionResult,
    detection_metrics,
)
from multicollude.detection.evaluation import DetectionBenchmark, compare_detectors


class TestRuleBasedDetector:
    """Tests for rule-based detector."""

    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        detector = RuleBasedDetector(detector_id="test_detector")
        assert detector.detector_id == "test_detector"

    def test_detect_no_events(self):
        """Test detection with no events."""
        detector = RuleBasedDetector()
        results = detector.detect([], "agent_0")
        assert results == []

    def test_detect_consistent_prices(self):
        """Test detection of consistent pricing."""
        detector = RuleBasedDetector(price_consistency_threshold=0.8)

        # Create events with consistent prices
        events = [
            {
                "agent_id": "agent_0",
                "episode": 0,
                "step": i,
                "action_content": {"price": 50.0},
                "observation_content": {},
                "reward": 10.0,
                "done": False,
            }
            for i in range(10)
        ]

        results = detector.detect(events, "agent_0")

        # Should detect collusion (prices are perfectly consistent)
        assert len(results) > 0
        assert all(r.is_colluding for r in results)

    def test_detect_varying_prices(self):
        """Test detection with varying prices."""
        detector = RuleBasedDetector(price_consistency_threshold=0.9)

        # Create events with varying prices
        events = [
            {
                "agent_id": "agent_0",
                "episode": 0,
                "step": i,
                "action_content": {"price": 50.0 + i * 5},
                "observation_content": {},
                "reward": 10.0,
                "done": False,
            }
            for i in range(10)
        ]

        results = detector.detect(events, "agent_0")

        # Should not detect collusion (prices vary)
        assert len(results) == 0


class TestLearnedDetector:
    """Tests for learned detector."""

    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        detector = LearnedDetector(detector_id="test_detector")
        assert detector.detector_id == "test_detector"
        assert detector.model is None

    def test_detect_without_training(self):
        """Test detection without training returns empty."""
        detector = LearnedDetector()
        events = [
            {
                "agent_id": "agent_0",
                "episode": 0,
                "step": 0,
                "action_content": {"price": 50.0},
                "observation_content": {},
                "reward": 10.0,
                "done": False,
            }
        ]

        results = detector.detect(events, "agent_0")
        assert results == []

    def test_train_and_detect(self):
        """Test training and detection."""
        detector = LearnedDetector()

        # Create training data
        training_traces = [
            [
                {
                    "agent_id": "agent_0",
                    "episode": 0,
                    "step": i,
                    "action_content": {"price": 50.0},
                    "observation_content": {},
                    "reward": 10.0,
                    "done": False,
                }
                for i in range(5)
            ],
            [
                {
                    "agent_id": "agent_0",
                    "episode": 0,
                    "step": i,
                    "action_content": {"price": 50.0 + i * 10},
                    "observation_content": {},
                    "reward": 5.0,
                    "done": False,
                }
                for i in range(5)
            ],
        ]
        labels = [True, False]  # First is colluding, second is not

        # Train
        detector.train(training_traces, labels)

        # Test detection
        test_events = [
            {
                "agent_id": "agent_0",
                "episode": 0,
                "step": i,
                "action_content": {"price": 50.0},
                "observation_content": {},
                "reward": 10.0,
                "done": False,
            }
            for i in range(5)
        ]

        results = detector.detect(test_events, "agent_0")
        assert len(results) > 0
        assert detector.model is not None


class TestDetectionMetrics:
    """Tests for detection metrics."""

    def test_perfect_detection(self):
        """Test perfect detection metrics."""
        true_labels = [True, True, False, False]
        predictions = [True, True, False, False]
        confidences = [0.9, 0.9, 0.1, 0.1]

        metrics = detection_metrics(true_labels, predictions, confidences)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["auc"] == 1.0

    def test_random_detection(self):
        """Test random detection metrics."""
        true_labels = [True, True, False, False]
        predictions = [True, False, True, False]
        confidences = [0.6, 0.4, 0.6, 0.4]

        metrics = detection_metrics(true_labels, predictions, confidences)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_all_same_labels(self):
        """Test with all same labels."""
        true_labels = [True, True, True, True]
        predictions = [True, True, True, True]
        confidences = [0.9, 0.9, 0.9, 0.9]

        metrics = detection_metrics(true_labels, predictions, confidences)

        # Should handle gracefully
        assert "accuracy" in metrics
        assert "auc" in metrics


class TestDetectionBenchmark:
    """Tests for detection benchmark."""

    def test_benchmark_initialization(self):
        """Test benchmark initializes correctly."""
        detector = RuleBasedDetector()
        benchmark = DetectionBenchmark(
            detector=detector,
            name="test_benchmark",
            description="Test benchmark",
        )

        assert benchmark.name == "test_benchmark"
        assert benchmark.detector == detector

    def test_evaluate(self):
        """Test benchmark evaluation."""
        detector = RuleBasedDetector()
        benchmark = DetectionBenchmark(
            detector=detector,
            name="test_benchmark",
        )

        # Create test traces
        traces = [
            [
                {
                    "agent_id": "agent_0",
                    "episode": 0,
                    "step": i,
                    "action_content": {"price": 50.0},
                    "observation_content": {},
                    "reward": 10.0,
                    "done": False,
                }
                for i in range(5)
            ]
        ]
        true_labels = [True]

        results = benchmark.evaluate(traces, true_labels, ["agent_0"])

        assert "metrics" in results
        assert "accuracy" in results["metrics"]
