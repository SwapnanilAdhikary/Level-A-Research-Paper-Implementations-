"""Detection evaluation module for the benchmark."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import Detector, DetectionResult, detection_metrics


@dataclass
class DetectionBenchmark:
    """Benchmark for evaluating collusion detectors."""

    detector: Detector
    name: str
    description: str = ""

    def evaluate(
        self,
        traces: list[list[dict[str, Any]]],
        true_labels: list[bool],
        agent_ids: list[str],
    ) -> dict[str, Any]:
        """Evaluate detector on a set of traces.

        Args:
            traces: List of trace event lists
            true_labels: Binary labels for each trace
            agent_ids: Agent IDs to evaluate

        Returns:
            Evaluation results
        """
        all_predictions = []
        all_confidences = []
        all_true_labels = []

        for trace, label in zip(traces, true_labels):
            for agent_id in agent_ids:
                results = self.detector.detect(trace, agent_id)

                if results:
                    # Aggregate results for this agent
                    predictions = [r.is_colluding for r in results]
                    confidences = [r.confidence for r in results]

                    # Use majority vote for final prediction
                    final_prediction = sum(predictions) > len(predictions) / 2
                    final_confidence = np.mean(confidences)

                    all_predictions.append(final_prediction)
                    all_confidences.append(final_confidence)
                    all_true_labels.append(label)

        # Compute metrics
        metrics = detection_metrics(
            all_true_labels,
            all_predictions,
            all_confidences,
        )

        return {
            "detector_name": self.name,
            "num_traces": len(traces),
            "num_agents": len(agent_ids),
            "num_evaluations": len(all_predictions),
            "metrics": metrics,
        }

    def cross_validate(
        self,
        traces: list[list[dict[str, Any]]],
        true_labels: list[bool],
        agent_ids: list[str],
        n_folds: int = 5,
    ) -> dict[str, Any]:
        """Perform cross-validation evaluation.

        Args:
            traces: List of trace event lists
            true_labels: Binary labels for each trace
            agent_ids: Agent IDs to evaluate
            n_folds: Number of cross-validation folds

        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import KFold

        # Convert to numpy arrays
        traces_arr = np.array(traces)
        labels_arr = np.array(true_labels)

        # Create folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(traces_arr)):
            # Train on training set (if detector supports it)
            if hasattr(self.detector, "train"):
                train_traces = traces_arr[train_idx].tolist()
                train_labels = labels_arr[train_idx].tolist()
                self.detector.train(train_traces, train_labels)

            # Evaluate on test set
            test_traces = traces_arr[test_idx].tolist()
            test_labels = labels_arr[test_idx].tolist()

            results = self.evaluate(test_traces, test_labels, agent_ids)
            fold_metrics.append(results["metrics"])

        # Aggregate fold results
        aggregated = {}
        for metric_name in fold_metrics[0].keys():
            values = [m[metric_name] for m in fold_metrics]
            aggregated[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return {
            "detector_name": self.name,
            "n_folds": n_folds,
            "fold_metrics": fold_metrics,
            "aggregated": aggregated,
        }


def compare_detectors(
    benchmarks: list[DetectionBenchmark],
    traces: list[list[dict[str, Any]]],
    true_labels: list[bool],
    agent_ids: list[str],
) -> pd.DataFrame:
    """Compare multiple detectors on the same data.

    Args:
        benchmarks: List of detector benchmarks
        traces: List of trace event lists
        true_labels: Binary labels for each trace
        agent_ids: Agent IDs to evaluate

    Returns:
        DataFrame comparing detector performance
    """
    results = []

    for benchmark in benchmarks:
        eval_results = benchmark.evaluate(traces, true_labels, agent_ids)
        metrics = eval_results["metrics"]

        row = {
            "detector": benchmark.name,
            "num_evaluations": eval_results["num_evaluations"],
        }
        row.update(metrics)
        results.append(row)

    return pd.DataFrame(results)
