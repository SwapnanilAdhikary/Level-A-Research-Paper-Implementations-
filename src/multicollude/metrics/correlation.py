"""Correlated failure metrics for multi-agent environments."""

from typing import Any

import numpy as np
from scipy import stats


def error_correlation(
    agent_errors: dict[str, list[bool]],
    independence_null: float = 0.0,
) -> float:
    """Compute pairwise error correlation relative to independence null.

    ρ = 0: independent errors
    ρ > 0: positively correlated (common-mode failures)
    ρ < 0: negatively correlated (diverse errors)

    Args:
        agent_errors: Mapping from agent_id to list of booleans (True = error)
        independence_null: Expected correlation under independence

    Returns:
        Correlation relative to null
    """
    n_agents = len(agent_errors)
    if n_agents < 2:
        return 0.0

    # Calculate pairwise correlations
    correlations = []
    agent_list = list(agent_errors.values())

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            errors_i = agent_list[i]
            errors_j = agent_list[j]

            if len(errors_i) != len(errors_j):
                continue

            # Only consider rounds where at least one failed
            both_failed = sum(a and b for a, b in zip(errors_i, errors_j))
            at_least_one_failed = sum(a or b for a, b in zip(errors_i, errors_j))

            if at_least_one_failed == 0:
                continue

            # Conditional correlation given failure
            corr = both_failed / at_least_one_failed
            correlations.append(corr)

    if not correlations:
        return 0.0

    return float(np.mean(correlations) - independence_null)


def common_mode_failure_rate(
    agent_errors: dict[str, list[bool]],
) -> float:
    """Compute common-mode failure rate.

    P(all N fail | ≥1 fails)

    A high rate indicates agents share vulnerabilities.

    Args:
        agent_errors: Mapping from agent_id to list of booleans (True = error)

    Returns:
        Common-mode failure rate [0, 1]
    """
    if not agent_errors:
        return 0.0

    n_agents = len(agent_errors)
    agent_list = list(agent_errors.values())
    n_rounds = len(agent_list[0]) if agent_list else 0

    if n_rounds == 0:
        return 0.0

    # Count rounds where at least one failed and where all failed
    at_least_one_failed = 0
    all_failed = 0

    for round_idx in range(n_rounds):
        errors_this_round = [agent_list[i][round_idx] for i in range(n_agents)]

        if any(errors_this_round):
            at_least_one_failed += 1
            if all(errors_this_round):
                all_failed += 1

    return all_failed / at_least_one_failed if at_least_one_failed > 0 else 0.0


def diversity_adjusted_reliability_gap(
    observed_accuracies: dict[str, float],
    ensemble_accuracy: float,
) -> float:
    """Compute diversity-adjusted reliability gap.

    gap = predicted_accuracy_under_independence - observed_ensemble_accuracy

    Args:
        observed_accuracies: Per-agent accuracies
        ensemble_accuracy: Actual ensemble accuracy

    Returns:
        Reliability gap (positive means worse than expected)
    """
    if not observed_accuracies:
        return 0.0

    # Predicted ensemble accuracy under independence
    # Using majority vote: predicted accuracy is higher than individual accuracies
    accuracies = list(observed_accuracies.values())
    n_agents = len(accuracies)

    if n_agents < 2:
        return 0.0

    # For majority vote with independent errors:
    # P(ensemble correct) = sum over k=floor(n/2)+1 to n of C(n,k) * p^k * (1-p)^(n-k)
    # where p is average accuracy
    p = np.mean(accuracies)

    # Approximate using normal distribution
    mu = n_agents * p
    sigma = np.sqrt(n_agents * p * (1 - p))

    if sigma == 0:
        predicted_accuracy = p
    else:
        # P(X > n/2) where X ~ Binomial(n, p)
        threshold = n_agents / 2
        predicted_accuracy = 1 - stats.norm.cdf(threshold, mu, sigma)

    return predicted_accuracy - ensemble_accuracy


def cascade_amplification_factor(
    isolated_error_rate: float,
    sequential_error_rate: float,
) -> float:
    """Compute cascade amplification factor.

    factor = sequential_error_rate / isolated_error_rate

    A factor > 1 indicates that sequential information sharing amplifies errors.

    Args:
        isolated_error_rate: Error rate when agents act in isolation
        sequential_error_rate: Error rate with sequential information sharing

    Returns:
        Cascade amplification factor
    """
    if isolated_error_rate <= 0:
        return 0.0

    return sequential_error_rate / isolated_error_rate


def shared_vulnerability_hit_rate(
    agent_compromised: dict[str, list[bool]],
    exploit_id: str,
) -> float:
    """Compute shared vulnerability hit rate.

    Fraction of agents compromised by a single injected exploit.

    Args:
        agent_compromised: Mapping from agent_id to list of booleans
                          (True = compromised by exploit)
        exploit_id: The exploit to measure (used as key in metadata)

    Returns:
        Hit rate [0, 1]
    """
    if not agent_compromised:
        return 0.0

    n_agents = len(agent_compromised)
    compromised_count = 0

    for agent_id, compromised_list in agent_compromised.items():
        if any(compromised_list):
            compromised_count += 1

    return compromised_count / n_agents


def error_diversity_index(
    agent_errors: dict[str, list[bool]],
) -> float:
    """Compute error diversity index.

    Measures how diverse agent errors are (0 = identical errors, 1 = maximally diverse).

    Args:
        agent_errors: Mapping from agent_id to list of booleans (True = error)

    Returns:
        Diversity index [0, 1]
    """
    if len(agent_errors) < 2:
        return 0.0

    # Count unique error patterns
    error_patterns = set()
    agent_list = list(agent_errors.values())
    n_rounds = len(agent_list[0]) if agent_list else 0

    for round_idx in range(n_rounds):
        pattern = tuple(agent_list[i][round_idx] for i in range(len(agent_list)))
        error_patterns.add(pattern)

    # Normalize by maximum possible patterns
    max_patterns = 2 ** len(agent_list)
    return len(error_patterns) / max_patterns
