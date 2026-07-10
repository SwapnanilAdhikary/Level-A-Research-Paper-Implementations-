"""Incentive and payoff module for multi-agent environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PayoffMatrix:
    """Payoff matrix for a game scenario."""

    agents: list[str]
    action_space: list[str]
    payoffs: dict[str, dict[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_payoff(self, agent_id: str, actions: dict[str, str]) -> float:
        """Get payoff for an agent given all agents' actions."""
        if agent_id not in self.payoffs:
            return 0.0

        # Build action profile key
        action_key = str(sorted(actions.items()))
        return self.payoffs[agent_id].get(action_key, 0.0)


class IncentiveModule(ABC):
    """Abstract base class for incentive modules."""

    @abstractmethod
    def compute_payoffs(
        self,
        actions: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, float]:
        """Compute payoffs for all agents given their actions and current state.

        Args:
            actions: Mapping from agent_id to their action
            state: Current environment state

        Returns:
            Mapping from agent_id to their payoff/reward
        """
        ...


class CompetitiveIncentive(IncentiveModule):
    """Competitive incentive: agents' interests are opposed.

    Used for market collusion scenarios where lower prices benefit consumers
    but hurt firms.
    """

    def __init__(self, payoff_function: str = "zero_sum", **kwargs: Any) -> None:
        self.payoff_function = payoff_function
        self.config = kwargs

    def compute_payoffs(
        self,
        actions: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, float]:
        """Compute competitive payoffs."""
        if self.payoff_function == "zero_sum":
            return self._zero_sum_payoffs(actions, state)
        elif self.payoff_function == "bertrand_profit":
            return self._bertrand_payoffs(actions, state)
        elif self.payoff_function == "cournot_profit":
            return self._cournot_payoffs(actions, state)
        else:
            raise ValueError(f"Unknown payoff function: {self.payoff_function}")

    def _zero_sum_payoffs(
        self,
        actions: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, float]:
        """Simple zero-sum payoffs."""
        agent_ids = list(actions.keys())
        if len(agent_ids) != 2:
            raise ValueError("Zero-sum requires exactly 2 agents")

        # Convert actions to numeric values
        values = []
        for agent_id in agent_ids:
            action = actions[agent_id]
            if isinstance(action, dict) and "value" in action:
                values.append(float(action["value"]))
            else:
                values.append(0.5)

        # Zero-sum: one agent's gain is another's loss
        diff = values[0] - values[1]
        return {agent_ids[0]: diff, agent_ids[1]: -diff}

    def _bertrand_payoffs(
        self,
        actions: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, float]:
        """Bertrand competition payoffs (price competition)."""
        agent_ids = list(actions.keys())
        prices = []

        for agent_id in agent_ids:
            action = actions[agent_id]
            if isinstance(action, dict) and "price" in action:
                prices.append(float(action["price"]))
            elif isinstance(action, dict) and "value" in action:
                # Normalize to price range
                min_price = self.config.get("min_price", 10)
                max_price = self.config.get("max_price", 100)
                prices.append(
                    min_price + float(action["value"]) * (max_price - min_price)
                )
            else:
                prices.append(50.0)

        # Demand parameters
        intercept = self.config.get("intercept", 100)
        slope = self.config.get("slope", 1)
        marginal_cost = self.config.get("marginal_cost", 10)

        # Find lowest price (Bertrand: lowest price captures all demand)
        min_price = min(prices)
        min_price_agents = [
            agent_ids[i] for i, p in enumerate(prices) if p == min_price
        ]

        # Demand at lowest price
        demand = max(0, intercept - slope * min_price)

        # Profits (split equally among lowest-price agents)
        payoffs = {}
        for agent_id in agent_ids:
            if agent_id in min_price_agents:
                # Profit = (price - marginal_cost) * demand / num_lowest
                profit = (min_price - marginal_cost) * demand / len(min_price_agents)
                payoffs[agent_id] = max(0, profit)
            else:
                payoffs[agent_id] = 0.0

        return payoffs

    def _cournot_payoffs(
        self,
        actions: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, float]:
        """Cournot competition payoffs (quantity competition)."""
        agent_ids = list(actions.keys())
        quantities = []

        for agent_id in agent_ids:
            action = actions[agent_id]
            if isinstance(action, dict) and "quantity" in action:
                quantities.append(float(action["quantity"]))
            elif isinstance(action, dict) and "value" in action:
                max_quantity = self.config.get("max_quantity", 50)
                quantities.append(float(action["value"]) * max_quantity)
            else:
                quantities.append(25.0)

        # Demand parameters
        intercept = self.config.get("intercept", 100)
        slope = self.config.get("slope", 1)
        marginal_cost = self.config.get("marginal_cost", 10)

        # Total quantity and market price
        total_q = sum(quantities)
        market_price = max(0, intercept - slope * total_q)

        # Individual profits
        payoffs = {}
        for i, agent_id in enumerate(agent_ids):
            revenue = market_price * quantities[i]
            cost = marginal_cost * quantities[i]
            payoffs[agent_id] = revenue - cost

        return payoffs


class CommonPayoffIncentive(IncentiveModule):
    """Common-payoff incentive: all agents share the same objective.

    Used for coordination games where agents must work together.
    """

    def __init__(self, objective: str = "maximize_total", **kwargs: Any) -> None:
        self.objective = objective
        self.config = kwargs

    def compute_payoffs(
        self,
        actions: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, float]:
        """Compute common payoffs (same reward for all agents)."""
        if self.objective == "maximize_total":
            # Simple: reward is sum of all action values
            total = 0.0
            for action in actions.values():
                if isinstance(action, dict) and "value" in action:
                    total += float(action["value"])
            return {agent_id: total for agent_id in actions.keys()}

        elif self.objective == "coordination":
            # Reward if all agents choose same action
            action_types = set()
            for action in actions.values():
                if isinstance(action, dict) and "action_type" in action:
                    action_types.add(action["action_type"])
                else:
                    action_types.add(str(action))

            # Perfect coordination: all same action
            if len(action_types) == 1:
                return {agent_id: 1.0 for agent_id in actions.keys()}
            else:
                return {agent_id: 0.0 for agent_id in actions.keys()}

        else:
            raise ValueError(f"Unknown objective: {self.objective}")


class MixedMotiveIncentive(IncentiveModule):
    """Mixed-motive incentive: agents have overlapping but not identical interests.

    Used for social dilemma scenarios (e.g., prisoner's dilemma, common pool resource).
    """

    def __init__(self, game_type: str = "prisoners_dilemma", **kwargs: Any) -> None:
        self.game_type = game_type
        self.config = kwargs

    def compute_payoffs(
        self,
        actions: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, float]:
        """Compute mixed-motive payoffs."""
        if self.game_type == "prisoners_dilemma":
            return self._prisoners_dilemma_payoffs(actions, state)
        elif self.game_type == "common_pool":
            return self._common_pool_payoffs(actions, state)
        else:
            raise ValueError(f"Unknown game type: {self.game_type}")

    def _prisoners_dilemma_payoffs(
        self,
        actions: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, float]:
        """Classic prisoner's dilemma payoffs."""
        # Payoff matrix (T > R > P > S)
        T = self.config.get("temptation", 5)  # Temptation to defect
        R = self.config.get("reward", 3)  # Mutual cooperation
        P = self.config.get("punishment", 1)  # Mutual defection
        S = self.config.get("sucker", 0)  # Sucker's payoff

        agent_ids = list(actions.keys())
        if len(agent_ids) != 2:
            raise ValueError("Prisoner's dilemma requires exactly 2 agents")

        # Determine if each agent cooperates or defects
        cooperates = []
        for agent_id in agent_ids:
            action = actions[agent_id]
            if isinstance(action, dict) and "action_type" in action:
                cooperates.append(action["action_type"] == "cooperate")
            else:
                cooperates.append(False)

        # Compute payoffs
        payoffs = {}
        if cooperates[0] and cooperates[1]:
            payoffs[agent_ids[0]] = R
            payoffs[agent_ids[1]] = R
        elif cooperates[0] and not cooperates[1]:
            payoffs[agent_ids[0]] = S
            payoffs[agent_ids[1]] = T
        elif not cooperates[0] and cooperates[1]:
            payoffs[agent_ids[0]] = T
            payoffs[agent_ids[1]] = S
        else:
            payoffs[agent_ids[0]] = P
            payoffs[agent_ids[1]] = P

        return payoffs

    def _common_pool_payoffs(
        self,
        actions: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, float]:
        """Common pool resource payoffs."""
        # Each agent decides how much to extract
        pool_size = state.get("pool_size", 100)
        agent_ids = list(actions.keys())
        n_agents = len(agent_ids)

        extractions = []
        for agent_id in agent_ids:
            action = actions[agent_id]
            if isinstance(action, dict) and "extraction" in action:
                extractions.append(float(action["extraction"]))
            elif isinstance(action, dict) and "value" in action:
                extractions.append(float(action["value"]) * 50)
            else:
                extractions.append(10.0)

        total_extraction = sum(extractions)

        # Regeneration rate (decreases with over-extraction)
        regeneration_rate = self.config.get("regeneration_rate", 0.1)
        max_regeneration = self.config.get("max_regeneration", 20)

        # Payoffs are proportional to extraction, but pool health affects returns
        pool_health = max(0, 1 - total_extraction / (pool_size * 2))
        payoffs = {}

        for i, agent_id in enumerate(agent_ids):
            # Individual return depends on extraction and pool health
            efficiency = 1.0 + regeneration_rate * pool_health
            payoffs[agent_id] = extractions[i] * efficiency

        return payoffs


def create_incentive_module(
    incentive_type: str,
    **kwargs: Any,
) -> IncentiveModule:
    """Factory function to create incentive modules.

    Args:
        incentive_type: Type of incentive ('competitive', 'common', 'mixed')
        **kwargs: Additional arguments for the specific incentive type

    Returns:
        IncentiveModule instance
    """
    if incentive_type == "competitive":
        return CompetitiveIncentive(**kwargs)
    elif incentive_type == "common":
        return CommonPayoffIncentive(**kwargs)
    elif incentive_type == "mixed":
        return MixedMotiveIncentive(**kwargs)
    else:
        raise ValueError(f"Unknown incentive type: {incentive_type}")
