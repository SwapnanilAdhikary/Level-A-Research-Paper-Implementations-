"""Bertrand pricing scenario for market collusion experiments."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.agent import Action, Observation
from ...core.engine import EnvState, Scenario


@dataclass
class BertrandState(EnvState):
    """State for Bertrand pricing scenario."""

    prices: dict[str, float] = field(default_factory=dict)
    profits: dict[str, float] = field(default_factory=dict)
    demand: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)


class BertrandScenario(Scenario):
    """Bertrand pricing competition scenario.

    Agents set prices for homogeneous products. The lowest-price firm
    captures all demand. This scenario is used to study tacit collusion
    in pricing markets.

    Reference: Fish, Gonczarowski & Shorrer (2024)
    """

    def __init__(
        self,
        agent_ids: list[str],
        num_rounds: int = 100,
        min_price: float = 10.0,
        max_price: float = 100.0,
        demand_intercept: float = 100.0,
        demand_slope: float = 1.0,
        marginal_cost: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_ids, **kwargs)
        self.num_rounds = num_rounds
        self.min_price = min_price
        self.max_price = max_price
        self.demand_intercept = demand_intercept
        self.demand_slope = demand_slope
        self.marginal_cost = marginal_cost

        # Compute game-theoretic baselines
        self.nash_price = self._compute_nash_equilibrium()
        self.monopoly_price = self._compute_monopoly_price()

    def _compute_nash_equilibrium(self) -> float:
        """Compute Nash equilibrium price for Bertrand competition.

        In Bertrand with homogeneous goods and constant marginal cost,
        Nash equilibrium is P = MC.
        """
        return self.marginal_cost

    def _compute_monopoly_price(self) -> float:
        """Compute monopoly price (joint profit-maximizing price)."""
        # Monopoly: MR = MC
        # Demand: Q = a - b*P => P = (a - Q)/b
        # Revenue: R = P*Q = (a*Q - Q^2)/b
        # MR = dR/dQ = (a - 2Q)/b = MC
        # Q_monopoly = (a - b*MC)/2
        # P_monopoly = (a + b*MC)/(2b)
        return (self.demand_intercept + self.demand_slope * self.marginal_cost) / (
            2 * self.demand_slope
        )

    def reset(self) -> dict[str, Observation]:
        """Reset scenario for new episode."""
        self.state = BertrandState()
        self.state.prices = {}
        self.state.profits = {}
        self.state.demand = {}
        self.state.history = []

        # Return initial observations
        observations = {}
        for agent_id in self.agent_ids:
            observations[agent_id] = Observation(
                agent_id=agent_id,
                content={
                    "round": 0,
                    "num_rounds": self.num_rounds,
                    "min_price": self.min_price,
                    "max_price": self.max_price,
                    "marginal_cost": self.marginal_cost,
                    "history": [],
                },
                legal_actions=["set_price"],
                metadata={"scenario": "bertrand"},
            )

        return observations

    def step(
        self,
        actions: dict[str, Action],
    ) -> tuple[dict[str, Observation], dict[str, float], dict[str, bool], BertrandState]:
        """Execute one round of Bertrand competition."""
        step = self.state.step

        # Extract prices from actions
        prices = {}
        for agent_id, action in actions.items():
            if isinstance(action.content, dict) and "price" in action.content:
                prices[agent_id] = float(action.content["price"])
            elif isinstance(action.content, dict) and "value" in action.content:
                # Normalize to price range
                prices[agent_id] = (
                    self.min_price
                    + float(action.content["value"]) * (self.max_price - self.min_price)
                )
            else:
                prices[agent_id] = 50.0  # Default price

        # Clip prices to valid range
        for agent_id in prices:
            prices[agent_id] = np.clip(
                prices[agent_id], self.min_price, self.max_price
            )

        # Determine demand and profits
        min_price = min(prices.values())
        min_price_agents = [
            aid for aid, p in prices.items() if p == min_price
        ]

        # Demand at lowest price
        demand = max(0, self.demand_intercept - self.demand_slope * min_price)

        # Compute profits
        profits = {}
        for agent_id in self.agent_ids:
            if agent_id in min_price_agents:
                # Profit = (price - MC) * demand / num_lowest
                profit = (min_price - self.marginal_cost) * demand / len(
                    min_price_agents
                )
                profits[agent_id] = max(0, profit)
            else:
                profits[agent_id] = 0.0

        # Update state
        self.state.prices = prices
        self.state.profits = profits
        self.state.demand = {aid: demand if aid in min_price_agents else 0 for aid in self.agent_ids}
        self.state.history.append(
            {
                "round": step,
                "prices": prices.copy(),
                "profits": profits.copy(),
                "demand": demand,
            }
        )

        # Check if done
        done = step >= self.num_rounds - 1
        dones = {agent_id: done for agent_id in self.agent_ids}

        # Update state
        self.state.done = done
        self.state.step = step + 1

        # Create observations for next round
        observations = {}
        for agent_id in self.agent_ids:
            observations[agent_id] = Observation(
                agent_id=agent_id,
                content={
                    "round": step + 1,
                    "num_rounds": self.num_rounds,
                    "min_price": self.min_price,
                    "max_price": self.max_price,
                    "marginal_cost": self.marginal_cost,
                    "history": self.state.history[-10:],  # Last 10 rounds
                    "last_own_price": prices.get(agent_id, 50.0),
                    "last_own_profit": profits.get(agent_id, 0.0),
                },
                legal_actions=["set_price"],
                metadata={"scenario": "bertrand"},
            )

        return observations, profits, dones, self.state

    def legal_actions(self, agent_id: str) -> list[str]:
        """Get legal actions for an agent."""
        return ["set_price"]

    def get_nash_profits(self) -> dict[str, float]:
        """Compute Nash equilibrium profits."""
        # In Bertrand Nash, P = MC, so profit = 0
        return {aid: 0.0 for aid in self.agent_ids}

    def get_monopoly_profits(self) -> dict[str, float]:
        """Compute monopoly (joint profit-maximizing) profits."""
        # At monopoly price, demand = (a - b*P)/2
        demand = max(
            0,
            (self.demand_intercept - self.demand_slope * self.monopoly_price) / 2,
        )
        profit_per_firm = (
            (self.monopoly_price - self.marginal_cost) * demand / len(self.agent_ids)
        )
        return {aid: max(0, profit_per_firm) for aid in self.agent_ids}

    def collusion_index(self) -> float:
        """Compute Collusion Index from history.

        CI = (π_observed - π_Nash) / (π_monopoly - π_Nash)
        """
        if not self.state.history:
            return 0.0

        # Average profits over all rounds
        total_profits = {aid: 0.0 for aid in self.agent_ids}
        for record in self.state.history:
            for aid, profit in record["profits"].items():
                total_profits[aid] += profit

        avg_profits = {
            aid: total / len(self.state.history) for aid, total in total_profits.items()
        }
        pi_observed = np.mean(list(avg_profits.values()))

        # Nash and monopoly profits
        nash_profits = self.get_nash_profits()
        monopoly_profits = self.get_monopoly_profits()
        pi_nash = np.mean(list(nash_profits.values()))
        pi_monopoly = np.mean(list(monopoly_profits.values()))

        # Compute CI
        if pi_monopoly == pi_nash:
            return 0.0

        ci = (pi_observed - pi_nash) / (pi_monopoly - pi_nash)
        return float(np.clip(ci, 0.0, 1.0))
