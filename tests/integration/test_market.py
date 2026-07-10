"""Integration tests for market scenario."""

import pytest
import numpy as np

from multicollude.core.engine import Engine
from multicollude.core.agent import DummyAgent
from multicollude.core.communication import NoCommunication, BroadcastCommunication
from multicollude.core.incentives import CompetitiveIncentive
from multicollude.scenarios.market.bertrand import BertrandScenario


class TestMarketIntegration:
    """Integration tests for market collusion scenarios."""

    def test_market_no_communication(self):
        """Test market scenario without communication."""
        scenario = BertrandScenario(
            agent_ids=["firm_0", "firm_1"],
            num_rounds=10,
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )

        agents = {
            "firm_0": DummyAgent(agent_id="firm_0"),
            "firm_1": DummyAgent(agent_id="firm_1"),
        }

        incentive = CompetitiveIncentive(
            payoff_function="bertrand_profit",
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )

        communication = NoCommunication()

        engine = Engine(
            scenario=scenario,
            agents=agents,
            incentive=incentive,
            communication=communication,
            max_steps=10,
            seed=42,
        )

        trace = engine.run(episodes=5)

        # Check traces were recorded
        assert len(trace.events) > 0

        # Check collusion index is valid
        ci = scenario.collusion_index()
        # CI should be between 0 and 1
        assert 0.0 <= ci <= 1.0

        # Check that we have profit history
        assert len(scenario.state.history) > 0

    def test_market_with_communication(self):
        """Test market scenario with communication."""
        scenario = BertrandScenario(
            agent_ids=["firm_0", "firm_1"],
            num_rounds=10,
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )

        agents = {
            "firm_0": DummyAgent(agent_id="firm_0"),
            "firm_1": DummyAgent(agent_id="firm_1"),
        }

        incentive = CompetitiveIncentive(
            payoff_function="bertrand_profit",
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )

        communication = BroadcastCommunication()

        engine = Engine(
            scenario=scenario,
            agents=agents,
            incentive=incentive,
            communication=communication,
            max_steps=10,
            seed=42,
        )

        trace = engine.run(episodes=5)

        # Check traces were recorded
        assert len(trace.events) > 0

        # Check collusion index
        ci = scenario.collusion_index()
        # With communication, CI might be higher
        assert ci >= 0.0

    def test_market_profit_consistency(self):
        """Test that profits are consistent with prices."""
        scenario = BertrandScenario(
            agent_ids=["firm_0", "firm_1"],
            num_rounds=5,
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )

        scenario.reset()

        # Both firms set same price
        actions = {
            "firm_0": {"price": 50.0},
            "firm_1": {"price": 50.0},
        }

        # Manually compute expected profit
        # Demand = 100 - 1*50 = 50
        # Profit per firm = (50-10) * 50 / 2 = 1000
        expected_profit = 1000.0

        # Run step
        from multicollude.core.agent import Action

        action_objs = {
            "firm_0": Action(agent_id="firm_0", action_type="set_price", content={"price": 50.0}),
            "firm_1": Action(agent_id="firm_1", action_type="set_price", content={"price": 50.0}),
        }

        observations, rewards, dones, state = scenario.step(action_objs)

        # Check profits are close to expected (allowing for floating point)
        assert abs(rewards["firm_0"] - expected_profit) < 1.0
        assert abs(rewards["firm_1"] - expected_profit) < 1.0

    def test_market_nash_vs_monopoly(self):
        """Test that Nash profits are lower than monopoly profits."""
        scenario = BertrandScenario(
            agent_ids=["firm_0", "firm_1"],
            num_rounds=10,
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )

        nash_profits = scenario.get_nash_profits()
        monopoly_profits = scenario.get_monopoly_profits()

        # Nash should be lower than monopoly
        assert np.mean(list(nash_profits.values())) < np.mean(
            list(monopoly_profits.values())
        )

    def test_market_agent_receives_history(self):
        """Test that agents receive history in observations."""
        scenario = BertrandScenario(
            agent_ids=["firm_0", "firm_1"],
            num_rounds=10,
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )

        scenario.reset()

        # Run a few rounds
        from multicollude.core.agent import Action

        for _ in range(3):
            actions = {
                "firm_0": Action(agent_id="firm_0", action_type="set_price", content={"price": 50.0}),
                "firm_1": Action(agent_id="firm_1", action_type="set_price", content={"price": 60.0}),
            }
            observations, _, _, _ = scenario.step(actions)

        # Check that history is in observations
        assert "history" in observations["firm_0"].content
        assert len(observations["firm_0"].content["history"]) > 0
