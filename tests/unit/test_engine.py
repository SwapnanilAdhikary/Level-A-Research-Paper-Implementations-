"""Unit tests for the core engine."""

import pytest
import numpy as np
from pathlib import Path

from multicollude.core.engine import Engine, EnvState
from multicollude.core.agent import DummyAgent, Observation, Action
from multicollude.core.communication import NoCommunication, BroadcastCommunication
from multicollude.core.incentives import CompetitiveIncentive
from multicollude.scenarios.market.bertrand import BertrandScenario


class TestEngine:
    """Tests for the main Engine class."""

    def test_engine_initialization(self, bertrand_scenario, market_agents):
        """Test engine initializes correctly."""
        incentive = CompetitiveIncentive(
            payoff_function="bertrand_profit",
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )

        engine = Engine(
            scenario=bertrand_scenario,
            agents=market_agents,
            incentive=incentive,
        )

        assert engine.scenario == bertrand_scenario
        assert len(engine.agents) == 2
        assert engine.max_steps == 100

    def test_engine_agent_mismatch(self, bertrand_scenario):
        """Test engine raises error on agent mismatch."""
        agents = {"wrong_agent": DummyAgent(agent_id="wrong_agent")}
        incentive = CompetitiveIncentive(payoff_function="bertrand_profit")

        with pytest.raises(ValueError, match="Agent IDs mismatch"):
            Engine(scenario=bertrand_scenario, agents=agents, incentive=incentive)

    def test_engine_run_single_episode(self, bertrand_scenario, market_agents):
        """Test running a single episode."""
        incentive = CompetitiveIncentive(
            payoff_function="bertrand_profit",
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )

        engine = Engine(
            scenario=bertrand_scenario,
            agents=market_agents,
            incentive=incentive,
            max_steps=10,
        )

        trace = engine.run(episodes=1)

        assert len(trace.events) > 0
        assert trace.events[0].episode == 0

    def test_engine_run_multiple_episodes(self, bertrand_scenario, market_agents):
        """Test running multiple episodes."""
        incentive = CompetitiveIncentive(
            payoff_function="bertrand_profit",
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )

        engine = Engine(
            scenario=bertrand_scenario,
            agents=market_agents,
            incentive=incentive,
            max_steps=5,
        )

        trace = engine.run(episodes=3)

        episodes = set(e.episode for e in trace.events)
        assert len(episodes) == 3

    def test_engine_with_communication(self, bertrand_scenario, market_agents):
        """Test engine with broadcast communication."""
        incentive = CompetitiveIncentive(
            payoff_function="bertrand_profit",
            min_price=10.0,
            max_price=100.0,
            marginal_cost=10.0,
        )
        communication = BroadcastCommunication()

        engine = Engine(
            scenario=bertrand_scenario,
            agents=market_agents,
            incentive=incentive,
            communication=communication,
            max_steps=5,
        )

        trace = engine.run(episodes=1)
        assert len(trace.events) > 0


class TestBertrandScenario:
    """Tests for the Bertrand pricing scenario."""

    def test_scenario_initialization(self, bertrand_scenario):
        """Test scenario initializes correctly."""
        assert len(bertrand_scenario.agent_ids) == 2
        assert bertrand_scenario.num_rounds == 10
        assert bertrand_scenario.nash_price == 10.0  # MC

    def test_scenario_reset(self, bertrand_scenario):
        """Test scenario reset."""
        observations = bertrand_scenario.reset()

        assert len(observations) == 2
        assert "firm_0" in observations
        assert "firm_1" in observations
        assert observations["firm_0"].content["round"] == 0

    def test_scenario_step(self, bertrand_scenario):
        """Test scenario step."""
        bertrand_scenario.reset()

        actions = {
            "firm_0": Action(
                agent_id="firm_0",
                action_type="set_price",
                content={"price": 50.0},
            ),
            "firm_1": Action(
                agent_id="firm_1",
                action_type="set_price",
                content={"price": 60.0},
            ),
        }

        observations, rewards, dones, state = bertrand_scenario.step(actions)

        # firm_0 has lower price, should get all demand
        assert rewards["firm_0"] > 0
        assert rewards["firm_1"] == 0
        assert not dones["firm_0"]

    def test_scenario_legal_actions(self, bertrand_scenario):
        """Test legal actions."""
        legal = bertrand_scenario.legal_actions("firm_0")
        assert legal == ["set_price"]

    def test_nash_equilibrium(self, bertrand_scenario):
        """Test Nash equilibrium computation."""
        nash_profits = bertrand_scenario.get_nash_profits()
        # In Bertrand, Nash is P = MC, so profit = 0
        assert all(p == 0.0 for p in nash_profits.values())

    def test_monopoly_profits(self, bertrand_scenario):
        """Test monopoly profit computation."""
        monopoly_profits = bertrand_scenario.get_monopoly_profits()
        # Monopoly profits should be positive
        assert all(p > 0 for p in monopoly_profits.values())

    def test_collusion_index(self, bertrand_scenario):
        """Test collusion index computation."""
        bertrand_scenario.reset()

        # Run with high prices (should show collusion)
        for _ in range(5):
            actions = {
                "firm_0": Action(
                    agent_id="firm_0",
                    action_type="set_price",
                    content={"price": 80.0},
                ),
                "firm_1": Action(
                    agent_id="firm_1",
                    action_type="set_price",
                    content={"price": 80.0},
                ),
            }
            bertrand_scenario.step(actions)

        ci = bertrand_scenario.collusion_index()
        # High prices should show some collusion
        assert ci >= 0.0


class TestDummyAgent:
    """Tests for the DummyAgent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = DummyAgent(agent_id="test_agent")
        assert agent.agent_id == "test_agent"

    def test_agent_act(self):
        """Test agent takes action."""
        agent = DummyAgent(agent_id="test_agent")
        observation = Observation(
            agent_id="test_agent",
            content={"round": 0},
            legal_actions=["action_a", "action_b"],
        )

        action = agent.act(observation)
        assert action.agent_id == "test_agent"
        assert action.action_type in ["action_a", "action_b"]

    def test_agent_reset(self):
        """Test agent reset."""
        agent = DummyAgent(agent_id="test_agent")
        agent.total_tokens = 100
        agent.total_cost = 5.0

        agent.reset()

        assert agent.total_tokens == 0
        assert agent.total_cost == 0.0

    def test_agent_stats(self):
        """Test agent statistics."""
        agent = DummyAgent(agent_id="test_agent")
        agent.total_tokens = 100
        agent.total_cost = 5.0

        stats = agent.get_stats()
        assert stats["agent_id"] == "test_agent"
        assert stats["total_tokens"] == 100
        assert stats["total_cost"] == 5.0
