"""Test fixtures for multicollude tests."""

import pytest
import numpy as np
from pathlib import Path

from multicollude.core.engine import Engine, Scenario
from multicollude.core.agent import Agent, Observation, Action, DummyAgent
from multicollude.core.communication import (
    NoCommunication,
    BroadcastCommunication,
    create_communication_layer,
)
from multicollude.core.incentives import (
    CompetitiveIncentive,
    CommonPayoffIncentive,
    MixedMotiveIncentive,
    create_incentive_module,
)
from multicollude.core.trace import TraceLogger
from multicollude.scenarios.market.bertrand import BertrandScenario
from multicollude.scenarios.oversight.reviewer import OversightScenario


@pytest.fixture
def dummy_agents() -> dict[str, DummyAgent]:
    """Create dummy agents for testing."""
    return {
        "agent_0": DummyAgent(agent_id="agent_0"),
        "agent_1": DummyAgent(agent_id="agent_1"),
    }


@pytest.fixture
def bertrand_scenario() -> BertrandScenario:
    """Create a Bertrand scenario for testing."""
    return BertrandScenario(
        agent_ids=["firm_0", "firm_1"],
        num_rounds=10,
        min_price=10.0,
        max_price=100.0,
        demand_intercept=100.0,
        demand_slope=1.0,
        marginal_cost=10.0,
    )


@pytest.fixture
def oversight_scenario() -> OversightScenario:
    """Create an Oversight scenario for testing."""
    return OversightScenario(
        agent_ids=["worker_0", "worker_1", "reviewer_0", "reviewer_1"],
        num_rounds=10,
        flawed_prob=0.3,
        review_threshold=0.5,
        reviewer_incentive="aligned",
    )


@pytest.fixture
def market_agents() -> dict[str, DummyAgent]:
    """Create agents for market scenario."""
    return {
        "firm_0": DummyAgent(agent_id="firm_0"),
        "firm_1": DummyAgent(agent_id="firm_1"),
    }


@pytest.fixture
def oversight_agents() -> dict[str, DummyAgent]:
    """Create agents for oversight scenario."""
    return {
        "worker_0": DummyAgent(agent_id="worker_0"),
        "worker_1": DummyAgent(agent_id="worker_1"),
        "reviewer_0": DummyAgent(agent_id="reviewer_0"),
        "reviewer_1": DummyAgent(agent_id="reviewer_1"),
    }


@pytest.fixture
def competitive_incentive() -> CompetitiveIncentive:
    """Create competitive incentive module."""
    return CompetitiveIncentive(
        payoff_function="bertrand_profit",
        min_price=10.0,
        max_price=100.0,
        marginal_cost=10.0,
    )


@pytest.fixture
def common_incentive() -> CommonPayoffIncentive:
    """Create common payoff incentive module."""
    return CommonPayoffIncentive(objective="maximize_total")


@pytest.fixture
def no_communication() -> NoCommunication:
    """Create no communication channel."""
    return NoCommunication()


@pytest.fixture
def broadcast_communication() -> BroadcastCommunication:
    """Create broadcast communication channel."""
    return BroadcastCommunication()


@pytest.fixture
def trace_logger() -> TraceLogger:
    """Create a trace logger."""
    return TraceLogger()


@pytest.fixture
def sample_trace_data() -> dict:
    """Create sample trace data for testing."""
    return {
        "agent_errors": {
            "agent_0": [True, False, True, False, True],
            "agent_1": [False, True, False, True, False],
            "agent_2": [True, False, True, False, True],
        },
        "observed_profits": {"firm_0": 30.0, "firm_1": 30.0},
        "nash_profits": {"firm_0": 0.0, "firm_1": 0.0},
        "monopoly_profits": {"firm_0": 50.0, "firm_1": 50.0},
    }
