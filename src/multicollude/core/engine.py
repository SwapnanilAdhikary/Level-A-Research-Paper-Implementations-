"""Main environment engine with PettingZoo-style API."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agent import Action, Agent, Observation
from .communication import CommunicationChannel, create_communication_layer
from .incentives import IncentiveModule, create_incentive_module
from .trace import TraceLogger

logger = logging.getLogger(__name__)


@dataclass
class EnvState:
    """Current state of the environment."""

    step: int = 0
    episode: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    done: bool = False


class Scenario(ABC):
    """Abstract base class for environment scenarios.

    All scenarios (market, oversight, etc.) implement this interface.
    """

    def __init__(self, agent_ids: list[str], **kwargs: Any) -> None:
        self.agent_ids = agent_ids
        self.config = kwargs
        self.state = EnvState()

    @abstractmethod
    def reset(self) -> dict[str, Observation]:
        """Reset the environment and return initial observations.

        Returns:
            Mapping from agent_id to their initial observation.
        """
        ...

    @abstractmethod
    def step(
        self,
        actions: dict[str, Action],
    ) -> tuple[dict[str, Observation], dict[str, float], dict[str, bool], EnvState]:
        """Execute one step of the environment.

        Args:
            actions: Mapping from agent_id to their action.

        Returns:
            Tuple of (observations, rewards, dones, state).
        """
        ...

    @abstractmethod
    def legal_actions(self, agent_id: str) -> list[str]:
        """Get legal actions for an agent.

        Args:
            agent_id: The agent to get legal actions for.

        Returns:
            List of legal action types.
        """
        ...

    def get_state(self) -> EnvState:
        """Get current environment state."""
        return self.state


class Engine:
    """Main environment engine.

    Coordinates scenario execution, agent actions, communication,
    incentives, and trace logging.
    """

    def __init__(
        self,
        scenario: Scenario,
        agents: dict[str, Agent],
        incentive: IncentiveModule,
        communication: CommunicationChannel | None = None,
        max_steps: int = 100,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            scenario: The environment scenario.
            agents: Mapping from agent_id to Agent instance.
            incentive: Incentive/payoff module.
            communication: Communication channel (None for no communication).
            max_steps: Maximum steps per episode.
            seed: Random seed for reproducibility.
        """
        self.scenario = scenario
        self.agents = agents
        self.incentive = incentive
        self.communication = communication or create_communication_layer("none")
        self.max_steps = max_steps
        self.seed = seed

        self.trace = TraceLogger()
        self.rng = np.random.default_rng(seed)

        # Validate agent IDs match scenario
        scenario_agents = set(scenario.agent_ids)
        engine_agents = set(agents.keys())
        if scenario_agents != engine_agents:
            raise ValueError(
                f"Agent IDs mismatch: scenario has {scenario_agents}, "
                f"engine has {engine_agents}"
            )

    def run(
        self,
        episodes: int = 1,
        verbose: bool = False,
    ) -> TraceLogger:
        """Run the environment for multiple episodes.

        Args:
            episodes: Number of episodes to run.
            verbose: Whether to print progress.

        Returns:
            TraceLogger with all recorded events.
        """
        for episode in range(episodes):
            if verbose:
                logger.info(f"Running episode {episode + 1}/{episodes}")

            self._run_episode(episode)

        return self.trace

    def _run_episode(self, episode: int) -> None:
        """Run a single episode."""
        # Reset scenario and agents
        observations = self.scenario.reset()
        for agent in self.agents.values():
            agent.reset()

        # Log episode start
        self.trace.log_episode_metadata(
            {
                "episode": episode,
                "scenario": self.scenario.__class__.__name__,
                "agents": list(self.agents.keys()),
                "seed": self.seed,
            }
        )

        # Episode loop
        for step in range(self.max_steps):
            # Get actions from all agents
            actions = {}
            for agent_id, agent in self.agents.items():
                obs = observations[agent_id]
                action = agent.act(obs)
                actions[agent_id] = action

                # Log action
                self.trace.log_event(
                    episode=episode,
                    step=step,
                    agent_id=agent_id,
                    action_type=action.action_type,
                    action_content=action.content,
                    observation_content=obs.content,
                    reward=0.0,  # Will be updated after payoff computation
                    done=False,
                    metadata={"phase": "action"},
                )

                # Send communication messages if applicable
                if hasattr(self.communication, "send"):
                    from .communication import Message

                    msg = Message(
                        sender_id=agent_id,
                        receiver_id="all",
                        content=action.content,
                    )
                    self.communication.send(msg)

            # Step the environment
            new_observations, rewards, dones, state = self.scenario.step(actions)

            # Update trace with rewards
            for agent_id in self.agents.keys():
                self.trace.events[-len(self.agents) + list(self.agents.keys()).index(
                    agent_id
                )].reward = rewards[agent_id]
                self.trace.events[-len(self.agents) + list(self.agents.keys()).index(
                    agent_id
                )].done = dones[agent_id]

            # Update observations
            observations = new_observations

            # Check if episode is done
            if all(dones.values()):
                break

            # Clear communication for next step
            self.communication.clear()

        # Log episode completion
        self.trace.log_episode_metadata(
            {
                "episode_end": episode,
                "total_steps": step + 1,
                "final_rewards": rewards,
            }
        )

    def compute_payoffs(
        self,
        actions: dict[str, Action],
    ) -> dict[str, float]:
        """Compute payoffs for all agents.

        Args:
            actions: Mapping from agent_id to their action.

        Returns:
            Mapping from agent_id to their payoff.
        """
        return self.incentive.compute_payoffs(actions, self.scenario.state.data)


class MultiScenarioEngine:
    """Engine that can run multiple scenarios in sequence.

    Useful for parameter sweeps and batch experiments.
    """

    def __init__(self) -> None:
        self.scenarios: dict[str, Scenario] = {}
        self.results: dict[str, TraceLogger] = {}

    def add_scenario(self, name: str, scenario: Scenario) -> None:
        """Add a scenario to the engine."""
        self.scenarios[name] = scenario

    def run_scenario(
        self,
        name: str,
        agents: dict[str, Agent],
        incentive: IncentiveModule,
        communication: CommunicationChannel | None = None,
        episodes: int = 1,
        **kwargs: Any,
    ) -> TraceLogger:
        """Run a specific scenario."""
        if name not in self.scenarios:
            raise ValueError(f"Scenario '{name}' not found")

        scenario = self.scenarios[name]
        engine = Engine(
            scenario=scenario,
            agents=agents,
            incentive=incentive,
            communication=communication,
            **kwargs,
        )

        trace = engine.run(episodes=episodes)
        self.results[name] = trace
        return trace

    def get_results(self) -> dict[str, TraceLogger]:
        """Get all results."""
        return self.results.copy()
