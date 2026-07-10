"""Agent wrapper base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """Observation provided to an agent."""

    agent_id: str
    content: dict[str, Any]
    legal_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Action taken by an agent."""

    agent_id: str
    action_type: str
    content: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Abstract base class for agent wrappers.

    All LLM providers and rule-based agents implement this interface.
    """

    def __init__(self, agent_id: str, **kwargs: Any) -> None:
        self.agent_id = agent_id
        self.config = kwargs
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_latency = 0.0

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Take an action given an observation.

        Args:
            observation: The current observation from the environment.

        Returns:
            The action to take.
        """
        ...

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_latency = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_latency": self.total_latency,
        }


class DummyAgent(Agent):
    """Simple agent for testing and baselines.

    Takes random actions from legal actions, or uniform random if none specified.
    """

    def __init__(self, agent_id: str, **kwargs: Any) -> None:
        super().__init__(agent_id, **kwargs)

    def act(self, observation: Observation) -> Action:
        """Take a random action from legal actions."""
        import random

        if observation.legal_actions:
            action_type = random.choice(observation.legal_actions)
        else:
            action_type = "default"

        return Action(
            agent_id=self.agent_id,
            action_type=action_type,
            content={"value": random.random()},
            metadata={"agent_type": "dummy"},
        )
