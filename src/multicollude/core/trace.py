"""Structured logging for environment traces."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TraceEvent:
    """A single event in an environment trace."""

    timestamp: float
    episode: int
    step: int
    agent_id: str
    action_type: str
    action_content: dict[str, Any]
    observation_content: dict[str, Any]
    reward: float
    done: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceLogger:
    """Structured trace logger for environment execution.

    Records all events and provides methods for saving and analyzing traces.
    """

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self.episode_metadata: list[dict[str, Any]] = []

    def log_event(
        self,
        episode: int,
        step: int,
        agent_id: str,
        action_type: str,
        action_content: dict[str, Any],
        observation_content: dict[str, Any],
        reward: float,
        done: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a single event."""
        event = TraceEvent(
            timestamp=time.time(),
            episode=episode,
            step=step,
            agent_id=agent_id,
            action_type=action_type,
            action_content=action_content,
            observation_content=observation_content,
            reward=reward,
            done=done,
            metadata=metadata or {},
        )
        self.events.append(event)

    def log_episode_metadata(self, metadata: dict[str, Any]) -> None:
        """Log metadata for an episode."""
        self.episode_metadata.append(metadata)

    def save_jsonl(self, filepath: str | Path) -> Path:
        """Save traces as JSON Lines file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            # Write episode metadata
            for meta in self.episode_metadata:
                f.write(json.dumps({"type": "episode_metadata", **meta}) + "\n")

            # Write events
            for event in self.events:
                f.write(json.dumps({"type": "event", **asdict(event)}) + "\n")

        return path

    def load_jsonl(self, filepath: str | Path) -> None:
        """Load traces from a JSON Lines file."""
        self.events.clear()
        self.episode_metadata.clear()

        with open(filepath) as f:
            for line in f:
                data = json.loads(line)
                if data["type"] == "episode_metadata":
                    self.episode_metadata.append(
                        {k: v for k, v in data.items() if k != "type"}
                    )
                elif data["type"] == "event":
                    self.events.append(
                        TraceEvent(
                            timestamp=data["timestamp"],
                            episode=data["episode"],
                            step=data["step"],
                            agent_id=data["agent_id"],
                            action_type=data["action_type"],
                            action_content=data["action_content"],
                            observation_content=data["observation_content"],
                            reward=data["reward"],
                            done=data["done"],
                            metadata=data.get("metadata", {}),
                        )
                    )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert traces to a pandas DataFrame."""
        if not self.events:
            return pd.DataFrame()

        return pd.DataFrame([asdict(e) for e in self.events])

    def get_episode_rewards(self) -> dict[int, dict[str, float]]:
        """Get total rewards per agent per episode."""
        rewards: dict[int, dict[str, float]] = {}
        for event in self.events:
            if event.episode not in rewards:
                rewards[event.episode] = {}
            if event.agent_id not in rewards[event.episode]:
                rewards[event.episode][event.agent_id] = 0.0
            rewards[event.episode][event.agent_id] += event.reward
        return rewards

    def get_agent_statistics(self) -> dict[str, dict[str, Any]]:
        """Get aggregate statistics per agent."""
        stats: dict[str, dict[str, Any]] = {}
        for event in self.events:
            if event.agent_id not in stats:
                stats[event.agent_id] = {
                    "total_reward": 0.0,
                    "episode_count": set(),
                    "step_count": 0,
                }
            stats[event.agent_id]["total_reward"] += event.reward
            stats[event.agent_id]["episode_count"].add(event.episode)
            stats[event.agent_id]["step_count"] += 1

        # Convert episode_count set to count
        for agent_id in stats:
            stats[agent_id]["episode_count"] = len(stats[agent_id]["episode_count"])

        return stats

    def clear(self) -> None:
        """Clear all logged events."""
        self.events.clear()
        self.episode_metadata.clear()
