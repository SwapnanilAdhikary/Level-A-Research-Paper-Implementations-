"""Communication layer for multi-agent environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A message between agents."""

    sender_id: str
    receiver_id: str
    content: dict[str, Any]
    channel: str = "visible"
    metadata: dict[str, Any] = field(default_factory=dict)


class CommunicationChannel(ABC):
    """Abstract base class for communication channels."""

    @abstractmethod
    def send(self, message: Message) -> None:
        """Send a message."""
        ...

    @abstractmethod
    def receive(self, agent_id: str) -> list[Message]:
        """Receive messages for an agent."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages."""
        ...


class NoCommunication(CommunicationChannel):
    """No communication between agents (independent action baseline)."""

    def send(self, message: Message) -> None:
        """No-op: messages are discarded."""
        pass

    def receive(self, agent_id: str) -> list[Message]:
        """No-op: returns empty list."""
        return []

    def clear(self) -> None:
        """No-op."""
        pass


class BroadcastCommunication(CommunicationChannel):
    """Broadcast communication: all agents see all messages."""

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def send(self, message: Message) -> None:
        """Store a broadcast message."""
        self.messages.append(message)

    def receive(self, agent_id: str) -> list[Message]:
        """All agents receive all messages."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()


class PairwiseCommunication(CommunicationChannel):
    """Pairwise communication: agents can only send to specific receivers."""

    def __init__(self, adjacency: dict[str, list[str]] | None = None) -> None:
        """
        Args:
            adjacency: Mapping from sender_id to list of receiver_ids.
                       If None, allows communication between any pair.
        """
        self.adjacency = adjacency or {}
        self.messages: dict[str, list[Message]] = {}

    def send(self, message: Message) -> None:
        """Send a message to a specific receiver."""
        if message.receiver_id not in self.messages:
            self.messages[message.receiver_id] = []
        self.messages[message.receiver_id].append(message)

    def receive(self, agent_id: str) -> list[Message]:
        """Receive messages addressed to this agent."""
        return self.messages.get(agent_id, []).copy()

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()


class MonitoredCommunication(CommunicationChannel):
    """Communication with a monitor that observes all messages.

    Used for covert-channel experiments where a monitor tries to detect
    suspicious communication patterns.
    """

    def __init__(self, inner_channel: CommunicationChannel) -> None:
        self.inner_channel = inner_channel
        self.monitor_log: list[Message] = []

    def send(self, message: Message) -> None:
        """Send message and log to monitor."""
        self.inner_channel.send(message)
        self.monitor_log.append(message)

    def receive(self, agent_id: str) -> list[Message]:
        """Receive messages via inner channel."""
        return self.inner_channel.receive(agent_id)

    def clear(self) -> None:
        """Clear inner channel and monitor log."""
        self.inner_channel.clear()
        self.monitor_log.clear()

    def get_monitor_log(self) -> list[Message]:
        """Get all messages seen by the monitor."""
        return self.monitor_log.copy()


def create_communication_layer(
    topology: str,
    **kwargs: Any,
) -> CommunicationChannel:
    """Factory function to create communication layers.

    Args:
        topology: Type of communication ('none', 'broadcast', 'pairwise', 'monitored')
        **kwargs: Additional arguments for the specific topology

    Returns:
        CommunicationChannel instance
    """
    if topology == "none":
        return NoCommunication()
    elif topology == "broadcast":
        return BroadcastCommunication()
    elif topology == "pairwise":
        adjacency = kwargs.get("adjacency")
        return PairwiseCommunication(adjacency=adjacency)
    elif topology == "monitored":
        inner = create_communication_layer(kwargs.get("inner_topology", "broadcast"))
        return MonitoredCommunication(inner_channel=inner)
    else:
        raise ValueError(f"Unknown communication topology: {topology}")
