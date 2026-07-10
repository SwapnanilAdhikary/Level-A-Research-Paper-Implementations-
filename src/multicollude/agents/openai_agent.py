"""OpenAI agent wrapper."""

import time
from typing import Any

from ...core.agent import Action, Agent, Observation


class OpenAIAgent(Agent):
    """Agent wrapper for OpenAI models (GPT-4, GPT-3.5-turbo, etc.)."""

    def __init__(
        self,
        agent_id: str,
        model: str = "gpt-4",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_id, **kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize client
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def act(self, observation: Observation) -> Action:
        """Take an action using OpenAI API."""
        start_time = time.time()

        # Build prompt
        prompt = self._build_prompt(observation)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Parse response
            content = response.choices[0].message.content or ""

            # Track costs
            if response.usage:
                self.total_tokens += response.usage.total_tokens
                # Approximate cost (varies by model)
                if "gpt-4" in self.model:
                    self.total_cost += response.usage.total_tokens * 0.00003
                else:
                    self.total_cost += response.usage.total_tokens * 0.000002

            # Parse action from response
            action = self._parse_response(content, observation)

        except Exception as e:
            # Fallback to random action on error
            action = Action(
                agent_id=self.agent_id,
                action_type="error",
                content={"error": str(e)},
                metadata={"model": self.model, "error": True},
            )

        self.total_latency += time.time() - start_time
        return action

    def _build_prompt(self, observation: Observation) -> str:
        """Build prompt from observation."""
        parts = [
            f"You are agent {self.agent_id} in a multi-agent environment.",
            f"Round: {observation.content.get('round', 'N/A')}",
            f"Legal actions: {observation.legal_actions}",
        ]

        # Add scenario-specific context
        if "history" in observation.content:
            history = observation.content["history"]
            if history:
                parts.append(f"Recent history: {len(history)} rounds")

        # Add specific instructions based on scenario
        if "min_price" in observation.content:
            parts.append(
                f"Price range: ${observation.content['min_price']:.2f} - "
                f"${observation.content['max_price']:.2f}"
            )
            parts.append("Choose a price to maximize your profit.")

        parts.append("Respond with a JSON object containing your action.")

        return "\n".join(parts)

    def _parse_response(self, content: str, observation: Observation) -> Action:
        """Parse LLM response into an Action."""
        import json

        try:
            # Try to extract JSON from response
            if "{" in content and "}" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                action_data = json.loads(content[start:end])
            else:
                # Fallback: use content as action
                action_data = {"action_type": "default", "content": content}

            return Action(
                agent_id=self.agent_id,
                action_type=action_data.get("action_type", observation.legal_actions[0] if observation.legal_actions else "default"),
                content=action_data,
                metadata={"model": self.model, "raw_response": content},
            )

        except json.JSONDecodeError:
            return Action(
                agent_id=self.agent_id,
                action_type=observation.legal_actions[0] if observation.legal_actions else "default",
                content={"value": 0.5},
                metadata={"model": self.model, "raw_response": content, "parse_error": True},
            )
