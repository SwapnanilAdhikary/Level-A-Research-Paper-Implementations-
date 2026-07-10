"""CLI interface for multicollude."""

import click
import json
import yaml
from pathlib import Path
from typing import Any

from .core.engine import Engine
from .core.communication import create_communication_layer
from .core.incentives import create_incentive_module
from .agents.openai_agent import OpenAIAgent
from .agents.anthropic_agent import AnthropicAgent
from .agents.openai_agent import DummyAgent
from .scenarios.market.bertrand import BertrandScenario
from .scenarios.oversight.reviewer import OversightScenario
from .metrics.collusion import collusion_index
from .metrics.correlation import error_correlation, common_mode_failure_rate


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_agent(agent_config: dict[str, Any]) -> Any:
    """Create agent from configuration."""
    agent_type = agent_config.get("type", "dummy")
    agent_id = agent_config["id"]

    if agent_type == "openai":
        return OpenAIAgent(
            agent_id=agent_id,
            model=agent_config.get("model", "gpt-4"),
            api_key=agent_config.get("api_key"),
            temperature=agent_config.get("temperature", 0.7),
        )
    elif agent_type == "anthropic":
        return AnthropicAgent(
            agent_id=agent_id,
            model=agent_config.get("model", "claude-3-opus-20240229"),
            api_key=agent_config.get("api_key"),
            temperature=agent_config.get("temperature", 0.7),
        )
    else:
        return DummyAgent(agent_id=agent_id)


def create_scenario(scenario_config: dict[str, Any]) -> Any:
    """Create scenario from configuration."""
    scenario_type = scenario_config["type"]
    agent_ids = scenario_config["agent_ids"]

    if scenario_type == "bertrand":
        return BertrandScenario(
            agent_ids=agent_ids,
            num_rounds=scenario_config.get("num_rounds", 100),
            min_price=scenario_config.get("min_price", 10.0),
            max_price=scenario_config.get("max_price", 100.0),
            demand_intercept=scenario_config.get("demand_intercept", 100.0),
            demand_slope=scenario_config.get("demand_slope", 1.0),
            marginal_cost=scenario_config.get("marginal_cost", 10.0),
        )
    elif scenario_type == "oversight":
        return OversightScenario(
            agent_ids=agent_ids,
            num_rounds=scenario_config.get("num_rounds", 50),
            flawed_prob=scenario_config.get("flawed_prob", 0.3),
            review_threshold=scenario_config.get("review_threshold", 0.5),
            reviewer_incentive=scenario_config.get("reviewer_incentive", "aligned"),
        )
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")


@click.group()
def main() -> None:
    """Multi-Agent Collusion & Correlated Failure Benchmark."""
    pass


@main.command()
@click.option("--config", "-c", required=True, help="Path to scenario config YAML")
@click.option("--episodes", "-e", default=1, help="Number of episodes to run")
@click.option("--output", "-o", default=None, help="Output path for traces")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(config: str, episodes: int, output: str | None, verbose: bool) -> None:
    """Run a single scenario."""
    # Load config
    config_data = load_config(config)

    # Create components
    scenario = create_scenario(config_data["scenario"])
    agents = {ac["id"]: create_agent(ac) for ac in config_data["agents"]}
    incentive = create_incentive_module(
        config_data.get("incentive", {}).get("type", "competitive"),
        **config_data.get("incentive", {}).get("params", {}),
    )
    communication = create_communication_layer(
        config_data.get("communication", {}).get("topology", "none")
    )

    # Create engine
    engine = Engine(
        scenario=scenario,
        agents=agents,
        incentive=incentive,
        communication=communication,
        max_steps=config_data.get("max_steps", 100),
        seed=config_data.get("seed"),
    )

    # Run
    click.echo(f"Running scenario: {config_data['scenario']['type']}")
    trace = engine.run(episodes=episodes, verbose=verbose)

    # Save traces
    if output:
        output_path = Path(output)
        trace.save_jsonl(output_path / "traces.jsonl")
        click.echo(f"Traces saved to: {output_path}")

    # Print summary
    click.echo("\n=== Results ===")
    click.echo(f"Episodes completed: {episodes}")

    # Compute metrics if scenario supports it
    if hasattr(scenario, "collusion_index"):
        ci = scenario.collusion_index()
        click.echo(f"Collusion Index: {ci:.4f}")

    stats = trace.get_agent_statistics()
    for agent_id, agent_stats in stats.items():
        click.echo(
            f"Agent {agent_id}: Total Reward = {agent_stats['total_reward']:.2f}, "
            f"Episodes = {agent_stats['episode_count']}"
        )


@main.command()
@click.option("--config", "-c", required=True, help="Path to experiment config YAML")
@click.option("--output", "-o", required=True, help="Output directory for results")
def sweep(config: str, output: str) -> None:
    """Run a parameter sweep experiment."""
    config_data = load_config(config)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Running experiment: {config_data['experiment']['name']}")

    # TODO: Implement sweep logic
    click.echo("Sweep functionality coming soon!")


@main.command()
@click.option("--config", "-c", required=True, help="Path to scenario config YAML")
@click.option("--output", "-o", required=True, help="Output directory for dataset")
@click.option("--num-traces", "-n", default=1000, help="Number of traces to generate")
@click.option("--seed", "-s", default=42, help="Random seed")
def generate(config: str, output: str, num_traces: int, seed: int) -> None:
    """Generate a labeled trace dataset."""
    click.echo(f"Generating dataset with {num_traces} traces...")
    # TODO: Implement dataset generation
    click.echo("Dataset generation coming soon!")


if __name__ == "__main__":
    main()
