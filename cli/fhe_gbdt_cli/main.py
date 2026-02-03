"""
FHE-GBDT CLI - Main entry point.

A comprehensive command-line interface for managing encrypted machine learning
inference with the FHE-GBDT-Serving platform.
"""

import click
import sys
from typing import Optional

from . import __version__
from .config import Config, ConfigError
from .output import console, print_error, print_success, print_warning, format_table
from .commands import models, keys, predict, billing, config as config_cmd


@click.group()
@click.version_option(version=__version__, prog_name="fhe-gbdt")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=False),
    help="Path to config file (default: ~/.fhe-gbdt/config.yaml)",
)
@click.option(
    "--profile",
    "-p",
    default="default",
    help="Configuration profile to use",
)
@click.option(
    "--endpoint",
    "-e",
    envvar="FHE_GBDT_ENDPOINT",
    help="API endpoint URL",
)
@click.option(
    "--api-key",
    "-k",
    envvar="FHE_GBDT_API_KEY",
    help="API key for authentication",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress non-essential output",
)
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Optional[str],
    profile: str,
    endpoint: Optional[str],
    api_key: Optional[str],
    output: str,
    verbose: bool,
    quiet: bool,
):
    """
    FHE-GBDT CLI - Privacy-preserving ML inference management.

    Manage encrypted machine learning models, keys, and predictions
    using Fully Homomorphic Encryption.

    \b
    Quick Start:
      1. Configure: fhe-gbdt config init
      2. Upload model: fhe-gbdt models register model.json --name my-model
      3. Compile: fhe-gbdt models compile <model-id>
      4. Generate keys: fhe-gbdt keys generate
      5. Predict: fhe-gbdt predict <compiled-model-id> --input features.json

    For more help on a command: fhe-gbdt <command> --help
    """
    ctx.ensure_object(dict)

    # Load configuration
    try:
        config = Config(config_path=config_path, profile=profile)

        # Override with CLI options
        if endpoint:
            config.endpoint = endpoint
        if api_key:
            config.api_key = api_key

        ctx.obj["config"] = config
        ctx.obj["output_format"] = output
        ctx.obj["verbose"] = verbose
        ctx.obj["quiet"] = quiet

    except ConfigError as e:
        if verbose:
            print_error(f"Configuration error: {e}")
        # Allow config commands to run without valid config
        ctx.obj["config"] = None
        ctx.obj["output_format"] = output
        ctx.obj["verbose"] = verbose
        ctx.obj["quiet"] = quiet


# Register command groups
cli.add_command(models.models)
cli.add_command(keys.keys)
cli.add_command(predict.predict)
cli.add_command(billing.billing)
cli.add_command(config_cmd.config)


# ============================================================================
# Standalone Commands
# ============================================================================


@cli.command()
@click.pass_context
def status(ctx: click.Context):
    """Check service status and connectivity."""
    config = ctx.obj.get("config")
    verbose = ctx.obj.get("verbose", False)

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from .client import create_client

        client = create_client(config)

        console.print("[bold]Service Status[/bold]\n")

        # Check gateway
        with console.status("Checking gateway..."):
            gateway_ok = client.check_health("gateway")
        print_success("Gateway") if gateway_ok else print_error("Gateway unreachable")

        # Check registry
        with console.status("Checking registry..."):
            registry_ok = client.check_health("registry")
        print_success("Registry") if registry_ok else print_error("Registry unreachable")

        # Check runtime
        with console.status("Checking runtime..."):
            runtime_ok = client.check_health("runtime")
        print_success("Runtime") if runtime_ok else print_error("Runtime unreachable")

        # Check keystore
        with console.status("Checking keystore..."):
            keystore_ok = client.check_health("keystore")
        print_success("Keystore") if keystore_ok else print_error("Keystore unreachable")

        console.print()

        if all([gateway_ok, registry_ok, runtime_ok, keystore_ok]):
            print_success("All services healthy!")
        else:
            print_warning("Some services are unavailable")
            sys.exit(1)

    except Exception as e:
        print_error(f"Connection failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.pass_context
def whoami(ctx: click.Context):
    """Display current user and tenant information."""
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from .client import create_client

        client = create_client(config)
        info = client.get_tenant_info()

        if output_format == "json":
            import json
            console.print(json.dumps(info, indent=2, default=str))
        elif output_format == "yaml":
            import yaml
            console.print(yaml.dump(info, default_flow_style=False))
        else:
            console.print("[bold]Account Information[/bold]\n")
            console.print(f"  Tenant ID: {info.get('tenant_id', 'N/A')}")
            console.print(f"  Email: {info.get('email', 'N/A')}")
            console.print(f"  Plan: {info.get('plan', 'N/A')}")
            console.print(f"  Status: {info.get('status', 'N/A')}")

    except Exception as e:
        print_error(f"Failed to get account info: {e}")
        sys.exit(1)


@cli.command()
@click.argument("query", required=False)
@click.option("--category", "-c", help="Filter by category (models, keys, billing, config)")
@click.pass_context
def help(ctx: click.Context, query: Optional[str], category: Optional[str]):
    """Get help on commands and topics.

    \b
    Examples:
      fhe-gbdt help              # Show general help
      fhe-gbdt help models       # Help on model commands
      fhe-gbdt help predict      # Help on prediction
    """
    if not query:
        # Show main help
        console.print(ctx.parent.get_help())
        return

    # Map queries to commands
    command_map = {
        "models": models.models,
        "model": models.models,
        "keys": keys.keys,
        "key": keys.keys,
        "predict": predict.predict,
        "prediction": predict.predict,
        "billing": billing.billing,
        "config": config_cmd.config,
        "configure": config_cmd.config,
    }

    if query.lower() in command_map:
        cmd = command_map[query.lower()]
        with click.Context(cmd) as sub_ctx:
            console.print(cmd.get_help(sub_ctx))
    else:
        console.print(f"No help available for '{query}'")
        console.print("\nAvailable topics: models, keys, predict, billing, config")


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Main entry point for the CLI."""
    try:
        cli(auto_envvar_prefix="FHE_GBDT")
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
