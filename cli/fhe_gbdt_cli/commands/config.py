"""Configuration management commands."""

import click
import json
import sys
from pathlib import Path
from typing import Optional

from ..output import console, print_error, print_success, print_warning


@click.group()
def config():
    """Manage CLI configuration.

    \b
    Commands for configuring the CLI, managing profiles,
    and setting credentials.
    """
    pass


@config.command()
@click.option("--endpoint", "-e", prompt="API endpoint", default="https://api.fhe-gbdt.dev", help="API endpoint URL")
@click.option("--api-key", "-k", prompt="API key", help="Your API key", hide_input=True)
@click.option("--profile", "-p", default="default", help="Profile name")
@click.pass_context
def init(ctx: click.Context, endpoint: str, api_key: str, profile: str):
    """Initialize CLI configuration.

    \b
    This creates a configuration file with your credentials.

    \b
    Example:
      fhe-gbdt config init
      fhe-gbdt config init --profile production
    """
    config_dir = Path.home() / ".fhe-gbdt"
    config_file = config_dir / "config.yaml"
    keys_dir = config_dir / "keys"

    # Create directories
    config_dir.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    import yaml

    if config_file.exists():
        existing_config = yaml.safe_load(config_file.read_text()) or {}
    else:
        existing_config = {"profiles": {}}

    # Update profile
    if "profiles" not in existing_config:
        existing_config["profiles"] = {}

    existing_config["profiles"][profile] = {
        "endpoint": endpoint,
        "api_key": api_key,
        "key_directory": str(keys_dir),
    }

    # Set default profile
    existing_config["default_profile"] = profile

    # Save config
    config_file.write_text(yaml.dump(existing_config, default_flow_style=False))
    config_file.chmod(0o600)  # Secure permissions

    print_success(f"Configuration saved to {config_file}")
    console.print(f"\n  Profile: {profile}")
    console.print(f"  Endpoint: {endpoint}")
    console.print(f"  Keys directory: {keys_dir}")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Generate keys: fhe-gbdt keys generate")
    console.print("  2. Upload eval keys: fhe-gbdt keys upload")
    console.print("  3. Register a model: fhe-gbdt models register model.json --name my-model")


@config.command()
@click.pass_context
def show(ctx: click.Context):
    """Show current configuration.

    \b
    Example:
      fhe-gbdt config show
    """
    output_format = ctx.obj.get("output_format", "table")
    cfg = ctx.obj.get("config")

    config_file = Path.home() / ".fhe-gbdt" / "config.yaml"

    if not config_file.exists():
        print_warning("No configuration file found.")
        console.print("Run 'fhe-gbdt config init' to create one.")
        return

    import yaml
    config_data = yaml.safe_load(config_file.read_text()) or {}

    if output_format == "json":
        # Redact API key
        for profile in config_data.get("profiles", {}).values():
            if "api_key" in profile:
                profile["api_key"] = "***REDACTED***"
        console.print(json.dumps(config_data, indent=2))
    else:
        console.print(f"\n[bold]Configuration[/bold] ({config_file})\n")

        default_profile = config_data.get("default_profile", "default")
        console.print(f"  Default profile: {default_profile}")

        for name, profile in config_data.get("profiles", {}).items():
            is_default = " (default)" if name == default_profile else ""
            console.print(f"\n  [{name}]{is_default}")
            console.print(f"    Endpoint: {profile.get('endpoint', 'N/A')}")
            console.print(f"    API Key: {'***' + profile.get('api_key', '')[-4:] if profile.get('api_key') else 'N/A'}")
            console.print(f"    Keys: {profile.get('key_directory', 'N/A')}")


@config.command()
@click.argument("key")
@click.argument("value")
@click.option("--profile", "-p", default="default", help="Profile to update")
@click.pass_context
def set(ctx: click.Context, key: str, value: str, profile: str):
    """Set a configuration value.

    \b
    Keys:
      endpoint      - API endpoint URL
      api_key       - API authentication key
      key_directory - Directory for FHE keys

    \b
    Example:
      fhe-gbdt config set endpoint https://api.example.com
      fhe-gbdt config set api_key sk_live_xxx --profile production
    """
    config_file = Path.home() / ".fhe-gbdt" / "config.yaml"

    if not config_file.exists():
        print_error("No configuration file found.")
        console.print("Run 'fhe-gbdt config init' to create one.")
        sys.exit(1)

    import yaml
    config_data = yaml.safe_load(config_file.read_text()) or {}

    if profile not in config_data.get("profiles", {}):
        print_error(f"Profile '{profile}' not found")
        sys.exit(1)

    valid_keys = ["endpoint", "api_key", "key_directory"]
    if key not in valid_keys:
        print_error(f"Unknown key '{key}'")
        console.print(f"Valid keys: {', '.join(valid_keys)}")
        sys.exit(1)

    config_data["profiles"][profile][key] = value
    config_file.write_text(yaml.dump(config_data, default_flow_style=False))

    print_success(f"Set {key} = {value if key != 'api_key' else '***'}")


@config.command()
@click.argument("name")
@click.pass_context
def use(ctx: click.Context, name: str):
    """Switch to a different profile.

    \b
    Example:
      fhe-gbdt config use production
    """
    config_file = Path.home() / ".fhe-gbdt" / "config.yaml"

    if not config_file.exists():
        print_error("No configuration file found.")
        sys.exit(1)

    import yaml
    config_data = yaml.safe_load(config_file.read_text()) or {}

    if name not in config_data.get("profiles", {}):
        print_error(f"Profile '{name}' not found")
        console.print(f"Available profiles: {', '.join(config_data.get('profiles', {}).keys())}")
        sys.exit(1)

    config_data["default_profile"] = name
    config_file.write_text(yaml.dump(config_data, default_flow_style=False))

    print_success(f"Switched to profile '{name}'")


@config.command(name="list")
@click.pass_context
def list_profiles(ctx: click.Context):
    """List available profiles.

    \b
    Example:
      fhe-gbdt config list
    """
    config_file = Path.home() / ".fhe-gbdt" / "config.yaml"

    if not config_file.exists():
        print_warning("No configuration file found.")
        return

    import yaml
    config_data = yaml.safe_load(config_file.read_text()) or {}

    default_profile = config_data.get("default_profile", "default")
    profiles = config_data.get("profiles", {})

    if not profiles:
        console.print("No profiles configured.")
        return

    console.print("\n[bold]Profiles[/bold]\n")
    for name in profiles:
        indicator = " *" if name == default_profile else "  "
        console.print(f"{indicator} {name}")


@config.command()
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx: click.Context, name: str, force: bool):
    """Delete a profile.

    \b
    Example:
      fhe-gbdt config delete old-profile
    """
    config_file = Path.home() / ".fhe-gbdt" / "config.yaml"

    if not config_file.exists():
        print_error("No configuration file found.")
        sys.exit(1)

    import yaml
    config_data = yaml.safe_load(config_file.read_text()) or {}

    if name not in config_data.get("profiles", {}):
        print_error(f"Profile '{name}' not found")
        sys.exit(1)

    if name == config_data.get("default_profile"):
        print_error("Cannot delete the default profile")
        console.print("Switch to another profile first with: fhe-gbdt config use <profile>")
        sys.exit(1)

    if not force:
        if not click.confirm(f"Delete profile '{name}'?"):
            console.print("Cancelled.")
            return

    del config_data["profiles"][name]
    config_file.write_text(yaml.dump(config_data, default_flow_style=False))

    print_success(f"Deleted profile '{name}'")


@config.command()
@click.pass_context
def path(ctx: click.Context):
    """Show configuration file path.

    \b
    Example:
      fhe-gbdt config path
    """
    config_file = Path.home() / ".fhe-gbdt" / "config.yaml"
    console.print(str(config_file))
