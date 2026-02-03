"""
Configuration management for FHE-GBDT CLI.

Handles loading, saving, and managing configuration from:
- ~/.fhe-gbdt/config.yaml
- Environment variables
- Command line overrides
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


# Default configuration directory
CONFIG_DIR = Path.home() / ".fhe-gbdt"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Environment variable prefix
ENV_PREFIX = "FHE_GBDT_"


class ProfileConfig(BaseModel):
    """Configuration for a single profile."""

    endpoint: str = Field(default="https://api.fhe-gbdt.io", description="API endpoint URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    timeout: int = Field(default=300, description="Request timeout in seconds")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")


class Config(BaseModel):
    """Main configuration model."""

    default_profile: str = Field(default="default", description="Default profile to use")
    output_format: str = Field(default="table", description="Default output format (json/table)")
    profiles: Dict[str, ProfileConfig] = Field(
        default_factory=lambda: {"default": ProfileConfig()}
    )

    class Config:
        extra = "allow"


def get_config_dir() -> Path:
    """Get the configuration directory, creating it if necessary."""
    config_dir = Path(os.environ.get(f"{ENV_PREFIX}CONFIG_DIR", str(CONFIG_DIR)))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file() -> Path:
    """Get the configuration file path."""
    return get_config_dir() / "config.yaml"


def load_config() -> Config:
    """
    Load configuration from file and environment variables.

    Priority (highest to lowest):
    1. Environment variables
    2. Config file
    3. Defaults
    """
    config_file = get_config_file()

    # Start with defaults
    config_data: Dict[str, Any] = {}

    # Load from file if exists
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                file_data = yaml.safe_load(f) or {}
                config_data.update(file_data)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in config file: {e}")
        except IOError as e:
            raise ConfigError(f"Failed to read config file: {e}")

    # Create config object
    config = Config(**config_data)

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    return config


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to configuration."""
    # Global overrides
    if env_val := os.environ.get(f"{ENV_PREFIX}DEFAULT_PROFILE"):
        config.default_profile = env_val

    if env_val := os.environ.get(f"{ENV_PREFIX}OUTPUT_FORMAT"):
        config.output_format = env_val

    # Profile-specific overrides (applied to default profile)
    profile_name = config.default_profile
    if profile_name not in config.profiles:
        config.profiles[profile_name] = ProfileConfig()

    profile = config.profiles[profile_name]

    if env_val := os.environ.get(f"{ENV_PREFIX}ENDPOINT"):
        profile.endpoint = env_val

    if env_val := os.environ.get(f"{ENV_PREFIX}API_KEY"):
        profile.api_key = env_val

    if env_val := os.environ.get(f"{ENV_PREFIX}TIMEOUT"):
        try:
            profile.timeout = int(env_val)
        except ValueError:
            pass

    if env_val := os.environ.get(f"{ENV_PREFIX}VERIFY_SSL"):
        profile.verify_ssl = env_val.lower() in ("true", "1", "yes")

    return config


def save_config(config: Config) -> None:
    """Save configuration to file."""
    config_file = get_config_file()

    try:
        config_data = config.model_dump(exclude_none=True)
        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
    except IOError as e:
        raise ConfigError(f"Failed to write config file: {e}")


def get_profile(config: Config, profile_name: Optional[str] = None) -> ProfileConfig:
    """Get a specific profile configuration."""
    name = profile_name or config.default_profile

    if name not in config.profiles:
        raise ConfigError(f"Profile '{name}' not found. Available profiles: {list(config.profiles.keys())}")

    return config.profiles[name]


def set_config_value(key: str, value: Any, profile: Optional[str] = None) -> None:
    """
    Set a configuration value.

    Args:
        key: Configuration key (e.g., 'endpoint', 'api_key', 'timeout')
        value: Value to set
        profile: Profile to update (defaults to current profile)
    """
    config = load_config()
    profile_name = profile or config.default_profile

    if profile_name not in config.profiles:
        config.profiles[profile_name] = ProfileConfig()

    profile_config = config.profiles[profile_name]

    # Handle profile-level settings
    if key in ["endpoint", "api_key", "timeout", "verify_ssl", "max_retries"]:
        if key == "timeout":
            value = int(value)
        elif key == "verify_ssl":
            value = str(value).lower() in ("true", "1", "yes")
        elif key == "max_retries":
            value = int(value)
        setattr(profile_config, key, value)
    # Handle global settings
    elif key == "default_profile":
        config.default_profile = value
    elif key == "output_format":
        if value not in ("json", "table"):
            raise ConfigError(f"Invalid output format: {value}. Must be 'json' or 'table'")
        config.output_format = value
    else:
        raise ConfigError(f"Unknown configuration key: {key}")

    save_config(config)


def get_config_value(key: str, profile: Optional[str] = None) -> Any:
    """
    Get a configuration value.

    Args:
        key: Configuration key
        profile: Profile to read from (defaults to current profile)

    Returns:
        The configuration value
    """
    config = load_config()
    profile_name = profile or config.default_profile

    # Global settings
    if key == "default_profile":
        return config.default_profile
    elif key == "output_format":
        return config.output_format

    # Profile-level settings
    if profile_name not in config.profiles:
        raise ConfigError(f"Profile '{profile_name}' not found")

    profile_config = config.profiles[profile_name]

    if hasattr(profile_config, key):
        return getattr(profile_config, key)
    else:
        raise ConfigError(f"Unknown configuration key: {key}")


def init_config(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    profile: str = "default",
) -> Path:
    """
    Initialize a new configuration file.

    Args:
        endpoint: API endpoint URL
        api_key: API key for authentication
        profile: Profile name to create

    Returns:
        Path to the created config file
    """
    config_file = get_config_file()

    # Create default config
    profile_config = ProfileConfig(
        endpoint=endpoint or "https://api.fhe-gbdt.io",
        api_key=api_key,
    )

    config = Config(
        default_profile=profile,
        output_format="table",
        profiles={profile: profile_config},
    )

    save_config(config)
    return config_file


def list_profiles() -> Dict[str, ProfileConfig]:
    """List all available profiles."""
    config = load_config()
    return config.profiles


def create_profile(name: str, base_profile: Optional[str] = None) -> None:
    """
    Create a new profile.

    Args:
        name: Name for the new profile
        base_profile: Optional profile to copy settings from
    """
    config = load_config()

    if name in config.profiles:
        raise ConfigError(f"Profile '{name}' already exists")

    if base_profile:
        if base_profile not in config.profiles:
            raise ConfigError(f"Base profile '{base_profile}' not found")
        new_profile = config.profiles[base_profile].model_copy()
    else:
        new_profile = ProfileConfig()

    config.profiles[name] = new_profile
    save_config(config)


def delete_profile(name: str) -> None:
    """
    Delete a profile.

    Args:
        name: Profile name to delete
    """
    config = load_config()

    if name not in config.profiles:
        raise ConfigError(f"Profile '{name}' not found")

    if name == config.default_profile:
        raise ConfigError("Cannot delete the default profile")

    if len(config.profiles) == 1:
        raise ConfigError("Cannot delete the last profile")

    del config.profiles[name]
    save_config(config)


class ConfigError(Exception):
    """Configuration-related error."""

    pass
