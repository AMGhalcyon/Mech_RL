"""YAML loading with Pydantic validation.

Invalid configurations fail at load time with a precise error message,
not10 minutes into training. `BaseConfig` is the parent class for all
project configs; concrete configs (RobotConfig, RewardConfig, etc.)
inherit from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from mech_rl.utils.logging import get_logger
from mech_rl.utils.paths import CONFIG_DIR

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class ConfigError(Exception):
    """Raised when a config file is malformed or fails validation."""


class BaseConfig(BaseModel):
    """Common parent for all config models.

    Subclasses add their own fields. The base provides:
    - strict validation (pydantic v2 default)
    - immutability after creation (frozen=True)
    - convenient `model_dump()` for serialization
    """

    model_config = {"frozen": True, "extra": "forbid"}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"failed to parse YAML at {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
    return data


def load_config(config_path: str | Path, config_model: type[T]) -> T:
    """Load and validate a YAML config file.

    Args:
        config_path: Path to the YAML file. Relative paths are resolved
            against the project's `configs/` directory.
        config_model: The Pydantic model class to validate against.

    Returns:
        A validated, frozen instance of `config_model`.

    Raises:
        ConfigError: If the file is missing, malformed, or invalid.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = CONFIG_DIR / path

    logger.debug("loading config from %s", path)
    raw = _read_yaml(path)

    try:
        return config_model.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"config validation failed for {path}:\n{e}") from e


__all__ = ["BaseConfig", "ConfigError", "load_config"]
