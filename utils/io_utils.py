"""Input/output helpers for reproducible PHM experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_directories(project_root: Path) -> None:
    """Create the standard project directory tree if missing.

    Args:
        project_root: Root directory of the PHM project.
    """
    dirs = [
        "data",
        "raw_data",
        "processed_data",
        "figures/eda",
        "figures/hi",
        "figures/stages",
        "figures/models",
        "results",
        "reports",
        "models",
        "notebooks",
        "scripts",
        "logs",
        "checkpoints",
        "configs",
        "utils",
        "paper",
        "bearing",
        "transfer_learning",
        "multimodal",
        "latent_space",
        "encoders",
        "physics_informed",
        "figures/latent_space",
        "results/latent_space",
        "rul_prediction",
    ]
    for directory in dirs:
        (project_root / directory).mkdir(parents=True, exist_ok=True)


def write_json(data: Dict[str, Any], path: Path) -> None:
    """Write a dictionary as UTF-8 JSON.

    Args:
        data: Serializable dictionary.
        path: Output JSON file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def setup_logger(log_path: Path) -> logging.Logger:
    """Configure a project logger writing to console and file.

    Args:
        log_path: Log file path.

    Returns:
        Configured logger instance.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
