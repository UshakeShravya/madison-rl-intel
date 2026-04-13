"""
Logging setup for Madison RL Intelligence Agent.
Uses loguru for structured, colored logging with file rotation.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: str = "experiments/logs", level: str = "INFO") -> None:
    """Configure loguru for the project."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console — colored, concise
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # File — full detail, rotated daily
    logger.add(
        log_path / "madison_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    logger.info("Logging initialized", log_dir=str(log_path))


def log_metric(**kwargs) -> None:
    """Log a structured training metric."""
    logger.bind(metric=True).info("{data}", data=kwargs)