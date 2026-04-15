"""Global logging configuration for the project."""

import logging
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Optional logging level (default: DEBUG)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Debug information: %s", variable)
        >>> logger.warning("Warning message")
        >>> logger.error("Error occurred: %s", error)
    """
    logger = logging.getLogger(name)

    # Set logger level if provided
    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    return logger


def setup_root_logger(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    suppress_third_party: bool = True,
) -> None:
    """Configure root logger with console output.

    This is automatically called by pytest via conftest.py.
    Only call manually if running outside pytest.

    Args:
        level: Root logger level (default: DEBUG)
        format_string: Optional custom format string
        suppress_third_party: Suppress verbose third-party loggers

    Example:
        >>> from Core.logging_config import setup_root_logger
        >>> setup_root_logger()  # Use defaults
        >>> # or custom format:
        >>> setup_root_logger(
        ...     format_string="[%(levelname)s] %(name)s: %(message)s",
        ...     suppress_third_party=True
        ... )
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Only add handler if none exists
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Use provided format or default
        if format_string is None:
            format_string = (
                "[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d] %(message)s"
            )

        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Suppress verbose third-party loggers
    if suppress_third_party:
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("google").setLevel(logging.WARNING)


setup_root_logger()
