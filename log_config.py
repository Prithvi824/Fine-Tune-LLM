"""
Logging configuration module for the AI_ML project.

This module provides a centralized logging configuration that:
- Routes logs to different files based on log levels
- Provides colored console output
- Includes module-specific loggers
- Handles log rotation to prevent large files

Usage:
    from log_config import get_logger

    logger = get_logger(__name__)
    logger.info("This is an info message")
    logger.error("This is an error message")
"""

# 1st party imports
import logging
import logging.handlers
from typing import Dict
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record) -> str:
        """
        Format the log record.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record.
        """

        # Add color to the log level
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class LogConfig:
    """Centralized logging configuration."""

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the LogConfig class.

        Args:
            log_dir (str, optional): Directory where log files will be stored. Defaults to "logs".
        """

        # Initialize the log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Store configured loggers to avoid duplicate configuration
        self._configured_loggers: Dict[str, logging.Logger] = {}

        # Configure root logger
        self._setup_root_logger()

    def _setup_root_logger(self) -> None:
        """Setup the root logger with basic configuration."""

        # Get the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Create formatters
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_formatter = ColoredFormatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        # Create file handlers for different log levels
        self._create_file_handlers(root_logger, file_formatter)

        # Create console handler
        self._create_console_handler(root_logger, console_formatter)

    def _create_file_handlers(
        self, logger: logging.Logger, formatter: logging.Formatter
    ) -> None:
        """Create file handlers for different log levels.

        Args:
            logger (logging.Logger): The logger to add handlers to.
            formatter (logging.Formatter): The formatter to use for the handlers.
        """

        # All logs (DEBUG and above)
        all_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "all.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        all_handler.setLevel(logging.DEBUG)
        all_handler.setFormatter(formatter)
        logger.addHandler(all_handler)

        # Info logs (INFO and above)
        info_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "info.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

        # Warning logs (WARNING and above)
        warning_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "warning.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        warning_handler.setLevel(logging.WARNING)
        warning_handler.setFormatter(formatter)
        logger.addHandler(warning_handler)

        # Error logs (ERROR and above)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "error.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    def _create_console_handler(
        self, logger: logging.Logger, formatter: logging.Formatter
    ) -> None:
        """Create console handler.

        Args:
            logger (logging.Logger): The logger to add handlers to.
            formatter (logging.Formatter): The formatter to use for the handlers.
        """

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Only show INFO and above in console
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for the specified module.

        Args:
            name (str): Name of the logger (typically __name__)

        Returns:
            logging.Logger: Configured logger instance
        """

        # If the logger is already configured, return it
        if name in self._configured_loggers:
            return self._configured_loggers[name]

        # Get the logger
        logger = logging.getLogger(name)

        # The root logger configuration will be inherited
        # Store the logger to avoid reconfiguration
        self._configured_loggers[name] = logger

        return logger


# Global instance
_log_config = LogConfig()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    This is the main function you should import and use throughout the project.

    Args:
        name (str): Name of the logger (typically __name__)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        from log_config import get_logger

        logger = get_logger(__name__)
        logger.info("Application started")
        logger.error("An error occurred")
    """
    return _log_config.get_logger(name)


def set_log_level(level: str):
    """
    Set the global log level.

    Args:
        level (str): Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """

    # Set the log level
    numeric_level = getattr(logging, level.upper(), None)

    # Check if the log level is valid
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Set the log level
    logging.getLogger().setLevel(numeric_level)


def set_console_log_level(level: str):
    """
    Set the console log level separately from file logging.

    Args:
        level (str): Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """

    # Set the log level
    numeric_level = getattr(logging, level.upper(), None)

    # Check if the log level is valid
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Find console handler and update its level
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(numeric_level)
            break
