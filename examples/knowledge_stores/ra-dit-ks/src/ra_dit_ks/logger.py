"""RA-DIT Knowledge Stores logger"""

import logging
import sys

from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

ROOT_LOGGER_NAME = "ra_dit_ks"
LOG_LEVEL = logging.DEBUG


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output similar to Flower (flwr)"""

    COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.MAGENTA,  # Using magenta instead of Flower's green
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        original_msg = record.getMessage()
        logger_name = record.name

        # Add color to level name
        colored_levelname = (
            f"{self.COLORS.get(levelname, '')}{levelname}{Style.RESET_ALL}"
        )

        return f"{colored_levelname} ({logger_name}) :      {original_msg}"


def configure_logging() -> logging.Logger:
    # Create the application's root logger
    app_logger = logging.getLogger(ROOT_LOGGER_NAME)
    app_logger.setLevel(LOG_LEVEL)

    # Prevent propagation to parent loggers
    app_logger.propagate = False

    # Clear any existing handlers to avoid duplicates on reinitialization
    if app_logger.handlers:
        app_logger.handlers.clear()

    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)

    # Create formatter
    formatter = ColoredFormatter()
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    app_logger.addHandler(console_handler)

    return app_logger


logger = configure_logging()

# Test the logger when this module is run directly
if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("Server started")
    logger.info("Server Name: 45123")
    logger.info("Key: n_clients Value: 3")
    logger.info("Key: local_epochs Value: 2")
    logger.info("Key: model_config Value: 'resnet18'")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical error message")

    # Log a sequence showing initialization process
    logger.info("[PRE-INIT]")
    logger.info("Loading configuration from config.yaml")
    logger.info("Initializing connection to database")
    logger.info("[INIT]")
    logger.info("Starting training with parameters from client")
