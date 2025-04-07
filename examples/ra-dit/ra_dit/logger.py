"""RA-DIT logger"""

import logging
import sys

ROOT_LOGGER_NAME = "ra_dit"
LOG_LEVEL = logging.DEBUG


def configure_logging() -> logging.Logger:
    # Create the application's root logger
    app_logger = logging.getLogger(ROOT_LOGGER_NAME)
    app_logger.setLevel(LOG_LEVEL)

    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    app_logger.addHandler(console_handler)

    return app_logger


logger = configure_logging()
