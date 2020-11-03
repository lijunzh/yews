import builtins
import logging
import os
import sys
from pathlib import Path
from typing import Union

try:
    import colorlog

    _HAS_COLORLOG = True

except ImportError:

    _HAS_COLORLOG = False
_Path = Union[str, Path]

# default logger name
_DEFAULT_LOGGER_NAME = __name__.split(".")[0]

# Show filename and line number in logs
_PRINT = "%(message)s"
_FORMAT = (
    "%(asctime)s %(levelname)-8s [%(filename)s: %(lineno)3d]: %(message)s"
)
_COLORED_FORMAT = "%(asctime)s %(log_color)s[%(filename)s %(lineno)3d]:%(reset)s %(message_log_color)s%(message)s"


def suppress_print() -> None:
    """Suppresses printing from the current process."""

    def ignore(*_objects, _sep=" ", _end="\n", _file=sys.stdout, _flush=False):
        pass

    builtins.print = ignore


def setup_logger(
    logger: logging.Logger,
    level: int = logging.INFO,
    stdout: bool = True,
    log_file: _Path = "stdout.log",
    colored_file: bool = True,
    enabled: bool = True,
):
    # set logger level: https://docs.python.org/3/library/logging.html#levels
    logger.setLevel(logging.DEBUG)
    # format logging string
    screen_formatter = logging.Formatter(_PRINT)
    log_formatter = logging.Formatter(_FORMAT)
    if _HAS_COLORLOG:
        colored_log_formatter = colorlog.ColoredFormatter(
            _COLORED_FORMAT,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={
                "message": {
                    "DEBUG": "cyan",
                    "INFO": "white",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red",
                }
            },
        )
    else:
        colored_log_formatter = log_formatter

    # add console and file handlers
    if stdout:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(screen_formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        if colored_file:
            file_handler.setFormatter(colored_log_formatter)
        else:
            file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    if enabled:
        logger.debug("Logging is enabled for main process %d.", os.getpid())
    else:
        logger.debug("Logging is diabled for child process %d.", os.getpid())
        suppress_print()
        logger.setLevel(40)

    if not _HAS_COLORLOG:
        logger.warning(
            "Package `colorlog` is not installed. Logging has regressed back to colorless."
        )
    elif colored_file:
        logger.debug("Colored logging is enabled.")
    else:
        logger.info(
            "Colored logging is available. Use `colored_file` parameters to enable it."
        )


def get_logger(name: str = _DEFAULT_LOGGER_NAME):
    """Retrieves the logger."""
    return logging.getLogger(name)
