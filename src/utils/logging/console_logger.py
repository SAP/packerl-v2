from __future__ import annotations

import logging
from typing import List
import inspect

from src.utils.utils import colored

"""
This module provides a custom global logging system for the project that mimics the behavior of ns-3's logging module.
"""

# Global console logger
_console_logger = None

# Define custom log levels
LOGIC_LEVEL = 12  # Between DEBUG (10) and INFO (20)
FUNCTION_LEVEL = 15  # Between LOGIC (12) and INFO (20)

# Add the new levels to the logging module
logging.addLevelName(LOGIC_LEVEL, "LOGIC")
logging.addLevelName(FUNCTION_LEVEL, "FUNCTION")

"""
Defines the logging severity levels as they are defined in ns3. Assigns a number to each level for comparison.
"""
LOG_LEVELS = {
    "none": logging.CRITICAL,  # we use CRITICAL for unconditional messages.
    "error": logging.ERROR,
    "warn": logging.WARNING,  # we also use WARNING for RL status messages at the iteration level (LOG_RL)
    "info": logging.INFO,
    "function": FUNCTION_LEVEL,
    "logic": LOGIC_LEVEL,
    "debug": logging.DEBUG,
    "all": -1,  # 0 represents the NOTSET level in the logging module, which does not what we want
}


def _log_console_line(severity: int, prefix: str, msg: str | List[str]):
    """
    Log a message to the console with a specific severity and prefix. If more than one line is provided in msg,
    each line will be logged separately with the same prefix.
    Args:
        severity: The logging severity level (e.g., logging.INFO, logging.ERROR)
        prefix: A prefix to add to the log message indicating the severity level/type of log
        msg: The message to log, can be a string or a list of strings.
    """
    if _console_logger:
        if isinstance(msg, str):
            _console_logger.log(severity, f"{prefix} {msg}")
        elif isinstance(msg, list):
            for line in msg:
                _console_logger.log(severity, f"{prefix} {line}")
    else:
        print(f"[LOGGER ERR] global logger not initialized. Please call setup_logging() first.")


def setup_console_logger(log_level: str):
    """
    Initialize the global console logger.
    Args:
        level: The logging level (e.g., logging.DEBUG, logging.INFO)
    """
    global _console_logger
    _console_logger = logging.getLogger("GlobalLogger")
    if log_level.lower() not in LOG_LEVELS:
        _log_console_line(logging.CRITICAL, f"{colored('[py:UC]', 54)}",
                  "Invalid log level specified. Defaulting to 'info'.")
        log_level = "info"
    _console_logger.setLevel(LOG_LEVELS[log_level.lower()])

    # Don't propagate to root (avoid root's prefix)
    _console_logger.propagate = False

    # Ensure no duplicate handlers are added
    if not _console_logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='[%(asctime)s] - %(message)s'  # or just '[%(asctime)s] %(message)s'
            # datefmt can be omitted to keep the "YYYY-mm-dd HH:MM:SS,ms" default
        )
        handler.setFormatter(formatter)
        _console_logger.addHandler(handler)

def LOG_CONSOLE_UNCOND(msg: str | List[str]):
    _log_console_line(logging.CRITICAL, f"{colored('[py:UC]', 0)}", msg)


def LOG_CONSOLE_ERROR(msg: str | List[str]):
    _log_console_line(logging.ERROR, f"{colored('[py:ER]', 196)}", msg)


def LOG_CONSOLE_WARN(msg: str | List[str]):
    _log_console_line(logging.WARNING, f"{colored('[py:WR]', 214)}", msg)


def LOG_CONSOLE_RUN(msg: str | List[str]):
    _log_console_line(logging.WARNING, f"{colored('[py:RN]', 69)}", msg)


def LOG_CONSOLE_INFO(msg: str | List[str]):
    _log_console_line(logging.INFO, f"{colored('[py:IN]', 54)}", msg)


def LOG_CONSOLE_FUNCTION(end: bool = False):
    _log_console_line(FUNCTION_LEVEL, f"{colored('[py:FN]', 94)}",
              f"{inspect.currentframe().f_back.f_globals.get('__name__', '<no module>')}"
              f".{inspect.currentframe().f_back.f_code.co_name}(){' END' if end else ''}")


def LOG_CONSOLE_LOGIC(msg: str | List[str]):
    _log_console_line(LOGIC_LEVEL, f"{colored('[py:LG]', 8)}", msg)


def LOG_CONSOLE_DEBUG(msg: str | List[str]):
    _log_console_line(logging.DEBUG, f"{colored('[py:DB]', 15)}", msg)


def LOG_CONSOLE_LEVEL_AT_LEAST(level_str: str):
    return False if _console_logger is None else LOG_LEVELS[level_str.lower()] >= _console_logger.getEffectiveLevel()


def LOG_CONSOLE_LEVEL_AT_MOST(level_str: str):
    return False if _console_logger is None else LOG_LEVELS[level_str.lower()] <= _console_logger.getEffectiveLevel()


def LOG_CONSOLE_LEVEL_EQUALS(level_str: str):
    return False if _console_logger is None else LOG_LEVELS[level_str.lower()] == _console_logger.getEffectiveLevel()


def finalize_console_logger():
    """
    Finalize the global console logger.
    """
    global _console_logger
    if _console_logger:
        for handler in _console_logger.handlers[:]:
            handler.close()
            _console_logger.removeHandler(handler)
        _console_logger = None

# ==============================================================================

if __name__ == "__main__":

    import sys
    log_level = "info" if len(sys.argv) < 2 else sys.argv[1]

    print("")
    print("the following line should print a [LOGGER ERR] message:")
    LOG_CONSOLE_UNCOND("This should not be printed")
    print("")

    setup_console_logger(log_level)  # Set global log level

    # check all log levels' log messages
    print("the following lines should print log messages:")
    LOG_CONSOLE_UNCOND("This is an unconditional message")
    LOG_CONSOLE_ERROR("This is an error message")
    LOG_CONSOLE_WARN("This is a warning message")
    LOG_CONSOLE_INFO("This is an info message")
    LOG_CONSOLE_FUNCTION()
    LOG_CONSOLE_LOGIC("This is a logic message")
    LOG_CONSOLE_DEBUG("This is a debug message")
    print("")

    # check log level comparison functions
    LOG_CONSOLE_UNCOND(f"logger is set to: {logging.getLevelName(_console_logger.getEffectiveLevel())}")
    for level_str in LOG_LEVELS.keys():
        print(f"LOG_LEVEL_AT_LEAST('{level_str}'): {LOG_CONSOLE_LEVEL_AT_LEAST(level_str)}")
        print(f"LOG_LEVEL_EQUALS('{level_str}'): {LOG_CONSOLE_LEVEL_EQUALS(level_str)}")
