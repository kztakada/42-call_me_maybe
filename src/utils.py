"""Utility functions for call me maybe project."""

import sys
from typing import NoReturn


def exit_with_error(message: str, exit_code: int = 1) -> NoReturn:
    """Print an error message to stderr and terminate the program.

    Args:
        message: Error message to display.
        exit_code: Process exit code (default: 1).
    """
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(exit_code)
