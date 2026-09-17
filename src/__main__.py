"""Main entry point for the call-me-maybe function calling tool."""
import argparse
from pathlib import Path

from src.config import (
    DEFAULT_FUNCTIONS_DEFINITION_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for mandatory requirements.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Function calling tool with constrained decoding."
    )

    # オプション引数 (デフォルト値付き)
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_DEFINITION_PATH,
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the input test prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output results JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    """Execute main pipeline entry point."""
    args = parse_args()

    # 受け取った引数の確認出力
    print("=== Parsed CLI Arguments ===")
    print(f"Functions Definition Path: {args.functions_definition}")
    print(f"Input Prompts Path:       {args.input}")
    print(f"Output Results Path:      {args.output}")


if __name__ == "__main__":
    main()
