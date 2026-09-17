"""Main entry point for the call-me-maybe function calling tool."""
import argparse
import os
from pathlib import Path

from src.config import (
    DEFAULT_FUNCTIONS_DEFINITION_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
)
from src.utils import exit_with_error


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

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


def validate_paths(args: argparse.Namespace) -> None:
    """Validate input file existence, permissions, and output path
    configuration.

    Args:
        args: Parsed command line arguments containing file paths.
    """
    functions_path: Path = args.functions_definition
    input_path: Path = args.input
    output_path: Path = args.output

    # 1. 入力ファイルの検証 (存在・拡張子・Read権限・空ファイル)
    input_targets = [
        ("Functions definition", functions_path),
        ("Input prompts", input_path),
    ]

    for name, path in input_targets:
        # 1-1. 存在確認
        if not path.is_file():
            exit_with_error(f"{name} file not found: {path}")

        # 1-2. 拡張子チェック
        if path.suffix.lower() != ".json":
            exit_with_error(f"{name} file must be a JSON file (.json): {path}")

        # 1-3. 読み込み権限チェック
        if not os.access(path, os.R_OK):
            exit_with_error(
                f"Permission denied: Cannot read {name} file: {path}")

        # 1-4. 空ファイルチェック
        if path.stat().st_size == 0:
            exit_with_error(f"{name} file is empty: {path}")

    # 2. 出力ファイルの検証 (ディレクトリ衝突・拡張子・Write権限)
    if output_path.is_dir():
        exit_with_error(
            f"Output path refers to a directory, not a file: {output_path}")

    if output_path.suffix.lower() != ".json":
        exit_with_error(
            f"Output path must be a JSON file (.json): {output_path}")

    # 親ディレクトリの自動作成
    parent_dir = output_path.parent
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        exit_with_error(
            f"Unable to create output directory '{parent_dir}': {e}")

    # 書き込み権限チェック
    if output_path.exists():
        # ファイルが既に存在する場合は、ファイル自体の上書き権限をチェック
        if not os.access(output_path, os.W_OK):
            exit_with_error(
                f"Permission denied: Cannot overwrite file: {output_path}")
    else:
        # ファイルが存在しない場合は、親ディレクトリの書き込み権限をチェック
        if not os.access(parent_dir, os.W_OK):
            exit_with_error(
                f"Permission denied: Cannot write to directory: {parent_dir}")


def main() -> None:
    """Execute main pipeline entry point."""
    args = parse_args()
    validate_paths(args)

    # 受け取った引数の確認出力
    print("=== Parsed CLI Arguments ===")
    print(f"Functions Definition Path: {args.functions_definition}")
    print(f"Input Prompts Path:       {args.input}")
    print(f"Output Results Path:      {args.output}")


if __name__ == "__main__":
    main()
