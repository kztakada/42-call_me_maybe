"""Configuration constants for call me maybe project."""

from pathlib import Path

# デフォルトファイルパスの定義
DEFAULT_FUNCTIONS_DEFINITION_PATH: Path = Path(
    "data/input/functions_definition.json")
DEFAULT_INPUT_PATH: Path = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT_PATH: Path = Path("data/output/function_calling_results.json")
