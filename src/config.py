"""設定モジュール

プロジェクト全体で使用する定数およびデフォルトパスを定義します
"""

from pathlib import Path

# デフォルトファイルパスの定義
DEFAULT_FUNCTIONS_DEFINITION_PATH: Path = Path(
    "data/input/functions_definition.json")
DEFAULT_INPUT_PATH: Path = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT_PATH: Path = Path("data/output/function_calling_results.json")
