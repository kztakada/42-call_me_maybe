"""テストで共通利用する pytest fixture"""

from pathlib import Path

import pytest


@pytest.fixture
def valid_input_files(tmp_path: Path) -> tuple[Path, Path]:
    """有効な関数定義ファイルと入力プロンプトファイルを作成する"""
    functions_file = tmp_path / "functions.json"
    functions_file.write_text('{"name": "test"}', encoding="utf-8")

    input_file = tmp_path / "input.json"
    input_file.write_text('["prompt"]', encoding="utf-8")

    return functions_file, input_file
