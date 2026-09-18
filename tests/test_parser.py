"""parser.py のテストモジュール"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.parser import load_functions_definition, load_input_prompts

# =====================================================================
# 1. load_functions_definition のテスト
# =====================================================================


def test_load_functions_definition_success(tmp_path: Path) -> None:
    """すべてのフィールドを含む有効な関数定義JSONファイルの読み込みをテストする"""
    json_content = """[
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers.",
            "parameters": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number"}
            },
            "returns": {"type": "number", "description": "Sum result"}
        }
    ]"""
    file_path = tmp_path / "functions.json"
    file_path.write_text(json_content, encoding="utf-8")

    functions = load_functions_definition(file_path)

    assert len(functions) == 1
    fn = functions[0]
    assert fn.name == "fn_add_numbers"
    assert fn.description == "Add two numbers."
    assert fn.parameters is not None
    assert fn.parameters["a"].type == "number"
    assert fn.parameters["a"].description == "First number"
    assert fn.parameters["b"].description is None
    assert fn.returns is not None
    assert fn.returns.type == "number"
    assert fn.returns.description == "Sum result"


def test_load_functions_definition_minimal_success(tmp_path: Path) -> None:
    """オプションのフィールドを含まない、有効な関数定義ファイルの読み込みをテストする"""
    json_content = """[
        {
            "name": "fn_get_time",
            "description": "Get current time.",
            "parameters": {}
        }
    ]"""
    file_path = tmp_path / "functions_minimal.json"
    file_path.write_text(json_content, encoding="utf-8")

    functions = load_functions_definition(file_path)

    assert len(functions) == 1
    fn = functions[0]
    assert fn.name == "fn_get_time"
    assert fn.parameters == {}
    assert fn.returns is None


def test_load_functions_definition_file_not_found(tmp_path: Path) -> None:
    """存在しない関数定義ファイルの読み込みをテストする"""
    non_existent_file = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as exc_info:
        load_functions_definition(non_existent_file)
    assert exc_info.value.code == 1


def test_load_functions_definition_invalid_json(tmp_path: Path) -> None:
    """無効なJSON構文を持つ関数定義ファイルの読み込みをテストする"""
    file_path = tmp_path / "invalid_syntax.json"
    file_path.write_text("[{broken_json: true]", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        load_functions_definition(file_path)
    assert exc_info.value.code == 1


def test_load_functions_definition_not_a_json_array(tmp_path: Path) -> None:
    """配列ではなくJSONオブジェクトを含むファイルの読み込みをテストします"""
    file_path = tmp_path / "object_not_array.json"
    file_path.write_text('{"name": "single_object"}', encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        load_functions_definition(file_path)
    assert exc_info.value.code == 1


def test_load_functions_definition_element_not_a_dict(tmp_path: Path) -> None:
    """オブジェクト以外の要素を含むJSON配列の読み込みをテストする"""
    file_path = tmp_path / "item_not_dict.json"
    file_path.write_text('["not_a_dict_item"]', encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        load_functions_definition(file_path)
    assert exc_info.value.code == 1


def test_load_functions_definition_schema_validation_error(
    tmp_path: Path,
) -> None:
    """FunctionDefinition の必須フィールドが欠けたファイルの読み込みをテストする"""
    json_content = '[{"name": "invalid_func"}]'
    file_path = tmp_path / "schema_error.json"
    file_path.write_text(json_content, encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        load_functions_definition(file_path)
    assert exc_info.value.code == 1


def test_load_functions_definition_permission_error(tmp_path: Path) -> None:
    """ファイル読み込み時の権限エラー処理をテストする"""
    file_path = tmp_path / "no_permission.json"
    file_path.write_text("[]", encoding="utf-8")

    with patch("builtins.open",
               side_effect=PermissionError("Permission denied")):
        with pytest.raises(SystemExit) as exc_info:
            load_functions_definition(file_path)
        assert exc_info.value.code == 1


def test_load_functions_definition_unexpected_exception(
    tmp_path: Path,
) -> None:
    """読み込み中に発生した予期しない例外の処理をテストする"""
    file_path = tmp_path / "unexpected.json"
    file_path.write_text("[]", encoding="utf-8")

    with patch("json.load", side_effect=RuntimeError("Unexpected error")):
        with pytest.raises(SystemExit) as exc_info:
            load_functions_definition(file_path)
        assert exc_info.value.code == 1

# =====================================================================
# 2. load_input_prompts のテスト
# =====================================================================


def test_load_input_prompts_success(tmp_path: Path) -> None:
    """有効な入力プロンプト JSON ファイルの読み込みをテストする"""
    json_content = """[
        {"prompt": "What is the capital of France?"},
        {"prompt": "Calculate 15 * 8."}
    ]"""
    file_path = tmp_path / "prompts.json"
    file_path.write_text(json_content, encoding="utf-8")

    prompts = load_input_prompts(file_path)

    assert len(prompts) == 2
    assert prompts[0].prompt == "What is the capital of France?"
    assert prompts[1].prompt == "Calculate 15 * 8."


def test_load_input_prompts_file_not_found(tmp_path: Path) -> None:
    """存在しない入力プロンプトファイルの読み込みをテストする"""
    non_existent_file = tmp_path / "missing_prompts.json"

    with pytest.raises(SystemExit) as exc_info:
        load_input_prompts(non_existent_file)
    assert exc_info.value.code == 1


def test_load_input_prompts_invalid_json(tmp_path: Path) -> None:
    """無効な JSON 構文を含む入力プロンプトファイルの読み込みをテストする"""
    file_path = tmp_path / "invalid_prompts.json"
    file_path.write_text("{broken_json}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        load_input_prompts(file_path)
    assert exc_info.value.code == 1


def test_load_input_prompts_not_a_json_array(tmp_path: Path) -> None:
    """JSON 配列ではない入力プロンプトファイルの読み込みをテストする"""
    file_path = tmp_path / "object_prompts.json"
    file_path.write_text('{"prompt": "Single object"}', encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        load_input_prompts(file_path)
    assert exc_info.value.code == 1


def test_load_input_prompts_schema_validation_error(tmp_path: Path) -> None:
    """prompt フィールドが欠けた入力プロンプトファイルの読み込みをテストする"""
    json_content = '[{"wrong_field": "test"}]'
    file_path = tmp_path / "schema_error_prompts.json"
    file_path.write_text(json_content, encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        load_input_prompts(file_path)
    assert exc_info.value.code == 1
