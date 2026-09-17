"""__main__.pyのテスト"""

import argparse
import sys
from pathlib import Path

import pytest

from src.__main__ import parse_args, validate_paths

# =====================================================================
# 1. parse_args() のテスト
# =====================================================================


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parse_args returns default paths when no CLI flags are provided."""
    # sys.argv を引数なしの状態にモック
    monkeypatch.setattr(sys, "argv", ["__main__.py"])

    args = parse_args()

    assert isinstance(args.functions_definition, Path)
    assert isinstance(args.input, Path)
    assert isinstance(args.output, Path)


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parse_args correctly parses explicit custom arguments."""
    custom_argv = [
        "__main__.py",
        "--functions_definition",
        "custom_fn.json",
        "--input",
        "custom_in.json",
        "--output",
        "custom_out.json",
    ]
    monkeypatch.setattr(sys, "argv", custom_argv)

    args = parse_args()

    assert args.functions_definition == Path("custom_fn.json")
    assert args.input == Path("custom_in.json")
    assert args.output == Path("custom_out.json")

# =====================================================================
# 2. validate_paths() の正常系・異常系テスト
# =====================================================================


def test_validate_paths_success(tmp_path: Path) -> None:
    """Test validate_paths succeeds when input files exist and are valid."""
    # 正常なダミーファイルの作成
    fn_file = tmp_path / "functions.json"
    fn_file.write_text('{"name": "test"}', encoding="utf-8")

    input_file = tmp_path / "input.json"
    input_file.write_text('["prompt"]', encoding="utf-8")

    output_file = tmp_path / "out_dir" / "results.json"

    args = argparse.Namespace(
        functions_definition=fn_file,
        input=input_file,
        output=output_file,
    )

    # 正常終了すること（SystemExit が発生せず親ディレクトリが自動作成されること）を確認
    validate_paths(args)
    assert output_file.parent.exists()


def test_validate_paths_file_not_found(tmp_path: Path) -> None:
    """Test validate_paths fails when input file does not exist."""
    missing_file = tmp_path / "non_existent.json"
    valid_file = tmp_path / "input.json"
    valid_file.write_text('["prompt"]', encoding="utf-8")
    output_file = tmp_path / "results.json"

    args = argparse.Namespace(
        functions_definition=missing_file,
        input=valid_file,
        output=output_file,
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)
    assert exc_info.value.code == 1


def test_validate_paths_non_json_extension(tmp_path: Path) -> None:
    """Test validate_paths fails when file extension is not .json."""
    invalid_ext_file = tmp_path / "functions.txt"
    invalid_ext_file.write_text("dummy", encoding="utf-8")

    valid_file = tmp_path / "input.json"
    valid_file.write_text('["prompt"]', encoding="utf-8")

    args = argparse.Namespace(
        functions_definition=invalid_ext_file,
        input=valid_file,
        output=tmp_path / "results.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)
    assert exc_info.value.code == 1


def test_validate_paths_empty_file(tmp_path: Path) -> None:
    """Test validate_paths fails when input file is 0 bytes."""
    empty_file = tmp_path / "functions.json"
    empty_file.touch()  # 0バイトファイル作成

    valid_file = tmp_path / "input.json"
    valid_file.write_text('["prompt"]', encoding="utf-8")

    args = argparse.Namespace(
        functions_definition=empty_file,
        input=valid_file,
        output=tmp_path / "results.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)
    assert exc_info.value.code == 1
