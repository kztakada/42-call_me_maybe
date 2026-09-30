"""__main__.pyのテストモジュール"""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

from src import __main__ as main_module
from src.__main__ import parse_args, validate_paths

# =====================================================================
# 1. parse_args() のテスト
# =====================================================================


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLIフラグが指定されていない場合、parse_argsがデフォルトのパスを返すことをテストする

    Args:
        monkeypatch: pytestのモンキーパッチ機能を利用してsys.argvをモックするためのフィクスチャ
    """
    # sys.argv を引数なしの状態にモック
    monkeypatch.setattr(sys, "argv", ["__main__.py"])

    args = parse_args()

    assert args.functions_definition == Path(
        "data/input/functions_definition.json"
    )
    assert args.input == Path("data/input/function_calling_tests.json")
    assert args.output == Path("data/output/function_calling_results.json")


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """parse_argsが明示的なカスタム引数を正しく解析することをテストする"""
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


def test_validate_paths_success(
    tmp_path: Path, valid_input_files: tuple[Path, Path]
) -> None:
    """入力ファイルが存在し、かつ有効である場合、validate_pathsテストは成功する"""
    fn_file, input_file = valid_input_files

    output_file = tmp_path / "out_dir" / "results.json"

    args = argparse.Namespace(
        functions_definition=fn_file,
        input=input_file,
        output=output_file,
    )

    # 正常終了すること（SystemExit が発生せず親ディレクトリが自動作成されること）を確認
    validate_paths(args)
    assert output_file.parent.exists()


def test_validate_paths_file_not_found(
    tmp_path: Path, valid_input_files: tuple[Path, Path]
) -> None:
    """入力ファイルが存在しない場合、validate_pathsテストは失敗する."""
    missing_file = tmp_path / "non_existent.json"
    _, valid_file = valid_input_files
    output_file = tmp_path / "results.json"

    args = argparse.Namespace(
        functions_definition=missing_file,
        input=valid_file,
        output=output_file,
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)
    assert exc_info.value.code == 1


def test_validate_paths_input_file_not_found(
    tmp_path: Path, valid_input_files: tuple[Path, Path]
) -> None:
    """入力プロンプトファイルが存在しない場合に終了することを確認する"""
    functions_file, _ = valid_input_files
    missing_file = tmp_path / "non_existent_input.json"

    args = argparse.Namespace(
        functions_definition=functions_file,
        input=missing_file,
        output=tmp_path / "results.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)

    assert exc_info.value.code == 1


def test_validate_paths_non_json_extension(
    tmp_path: Path, valid_input_files: tuple[Path, Path]
) -> None:
    """ファイル拡張子が .json でない場合、validate_paths テストが失敗します"""
    invalid_ext_file = tmp_path / "functions.txt"
    invalid_ext_file.write_text("dummy", encoding="utf-8")

    _, valid_file = valid_input_files

    args = argparse.Namespace(
        functions_definition=invalid_ext_file,
        input=valid_file,
        output=tmp_path / "results.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)
    assert exc_info.value.code == 1


def test_validate_paths_input_non_json_extension(
    tmp_path: Path, valid_input_files: tuple[Path, Path]
) -> None:
    """入力プロンプトファイルの拡張子も検証されることを確認する"""
    functions_file, _ = valid_input_files
    invalid_ext_file = tmp_path / "input.txt"
    invalid_ext_file.write_text("dummy", encoding="utf-8")

    args = argparse.Namespace(
        functions_definition=functions_file,
        input=invalid_ext_file,
        output=tmp_path / "results.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)

    assert exc_info.value.code == 1


def test_validate_paths_accepts_uppercase_json_extension(
    tmp_path: Path,
) -> None:
    """JSON拡張子の大文字小文字を区別しないことを確認する
    """
    functions_file = tmp_path / "functions.JSON"
    functions_file.write_text('{"name": "test"}', encoding="utf-8")
    input_file = tmp_path / "input.JsOn"
    input_file.write_text('["prompt"]', encoding="utf-8")

    args = argparse.Namespace(
        functions_definition=functions_file,
        input=input_file,
        output=tmp_path / "results.JsOn",
    )

    validate_paths(args)


def test_validate_paths_empty_file(
    tmp_path: Path, valid_input_files: tuple[Path, Path]
) -> None:
    """入力ファイルが0バイトの場合、テスト validate_paths が失敗します"""
    empty_file = tmp_path / "functions.json"
    empty_file.write_text("", encoding="utf-8")

    _, valid_file = valid_input_files

    args = argparse.Namespace(
        functions_definition=empty_file,
        input=valid_file,
        output=tmp_path / "results.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)
    assert exc_info.value.code == 1


def test_validate_paths_empty_input_file(
    tmp_path: Path, valid_input_files: tuple[Path, Path]
) -> None:
    """入力プロンプトファイルが空の場合に終了することを確認する"""
    functions_file, _ = valid_input_files
    empty_file = tmp_path / "input.json"
    empty_file.write_text("", encoding="utf-8")

    args = argparse.Namespace(
        functions_definition=functions_file,
        input=empty_file,
        output=tmp_path / "results.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)

    assert exc_info.value.code == 1


def test_validate_paths_rejects_output_directory(
    tmp_path: Path, valid_input_files: tuple[Path, Path]
) -> None:
    """出力先がディレクトリの場合に終了することを確認する"""
    functions_file, input_file = valid_input_files
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    args = argparse.Namespace(
        functions_definition=functions_file,
        input=input_file,
        output=output_dir,
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)

    assert exc_info.value.code == 1


def test_validate_paths_rejects_non_json_output(
    tmp_path: Path, valid_input_files: tuple[Path, Path]
) -> None:
    """出力先の拡張子もJSONに限定されることを確認する"""
    functions_file, input_file = valid_input_files

    args = argparse.Namespace(
        functions_definition=functions_file,
        input=input_file,
        output=tmp_path / "results.txt",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)

    assert exc_info.value.code == 1


def test_validate_paths_rejects_unreadable_input(
    tmp_path: Path,
    valid_input_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """入力ファイルの読み込み権限がない場合に終了することを確認する"""
    functions_file, input_file = valid_input_files

    def deny_input_read(path: Path, mode: int) -> bool:
        return path != input_file

    monkeypatch.setattr(os, "access", deny_input_read)
    args = argparse.Namespace(
        functions_definition=functions_file,
        input=input_file,
        output=tmp_path / "results.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)

    assert exc_info.value.code == 1


def test_validate_paths_rejects_existing_unwritable_output(
    tmp_path: Path,
    valid_input_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """既存の出力ファイルに書き込み権限がない場合に終了することを確認する"""
    functions_file, input_file = valid_input_files
    output_file = tmp_path / "results.json"
    output_file.write_text("[]", encoding="utf-8")

    def deny_output_write(path: Path, mode: int) -> bool:
        return path != output_file

    monkeypatch.setattr(os, "access", deny_output_write)
    args = argparse.Namespace(
        functions_definition=functions_file,
        input=input_file,
        output=output_file,
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)

    assert exc_info.value.code == 1


def test_validate_paths_rejects_unwritable_output_directory(
    tmp_path: Path,
    valid_input_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新規出力ファイルを作れない場合に終了することを確認する"""
    functions_file, input_file = valid_input_files

    def deny_directory_write(path: Path, mode: int) -> bool:
        return path != tmp_path

    monkeypatch.setattr(os, "access", deny_directory_write)
    args = argparse.Namespace(
        functions_definition=functions_file,
        input=input_file,
        output=tmp_path / "results.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)

    assert exc_info.value.code == 1


def test_validate_paths_rejects_output_directory_creation_failure(
    tmp_path: Path,
    valid_input_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """出力先ディレクトリを作成できない場合に終了することを確認する"""
    functions_file, input_file = valid_input_files

    def fail_mkdir(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        raise OSError("cannot create directory")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    args = argparse.Namespace(
        functions_definition=functions_file,
        input=input_file,
        output=tmp_path / "nested" / "results.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_paths(args)

    assert exc_info.value.code == 1


def test_main_runs_pipeline_and_writes_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mainがパイプラインを実行し、結果をJSONとして保存することを確認する"""
    output_file = tmp_path / "results.json"
    args = argparse.Namespace(
        functions_definition=tmp_path / "functions.json",
        input=tmp_path / "input.json",
        output=output_file,
    )
    functions = [object()]
    prompts = [object(), object()]
    pipeline_calls: dict[str, object] = {}

    class FakeResult:
        def model_dump(self, mode: str) -> dict[str, str]:
            assert mode == "json"
            return {"result": "ok"}

    class FakePipeline:
        def __init__(self, functions: list[object], model: object) -> None:
            pipeline_calls["functions"] = functions
            pipeline_calls["model"] = model

        def run(self, input_prompts: list[object]) -> list[FakeResult]:
            pipeline_calls["prompts"] = input_prompts
            return [FakeResult()]

    fake_model = object()
    monkeypatch.setattr(main_module, "parse_args", lambda: args)
    monkeypatch.setattr(main_module, "validate_paths", lambda received: None)
    monkeypatch.setattr(
        main_module, "load_functions_definition", lambda path: functions
    )
    monkeypatch.setattr(
        main_module, "load_input_prompts", lambda path: prompts
    )
    monkeypatch.setattr(main_module, "Small_LLM_Model", lambda: fake_model)
    monkeypatch.setattr(main_module, "FunctionCallingPipeline", FakePipeline)

    main_module.main()

    assert pipeline_calls == {
        "functions": functions,
        "model": fake_model,
        "prompts": prompts,
    }
    assert json.loads(output_file.read_text(encoding="utf-8")) == [
        {"result": "ok"}
    ]
