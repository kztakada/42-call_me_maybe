"""JSON データのパーサーモジュール

関数定義 JSON およびテストプロンプト JSON の読み込み、
検証、 および Pydantic モデルへの変換を行います
"""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from src.utils import exit_with_error

# Pydantic モデル制約付きのジェネリクス型変数の定義
T = TypeVar("T", bound=BaseModel)

# =====================================================================
# Pydantic データモデル定義
# =====================================================================


class ParameterProperty(BaseModel):
    """関数の各引数の型定義モデル"""

    type: str
    description: str | None = None

    model_config = ConfigDict(extra="ignore")


class ReturnProperty(BaseModel):
    """関数の戻り値の型定義モデル"""

    type: str
    description: str | None = None

    model_config = ConfigDict(extra="ignore")


class FunctionDefinition(BaseModel):
    """関数の型定義モデル"""

    name: str
    description: str
    parameters: dict[str, ParameterProperty] | None = None
    returns: ReturnProperty | None = None

    model_config = ConfigDict(extra="ignore")


class InputPrompt(BaseModel):
    """入力プロンプトの型定義モデル"""

    prompt: str

    model_config = ConfigDict(extra="ignore")

# =====================================================================
# 共通ジェネリクス JSON ロード＆バリデーション関数
# =====================================================================


def _load_json_array(
    file_path: Path,
    model_cls: type[T],
    resource_name: str,
) -> list[T]:
    """JSON 配列ファイルを読み込み、指定された Pydantic モデルで検証します

    Args:
        file_path: 読み込み対象の JSON ファイルパス
        model_cls: 各要素の変換先 Pydantic モデルクラス
        resource_name: エラーメッセージに表示するリソース識別名

    Returns:
        検証済みの Pydantic モデルインスタンスのリスト
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data: Any = json.load(f)

        if not isinstance(raw_data, list):
            exit_with_error(
                f"{resource_name} file must contain a JSON array: "
                f"{file_path}"
            )

        results: list[T] = []
        for item in raw_data:
            if not isinstance(item, dict):
                exit_with_error(
                    f"Each {resource_name} item must be a JSON object: "
                    f"{file_path}"
                )
            results.append(model_cls(**item))

        return results

    except FileNotFoundError:
        exit_with_error(f"{resource_name} file not found: {file_path}")
    except json.JSONDecodeError as e:
        exit_with_error(f"Invalid JSON format in {file_path}: {e}")
    except ValidationError as e:
        exit_with_error(f"Schema validation failed for {file_path}:\n{e}")
    except OSError as e:
        exit_with_error(f"File I/O error on {file_path}: {e}")
    except PermissionError as e:
        exit_with_error(f"Permission denied for {file_path}: {e}")
    except UnicodeDecodeError as e:
        exit_with_error(f"Encoding error while reading {file_path}: {e}")
    except Exception as e:
        exit_with_error(
            f"An unexpected error occurred while processing {file_path}: {e}"
        )


# =====================================================================
# JSON ロード＆バリデーションのラッパー関数
# =====================================================================

def load_functions_definition(file_path: Path) -> list[FunctionDefinition]:
    """JSON ファイルから関数定義一覧を読み込み、バリデーションを行います

    Args:
        file_path: 関数スキーマを含む JSON ファイルのパス

    Returns:
        検証済みの関数定義モデルのリスト
    """
    return _load_json_array(
        file_path=file_path,
        model_cls=FunctionDefinition,
        resource_name="Functions definition",
    )


def load_input_prompts(file_path: Path) -> list[InputPrompt]:
    """JSON ファイルから入力プロンプト一覧を読み込み、バリデーションを行います

    Args:
        file_path: 入力プロンプトを含む JSON ファイルのパス

    Returns:
        検証済みの入力プロンプトモデルのリスト
    """
    return _load_json_array(
        file_path=file_path,
        model_cls=InputPrompt,
        resource_name="Input prompts",
    )


__all__ = [
    "FunctionDefinition",
    "InputPrompt",
    "ParameterProperty",
    "ReturnProperty",
    "load_functions_definition",
    "load_input_prompts",
]
