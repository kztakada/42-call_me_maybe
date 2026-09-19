"""src/core/prompt.py の PromptBuilder クラスのテスト"""

import pytest

from src.core.parser import (
    FunctionDefinition,
    ParameterProperty,
    ReturnProperty,
)
from src.core.prompt import PromptBuilder

# =====================================================================
# 1. 正常系テスト (Success Cases)
# =====================================================================


def test_prompt_builder_init_success() -> None:
    """有効な関数定義で PromptBuilder を初期化できることを確認する"""
    functions = [
        FunctionDefinition(
            name="fn_add",
            description="Add two numbers.",
            parameters={
                "a": ParameterProperty(type="number", description="First"),
                "b": ParameterProperty(type="number"),
            },
            returns=ReturnProperty(type="number"),
        )
    ]

    builder = PromptBuilder(functions, format_type="chatml")

    assert builder.system_prompt.startswith("<|im_start|>system\n")
    assert builder.system_prompt.endswith("<|im_end|>")
    assert '"name": "fn_add"' in builder.system_prompt
    assert '"description": "Add two numbers."' in builder.system_prompt
    assert '"type": "number"' in builder.system_prompt


def test_prompt_builder_empty_functions() -> None:
    """空の関数リストで PromptBuilder を初期化できることを確認する"""
    builder = PromptBuilder([], format_type="chatml")

    assert "<|im_start|>system\n" in builder.system_prompt
    assert "[\n]" in builder.system_prompt or "[]" in builder.system_prompt


def test_prompt_builder_build_prompt_success() -> None:
    """ユーザー入力から ChatML の完全なプロンプトを生成できることを確認する"""
    functions = [
        FunctionDefinition(
            name="fn_get_weather",
            description="Get current weather.",
            parameters={},
        )
    ]
    builder = PromptBuilder(functions)
    user_prompt = "What's the weather in Tokyo?"

    full_prompt = builder.build_prompt(user_prompt)

    expected_user = (
        "<|im_start|>user\nWhat's the weather in Tokyo?<|im_end|>"
    )
    expected_assistant = "<|im_start|>assistant\n"

    assert builder.system_prompt in full_prompt
    assert expected_user in full_prompt
    assert full_prompt.endswith(expected_assistant)


def test_prompt_builder_multiple_build_prompt_calls() -> None:
    """複数回 build_prompt を呼び出しても各ユーザー入力が正しく保持されることを確認する"""
    functions = [
        FunctionDefinition(
            name="fn_calc",
            description="Calculate expression.",
            parameters={},
        )
    ]
    builder = PromptBuilder(functions)

    prompt1 = builder.build_prompt("Calculate 1 + 1")
    prompt2 = builder.build_prompt("Calculate 5 * 10")

    assert "Calculate 1 + 1" in prompt1
    assert "Calculate 5 * 10" in prompt2
    assert "Calculate 5 * 10" not in prompt1

# =====================================================================
# 2. 異常系・エッジケーステスト (Abnormal & Edge Cases)
# =====================================================================


def test_prompt_builder_unsupported_format_init() -> None:
    """未対応の format_type を指定した場合に初期化が失敗することを確認する"""
    functions: list[FunctionDefinition] = []

    with pytest.raises(ValueError, match="Unsupported prompt format"):
        PromptBuilder(
            functions,
            format_type="invalid_format",  # type: ignore[arg-type]
        )


def test_prompt_builder_unsupported_format_build_prompt() -> None:
    """実行時に format_type が未対応の値になった場合に build_prompt が失敗することを確認する"""
    functions: list[FunctionDefinition] = []
    builder = PromptBuilder(functions, format_type="chatml")

    # 手動で format_type を非サポート形式に変更した場合のハンドリングテスト
    builder.format_type = "unsupported_runtime"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unsupported prompt format"):
        builder.build_prompt("Hello")
