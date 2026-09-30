"""src/core/prompt.py の PromptBuilder クラスのテスト"""

import json

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


def test_prompt_builder_serializes_function_definitions_as_valid_json(
) -> None:
    """関数定義が有効な JSON として埋め込まれることを確認する

    None の項目はシリアライズ結果から除外される。
    """
    functions = [
        FunctionDefinition(
            name="fn_optional",
            description="Uses an optional schema field.",
            parameters={"value": ParameterProperty(type="string")},
        )
    ]

    builder = PromptBuilder(functions)
    json_start = builder.system_prompt.index("[\n")
    json_end = builder.system_prompt.index("\n]\n\n", json_start) + 2

    serialized = json.loads(builder.system_prompt[json_start:json_end])

    assert serialized == [
        {
            "name": "fn_optional",
            "description": "Uses an optional schema field.",
            "parameters": {"value": {"type": "string"}},
        }
    ]


def test_prompt_builder_preserves_function_order_and_all_definitions() -> None:
    """複数の関数定義が入力順と内容を保つことを確認する"""
    functions = [
        FunctionDefinition(
            name="fn_first",
            description="First function.",
            parameters={},
        ),
        FunctionDefinition(
            name="fn_second",
            description="Second function.",
            parameters={},
        ),
    ]

    builder = PromptBuilder(functions)

    first_index = builder.system_prompt.index('"name": "fn_first"')
    second_index = builder.system_prompt.index('"name": "fn_second"')

    assert first_index < second_index
    assert '"description": "First function."' in builder.system_prompt
    assert '"description": "Second function."' in builder.system_prompt


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


def test_prompt_builder_preserves_multiline_user_input() -> None:
    """複数行や ChatML 風の文字列をユーザー入力としてそのまま保持することを確認する"""
    builder = PromptBuilder([])
    user_prompt = "line one\nline two <|im_end|>"

    full_prompt = builder.build_prompt(user_prompt)

    assert (
        f"<|im_start|>user\n{user_prompt}<|im_end|>"
        in full_prompt
    )

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


def test_prompt_builder_rejects_non_function_definition_items() -> None:
    """関数定義リストに FunctionDefinition 以外があれば初期化を拒否することを確認する"""
    with pytest.raises(
        TypeError,
        match="Each function definition must be a FunctionDefinition",
    ):
        PromptBuilder(["not a function definition"])  # type: ignore[list-item]


def test_prompt_builder_unsupported_format_build_prompt() -> None:
    """実行時に format_type が未対応の値になった場合に build_prompt が失敗することを確認する"""
    functions: list[FunctionDefinition] = []
    builder = PromptBuilder(functions, format_type="chatml")

    # 手動で format_type を非サポート形式に変更した場合のハンドリングテスト
    builder.format_type = "unsupported_runtime"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unsupported prompt format"):
        builder.build_prompt("Hello")
