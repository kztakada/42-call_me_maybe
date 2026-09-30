"""pipeline.py のテストモジュール"""

from typing import cast
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from llm_sdk import Small_LLM_Model

from src.core.parser import (
    FunctionCallOutput,
    FunctionDefinition,
    InputPrompt,
)
from src.core.pipeline import FunctionCallingPipeline


def test_run_includes_input_prompts_in_outputs() -> None:
    """run が各入力プロンプトを対応する出力へ設定することをテストする"""
    model = cast(Small_LLM_Model, Mock())
    pipeline = FunctionCallingPipeline(functions=[], model=model)
    input_prompts = [
        InputPrompt(prompt="Calculate 1 + 1."),
        InputPrompt(prompt="What is the weather in Tokyo?"),
    ]
    generated_outputs = [
        FunctionCallOutput(prompt="", name="fn_add", parameters={"a": 1}),
        FunctionCallOutput(
            prompt="", name="fn_weather", parameters={"city": "Tokyo"}
        ),
    ]

    with patch.object(
        pipeline,
        "_generate_single",
        side_effect=generated_outputs,
    ):
        results = pipeline.run(input_prompts)

    assert [result.prompt for result in results] == [
        "Calculate 1 + 1.",
        "What is the weather in Tokyo?",
    ]
    assert results[0].name == "fn_add"
    assert results[1].name == "fn_weather"


def test_run_with_no_input_prompts_returns_empty_list() -> None:
    """入力プロンプトが空の場合、空の結果リストを返すことをテストする"""
    mock_model = Mock(spec=Small_LLM_Model)
    model = cast(Small_LLM_Model, mock_model)
    pipeline = FunctionCallingPipeline(functions=[], model=model)

    results = pipeline.run([])

    assert results == []
    mock_model.encode.assert_not_called()


def test_init_with_invalid_function_definition_raises_type_error() -> None:
    """不正な関数定義で初期化した場合に TypeError が発生することをテストする"""
    model = cast(Small_LLM_Model, Mock())

    with pytest.raises(TypeError, match="Each function definition"):
        FunctionCallingPipeline(
            functions=cast(
                list[FunctionDefinition],
                ["invalid function"],
            ),
            model=model,
        )


def test_generate_single_returns_parsed_output() -> None:
    """デコーダー完了後に生成 JSON を解析して返すことをテストする"""
    model = MagicMock(spec=Small_LLM_Model)
    model.encode.return_value = np.array([[10, 11]], dtype=np.int64)
    model.get_logits_from_input_ids.return_value = [0.1, 0.9]
    model.decode.return_value = (
        '{"name": "fn_add", "parameters": {"a": 2}}'
    )
    decoder = MagicMock()
    decoder.is_finished.side_effect = [False, True]
    decoder.step.return_value = 12
    pipeline = FunctionCallingPipeline(functions=[], model=model)

    with patch(
        "src.core.pipeline.BaseConstrainedDecoder",
        return_value=decoder,
    ) as decoder_class:
        result = pipeline._generate_single("full prompt")

    assert result == FunctionCallOutput(
        prompt="", name="fn_add", parameters={"a": 2}
    )
    model.encode.assert_called_once_with("full prompt")
    assert model.get_logits_from_input_ids.call_count == 1
    model.decode.assert_called_once_with([12])
    decoder_class.assert_called_once_with(functions=[], model=model)
    decoder.step.assert_called_once_with([0.1, 0.9])


def test_generate_single_stops_at_max_new_tokens() -> None:
    """デコーダー未完了でも生成上限に達したら停止することをテストする"""
    model = MagicMock(spec=Small_LLM_Model)
    model.encode.return_value = np.array([[10]], dtype=np.int64)
    model.get_logits_from_input_ids.return_value = [1.0]
    model.decode.return_value = (
        '{"name": "fn_add", "parameters": {}}'
    )
    decoder = MagicMock()
    decoder.is_finished.return_value = False
    decoder.step.side_effect = [20, 21]
    pipeline = FunctionCallingPipeline(functions=[], model=model)

    with patch(
        "src.core.pipeline.BaseConstrainedDecoder",
        return_value=decoder,
    ):
        result = pipeline._generate_single("full prompt", max_new_tokens=2)

    assert result.name == "fn_add"
    assert decoder.step.call_count == 2
    assert model.get_logits_from_input_ids.call_count == 2
    model.decode.assert_called_once_with([20, 21])


def test_generate_single_invalid_output_raises_value_error() -> None:
    """生成 JSON が不正な場合に ValueError を返すことをテストする"""
    model = MagicMock(spec=Small_LLM_Model)
    model.encode.return_value = np.array([[10]], dtype=np.int64)
    model.decode.return_value = "not json"
    decoder = MagicMock()
    decoder.is_finished.side_effect = [False, True]
    decoder.step.return_value = 20
    pipeline = FunctionCallingPipeline(functions=[], model=model)

    with patch(
        "src.core.pipeline.BaseConstrainedDecoder",
        return_value=decoder,
    ):
        with pytest.raises(ValueError, match="Failed to parse generated text"):
            pipeline._generate_single("full prompt")


def test_generate_single_propagates_model_error() -> None:
    """モデルのエラーをパイプラインが握りつぶさず伝播することをテストする"""
    model = MagicMock(spec=Small_LLM_Model)
    model.encode.side_effect = RuntimeError("model failure")
    pipeline = FunctionCallingPipeline(functions=[], model=model)

    with pytest.raises(RuntimeError, match="model failure"):
        pipeline._generate_single("full prompt")


def test_generate_single_propagates_decoder_error() -> None:
    """デコーダーのエラーをパイプラインが伝播することをテストする"""
    model = MagicMock(spec=Small_LLM_Model)
    model.encode.return_value = np.array([[10]], dtype=np.int64)
    decoder = MagicMock()
    decoder.is_finished.return_value = False
    decoder.step.side_effect = RuntimeError("decoder failure")
    pipeline = FunctionCallingPipeline(functions=[], model=model)

    with patch(
        "src.core.pipeline.BaseConstrainedDecoder",
        return_value=decoder,
    ):
        with pytest.raises(RuntimeError, match="decoder failure"):
            pipeline._generate_single("full prompt")
