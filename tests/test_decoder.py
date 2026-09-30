"""src/core/decoder.pyのテストモジュール"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from llm_sdk import Small_LLM_Model
from src.core.decoder import BaseConstrainedDecoder, DecoderState
from src.core.parser import FunctionDefinition, ParameterProperty

# =====================================================================
# Fixtures (テスト用モックおよびダミーデータのセットアップ)
# =====================================================================


@pytest.fixture
def mock_functions() -> list[FunctionDefinition]:
    """関数定義のテスト用フィクスチャ"""
    return [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers.",
            parameters={
                "a": ParameterProperty(type="number"),
                "b": ParameterProperty(type="number"),
            },
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Greet a person.",
            parameters={
                "name": ParameterProperty(type="string"),
            },
        ),
    ]


@pytest.fixture
def dummy_vocab_file(tmp_path: Path) -> Path:
    """テスト用のダミー vocab.json ファイルを作成する"""
    vocab_data = {
        "{": 0,
        '"': 1,
        "name": 2,
        '": "': 3,
        "fn_add_numbers": 4,
        "fn_greet": 5,
        '", "parameters": {': 6,
        '"a"': 7,
        ': ': 8,
        "123": 9,
        ", ": 10,
        '"b"': 11,
        "456": 12,
        "}": 13,
        "}}": 14,
        "invalid_token": 15,
        "true": 16,
        "false": 17,
        "hello": 18,
    }
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(vocab_data), encoding="utf-8")
    return vocab_file


@pytest.fixture
def mock_model(dummy_vocab_file: Path) -> MagicMock:
    """Small_LLM_Model モックのテスト用フィクスチャ"""
    model = MagicMock(spec=Small_LLM_Model)
    model.get_path_to_vocab_file.return_value = str(dummy_vocab_file)

    # トークン ID を文字列へ変換する簡易デコード用マッピング
    id_map = {
        0: "{",
        1: '"',
        2: "name",
        3: '": "',
        4: "fn_add_numbers",
        5: "fn_greet",
        6: '", "parameters": {',
        7: '"a"',
        8: ": ",
        9: "123",
        10: ", ",
        11: '"b"',
        12: "456",
        13: "}",
        14: "}}",
        15: "invalid_token",
        16: "true",
        17: "false",
        18: "hello",
    }
    model.decode.side_effect = lambda ids: "".join(
        id_map.get(idx, "") for idx in ids
    )
    encode_map = {
        '{"name": "fn_add_numbers", "parameters": {': [0, 1, 2, 3, 4, 6],
        '{"name": "fn_greet", "parameters": {': [0, 1, 2, 3, 5, 6],
    }
    model.encode.side_effect = lambda text: np.array(
        [encode_map[text]], dtype=np.int64
    )
    return model


@pytest.fixture
def decoder(
    mock_functions: list[FunctionDefinition], mock_model: MagicMock
) -> BaseConstrainedDecoder:
    """初期化済み BaseConstrainedDecoder のテスト用フィクスチャ"""
    return BaseConstrainedDecoder(
        functions=mock_functions,
        model=mock_model,
    )

# =====================================================================
# 1. 正常系テスト（成功と状態遷移）
# =====================================================================


def test_decoder_initialization(decoder: BaseConstrainedDecoder) -> None:
    """Test decoder initial state and configuration."""
    assert not decoder.is_finished()
    assert decoder._state == DecoderState.START
    assert decoder._buffer == ""


def test_decoder_full_generation_flow(
    decoder: BaseConstrainedDecoder,
) -> None:
    """START から FINISHED までのデコード手順全体をテストする"""
    # トークン生成順序:
    # { -> "name": " -> fn_add_numbers -> ", "parameters": { ... -> }}
    token_sequence = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 8, 12, 13, 13]

    for token_id in token_sequence:
        assert not decoder.is_finished()
        # ダミーの Logit 配列 (該当の token_id の確率を高く設定)
        raw_logits = [0.0] * 20
        raw_logits[token_id] = 10.0

        selected_id = decoder.step(raw_logits)
        assert selected_id == token_id

    # 最終トークン生成後に FINISHED ステートになっていること
    assert decoder.is_finished()


def test_decoder_string_parameter_generation(
    decoder: BaseConstrainedDecoder,
) -> None:
    """文字列引数を含む関数呼び出しを最後まで生成できることをテストする"""
    token_sequence = [
        0, 1, 2, 3, 5, 6,
        1, 2, 1, 8, 1, 18, 1, 13, 13,
    ]

    for token_id in token_sequence:
        raw_logits = [0.0] * 20
        raw_logits[token_id] = 10.0
        assert decoder.step(raw_logits) == token_id

    assert decoder.is_finished()
    assert decoder._buffer == (
        '{"name": "fn_greet", "parameters": {"name": "hello"}}'
    )

# =====================================================================
# 2. Logit マスキング機能テスト
# =====================================================================


def test_apply_mask_start_state(decoder: BaseConstrainedDecoder) -> None:
    """START 状態で '{' トークンだけが許可される Logit マスキングをテストする"""
    raw_logits = [1.0] * 20
    masked = decoder._apply_mask(raw_logits)

    # トークン ID 0 ("{") は保持され、無関係なトークン（例: ID 15）は -inf になる
    assert masked[0] == 1.0
    assert masked[15] == -np.inf


def test_apply_mask_function_name_state(
    decoder: BaseConstrainedDecoder,
) -> None:
    """FUNCTION_NAME 状態での Logit マスキングをテストする"""
    # START から FUNCTION_NAME へ状態を移行
    decoder._buffer = '{"name": "'
    decoder._generated_token_ids = [0, 1, 2, 3]
    decoder._state = DecoderState.FUNCTION_NAME

    raw_logits = [1.0] * 20
    masked = decoder._apply_mask(raw_logits)

    # 許可された関数名（"fn_add_numbers": 4、"fn_greet": 5）は保持される
    assert masked[4] == 1.0
    assert masked[5] == 1.0
    # 無効なトークン（例: 15）は -inf にマスクされる
    assert masked[15] == -np.inf


def test_apply_mask_number_parameter_type(
    decoder: BaseConstrainedDecoder, mock_functions: list[FunctionDefinition]
) -> None:
    """number 型パラメータに対する Logit マスキングをテストする"""
    decoder._selected_function = mock_functions[0]  # fn_add_numbers
    decoder._state = DecoderState.PARAM_VALUE
    decoder._parameter_index = 0  # 引数 'a' (number 型)

    raw_logits = [1.0] * 20
    masked = decoder._apply_mask(raw_logits)

    # 数値トークン（ID 9: "123"）は許可され、非数値トークン（ID 15）はマスクされる
    assert masked[9] == 1.0
    assert masked[15] == -np.inf


def test_apply_mask_parameter_key_is_schema_constrained(
    decoder: BaseConstrainedDecoder, mock_functions: list[FunctionDefinition]
) -> None:
    """PARAMS_KEY 状態では現在の引数キーの次のトークンだけを許可する"""
    decoder._selected_function = mock_functions[0]
    decoder._state = DecoderState.PARAMS_KEY
    decoder._parameter_index = 0

    first_key_token = decoder._apply_mask([1.0] * 20)
    assert first_key_token[7] == 1.0
    assert first_key_token[15] == -np.inf

    decoder._phase_text = '"a"'
    colon_token = decoder._apply_mask([1.0] * 20)
    assert colon_token[8] == 1.0
    assert colon_token[7] == -np.inf


def test_apply_mask_boolean_parameter_type(
    decoder: BaseConstrainedDecoder, mock_functions: list[FunctionDefinition]
) -> None:
    """boolean 型では true/false と適切な区切りだけを許可する"""
    function = mock_functions[0].model_copy(deep=True)
    assert function.parameters is not None
    function.parameters["a"] = ParameterProperty(type="boolean")
    decoder._selected_function = function
    decoder._state = DecoderState.PARAM_VALUE
    decoder._parameter_index = 0

    masked = decoder._apply_mask([1.0] * 20)
    assert masked[16] == 1.0
    assert masked[17] == 1.0
    assert masked[9] == -np.inf


def test_apply_mask_string_parameter_requires_json_string(
    decoder: BaseConstrainedDecoder, mock_functions: list[FunctionDefinition]
) -> None:
    """string 型では引用符から始まり、文字列途中に区切りを許可しない"""
    function = mock_functions[0].model_copy(deep=True)
    assert function.parameters is not None
    function.parameters["a"] = ParameterProperty(type="string")
    decoder._selected_function = function
    decoder._state = DecoderState.PARAM_VALUE
    decoder._parameter_index = 0

    start = decoder._apply_mask([1.0] * 20)
    assert start[1] == 1.0
    assert start[18] == -np.inf

    decoder._value_text = '"'
    content = decoder._apply_mask([1.0] * 20)
    assert content[18] == 1.0
    assert content[10] == -np.inf

# =====================================================================
# 3. 異常系・エッジケーステスト（異常系とフォールバック）
# =====================================================================


def test_apply_mask_fallback_when_no_valid_tokens(
    decoder: BaseConstrainedDecoder,
) -> None:
    """有効なトークン候補が空の場合のフォールバック処理をテストする"""
    # 有効なトークンが存在しない状態を意図的に再現する
    decoder._state = DecoderState.PARAM_VALUE
    decoder._selected_function = None

    raw_logits = [2.0] * 20
    # クラッシュ（ValueError）せずにフォールバック値が返る
    masked = decoder._apply_mask(raw_logits)
    assert masked == raw_logits


def test_apply_mask_end_json_only_accepts_closing_brace(
    decoder: BaseConstrainedDecoder,
) -> None:
    """END_JSON 状態では閉じ波括弧だけを候補にする"""
    decoder._state = DecoderState.END_JSON

    masked = decoder._apply_mask([1.0] * 20)

    assert masked[13] == 1.0
    assert masked[14] == -np.inf
    assert masked[15] == -np.inf


def test_empty_parameter_function_can_finish(
    mock_model: MagicMock,
) -> None:
    """引数のない関数は parameters の空オブジェクトを閉じて完了できる"""
    function = FunctionDefinition(
        name="fn_add_numbers",
        description="No parameters.",
        parameters={},
    )
    mock_model.encode.side_effect = lambda text: np.array(
        [[0, 1, 2, 3, 4, 6]], dtype=np.int64
    )
    decoder = BaseConstrainedDecoder(functions=[function], model=mock_model)

    for token_id in [0, 1, 2, 3, 4, 6, 13, 13]:
        raw_logits = [0.0] * 20
        raw_logits[token_id] = 10.0
        assert decoder.step(raw_logits) == token_id

    assert decoder.is_finished()


def test_decoder_step_with_empty_logits_raises_error(
    decoder: BaseConstrainedDecoder,
) -> None:
    """空の logits リストで step を実行した場合の動作をテストする"""
    with pytest.raises((ValueError, IndexError)):
        decoder.step([])
