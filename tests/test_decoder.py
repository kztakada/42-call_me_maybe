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
    assert isinstance(masked, list)
    assert len(masked) == 20


def test_decoder_step_with_empty_logits_raises_error(
    decoder: BaseConstrainedDecoder,
) -> None:
    """空の logits リストで step を実行した場合の動作をテストする"""
    with pytest.raises((ValueError, IndexError)):
        decoder.step([])
