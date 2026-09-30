"""制約付きデコーダーモジュール

LLM から取得した Logits に対して JSON 構文およびスキーマ制約を適用し、
100% 正しい JSON 形式の関数呼び出しを生成するデコーダーを提供します
"""

import json
import re
from enum import Enum, auto
from typing import cast

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model
from src.core.parser import FunctionDefinition


class DecoderState(Enum):
    """JSON 構造およびスキーマ生成の状態を表す列挙型"""

    START = auto()
    FUNCTION_NAME = auto()
    PARAMS_KEY = auto()
    PARAM_VALUE = auto()
    END_JSON = auto()
    FINISHED = auto()


class BaseConstrainedDecoder(BaseModel):
    """制約付きデコーディングを行うステートマシンデコーダーの基底クラス

    デコーダーは、LLM が次に出力できるトークンを現在の状態と関数スキーマから絞り込みます
    許可されたトークンの中から最大 Logit のものだけを生成し、
    生成後のテキストを使って次の状態へ遷移する、という処理を繰り返します

    Attributes:
        model_config (ConfigDict): Pydantic のモデル設定
        functions (list[FunctionDefinition]): 利用可能な関数定義のリスト
        model (Small_LLM_Model): SDK の Small_LLM_Model インスタンス
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    functions: list[FunctionDefinition]
    model: Small_LLM_Model

    # Pydantic 内部で管理するプライベート属性
    # 生成途中の JSON と、現在どの関数・引数を処理しているかを保持します
    _state: DecoderState = PrivateAttr(default=DecoderState.START)
    _buffer: str = PrivateAttr(default="")
    _is_finished: bool = PrivateAttr(default=False)
    _selected_function: FunctionDefinition | None = PrivateAttr(default=None)
    _generated_token_ids: list[int] = PrivateAttr(default_factory=list)
    _function_name_token_sequences: list[list[int]] = PrivateAttr(
        default_factory=list
    )
    _id_to_decoded: dict[int, str] = PrivateAttr(default_factory=dict)
    _number_token_ids: set[int] = PrivateAttr(default_factory=set)
    _boolean_token_ids: set[int] = PrivateAttr(default_factory=set)
    _string_token_ids: set[int] = PrivateAttr(default_factory=set)
    _parameter_index: int = PrivateAttr(default=0)
    _parameter_phase: str = PrivateAttr(default="key")
    _phase_text: str = PrivateAttr(default="")
    _value_text: str = PrivateAttr(default="")

    def model_post_init(self, __context: object) -> None:
        """モデル初期化後にボキャブラリファイルをロードして、全件キャッシュします"""

        # 語彙全体を一度だけデコードしておくことで、各 step でモデルを呼び出さずに
        # トークン文字列を参照できるようにします
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as vocab_file:
            vocab: dict[str, int] = json.load(vocab_file)
        self._id_to_decoded = {
            token_id: self.model.decode([token_id])
            for token_id in vocab.values()
        }
        self._cache_value_token_ids()

        # 関数ごとの完全な JSON プレフィックスをトークン列として保存します
        # FUNCTION_NAME 状態では、この列のうち生成済み部分に続くトークンだけを許可します
        prefix = '{"name": "'
        suffix = '", "parameters": {'
        self._function_name_token_sequences = [
            self.model.encode(prefix + fn.name + suffix).squeeze(0).tolist()
            for fn in self.functions
        ]

    def _cache_value_token_ids(self) -> None:
        """値の型ごとに、文法上の候補になり得るトークンIDをキャッシュします"""
        delimiters = {",", ", ", "}"}
        for token_id, token_text in self._id_to_decoded.items():
            if not token_text:
                continue

            # ここでは候補を粗く分類し、最終的な可否は
            # _value_token_is_allowed で現在の値との組み合わせから判定します
            if token_text in delimiters or all(
                char in "-0123456789." for char in token_text
            ):
                self._number_token_ids.add(token_id)
            if token_text in delimiters or all(
                char in "truefals" for char in token_text
            ):
                self._boolean_token_ids.add(token_id)
            if not any(ord(char) < 0x20 for char in token_text):
                self._string_token_ids.add(token_id)

    def is_finished(self) -> bool:
        """デコーディング処理が完了したかを返します

        Returns:
            完了している場合は True、それ以外は False
        """
        return self._is_finished

    def step(self, raw_logits: list[float]) -> int:
        """生の Logits にマスクを適用し、最適なトークン ID を選択して状態を更新します

        Args:
            raw_logits: LLM から取得した未加工の Logits リスト

        Returns:
            選択された次のトークン ID
        """
        # 1. 現在の状態に合うトークンだけを残す
        masked_logits = self._apply_mask(raw_logits)

        # 2. 残った候補から最大 Logit のトークンを選ぶ (Greedy Decoding)
        selected_token_id = int(np.argmax(masked_logits))

        # 3. 生成文字列を蓄積して状態を進め、最後にトークン列を記録する
        token_text = self.model.decode([selected_token_id])
        self._buffer += token_text
        self._update_state(token_text)
        self._generated_token_ids.append(selected_token_id)

        return selected_token_id

    def _apply_mask(self, logits: list[float]) -> list[float]:
        """現在のステートとスキーマに基づき、不適切なトークンの Logit を -inf にマスクします

        Args:
            logits: LLM から取得した未加工の Logits リスト

        Returns:
            マスク処理適用後の Logits リスト
        """
        logits_array = np.array(logits, dtype=np.float32)
        valid_token_ids: set[int] = set()

        # 状態ごとに「次に置けるトークン」を求めます
        # 関数名と固定 JSON 部分は事前計算したプレフィックス、引数部分はスキーマと現在の値を使います
        if self._state in (DecoderState.START, DecoderState.FUNCTION_NAME):
            for target_ids in self._function_name_token_sequences:
                generated = self._generated_token_ids
                if target_ids[: len(generated)] == generated:
                    next_index = len(generated)
                    if next_index < len(target_ids):
                        valid_token_ids.add(target_ids[next_index])

        elif self._state == DecoderState.PARAMS_KEY:
            valid_token_ids = self._allowed_parameter_key_tokens()

        elif self._state == DecoderState.PARAM_VALUE:
            valid_token_ids = self._allowed_parameter_value_tokens()
        elif self._state == DecoderState.END_JSON:
            valid_token_ids = {
                token_id
                for token_id, token_text in self._id_to_decoded.items()
                if token_text.startswith("}")
                and not token_text[1:].strip()
            }

        # 語彙の分割方法によって候補を作れない場合があります
        # 文字列の途中だけは区切り記号を除外した緩やかなフォールバックを使い、生成を継続可能にします
        if not valid_token_ids:
            if (
                self._state == DecoderState.PARAM_VALUE
                and self._value_text.startswith('"')
                and not self._value_text.endswith('"')
            ):
                fallback = logits_array.copy()
                for token_id, token_text in self._id_to_decoded.items():
                    if token_text.startswith((",", "}")):
                        fallback[token_id] = -np.inf
                if np.isfinite(fallback).any():
                    return cast(list[float], fallback.tolist())
            return logits

        # 許可トークン以外を -inf にして、argmax の対象から外します
        mask = np.full_like(logits_array, fill_value=-np.inf)
        valid_indices = list(valid_token_ids)
        mask[valid_indices] = logits_array[valid_indices]

        return cast(list[float], mask.tolist())

    def _allowed_parameter_key_tokens(self) -> set[int]:
        """現在の引数に必要な JSON キーのトークンだけを返します"""
        selected_function = self._selected_function
        if selected_function is None:
            return set()

        parameters = selected_function.parameters or {}
        if self._parameter_index >= len(parameters):
            # すべての引数を処理したら parameters オブジェクトを閉じます
            return self._allowed_tokens_for_literal("}", self._phase_text)

        # スキーマに定義された順序で、現在のキーと ': ' の固定リテラルを生成します
        key = list(parameters)[self._parameter_index]
        literal = json.dumps(key) + ": "
        return self._allowed_tokens_for_literal(literal, self._phase_text)

    def _allowed_parameter_value_tokens(self) -> set[int]:
        """選択された引数型に適合する値のトークンだけを返します"""
        selected_function = self._selected_function
        if selected_function is None:
            return set()

        parameters = selected_function.parameters or {}
        key = list(parameters)[self._parameter_index]
        parameter_type = parameters[key].type
        # 型ごとの粗い候補集合を、現在の値の文法検査でさらに絞り込みます
        token_ids_by_type = {
            "number": self._number_token_ids,
            "boolean": self._boolean_token_ids,
            "string": self._string_token_ids,
        }
        candidate_token_ids = token_ids_by_type.get(parameter_type, set())
        allowed: set[int] = set()
        for token_id in candidate_token_ids:
            token_text = self._id_to_decoded[token_id]
            if not token_text:
                continue
            if self._value_token_is_allowed(
                token_text, parameter_type, self._value_text
            ):
                allowed.add(token_id)
        return allowed

    def _allowed_tokens_for_literal(
        self, literal: str, current: str
    ) -> set[int]:
        """リテラルの未生成部分に続けられるトークンを返します"""
        allowed: set[int] = set()
        for token_id, token_text in self._id_to_decoded.items():
            if not token_text:
                continue
            candidate = current + token_text
            if literal.startswith(candidate) or candidate == literal:
                allowed.add(token_id)
        return allowed

    def _value_token_is_allowed(
        self, token_text: str, parameter_type: str, current: str
    ) -> bool:
        """値トークンが型の文法を壊さないか判定します"""
        candidate = current + token_text
        selected_function = self._selected_function
        if selected_function is None:
            return False
        parameters = selected_function.parameters or {}
        is_last_parameter = self._parameter_index == len(parameters) - 1
        delimiters = {"}" if is_last_parameter else ","}
        delimiters.update({", "} if not is_last_parameter else set())
        if parameter_type == "number":
            # 数値は生成途中では空の整数部や小数部を許可し、区切り時に完成形を要求します
            if token_text in delimiters:
                return bool(re.fullmatch(r"-?(?:\d+)(?:\.\d+)?", current))
            return bool(re.fullmatch(r"-?\d*(?:\.\d*)?", candidate))

        if parameter_type == "boolean":
            # true / false のいずれかの接頭辞、または完成した値の後の区切りだけを許可します
            return (
                "true".startswith(candidate)
                or "false".startswith(candidate)
                or (
                    token_text in delimiters
                    and current in {"true", "false"}
                )
            )

        if parameter_type == "string":
            # 文字列は開始引用符から始め、JSON のエスケープ規則を守って閉じます
            if not current:
                return token_text == '"'
            if current.startswith('"') and (
                current == '"' or not current.endswith('"')
            ):
                if token_text.startswith((",", "}")):
                    return False
                if any(ord(char) < 0x20 for char in token_text):
                    return False
                string_end = self._scan_json_string(candidate)
                if string_end < 0:
                    return False
                if string_end == len(candidate):
                    return self._is_json_string(candidate) or (
                        self._is_json_string_prefix(candidate)
                    )
                return candidate[string_end:] in delimiters and (
                    self._is_json_string(candidate[:string_end])
                )
            return token_text in delimiters

        return False

    def _update_state(self, token_text: str) -> None:
        """新たに生成されたトークン文字列に応じて内部の構文ステートを更新します

        Args:
            token_text: 今回追加されたトークンのデコード済み文字列
        """
        # バッファ内の固定プレフィックスや値の終端が完成したタイミングで状態を遷移します
        if self._state == DecoderState.START and "{" in self._buffer:
            # JSON の開始後は、登録済み関数名のプレフィックスを照合します
            self._state = DecoderState.FUNCTION_NAME
        elif self._state == DecoderState.FUNCTION_NAME:
            for function in self.functions:
                prefix = (
                    '{"name": "' + function.name
                    + '", "parameters": {'
                )
                if self._buffer.startswith(prefix):
                    self._selected_function = function
                    self._state = DecoderState.PARAMS_KEY
                    self._phase_text = ""
                    break
        elif self._state == DecoderState.PARAMS_KEY:
            # キー部分はトークン境界をまたぐため、フェーズ専用バッファで照合します
            self._phase_text += token_text
            if self._selected_function is None:
                return
            parameters = self._selected_function.parameters or {}
            if self._parameter_index >= len(parameters):
                if self._phase_text.startswith("}"):
                    self._state = DecoderState.END_JSON
                    self._phase_text = ""
            else:
                key = list(parameters)[self._parameter_index]
                literal = json.dumps(key) + ": "
                if self._phase_text.startswith(literal):
                    self._state = DecoderState.PARAM_VALUE
                    self._phase_text = ""
                    self._value_text = ""
        elif self._state == DecoderState.PARAM_VALUE:
            # 値が区切り記号まで完成したら次の引数、または JSON 終了へ進みます
            self._value_text += token_text
            if self._value_is_complete(token_text):
                self._parameter_index += 1
                selected_function = self._selected_function
                if selected_function is None:
                    return
                parameters = selected_function.parameters or {}
                if self._parameter_index == len(parameters):
                    self._state = DecoderState.END_JSON
                else:
                    self._state = DecoderState.PARAMS_KEY
                self._phase_text = ""
                self._value_text = ""
        elif self._state == DecoderState.END_JSON:
            # parameters を閉じる '}' を受け取った時点で生成完了です
            self._phase_text += token_text
            if self._phase_text.startswith("}"):
                self._state = DecoderState.FINISHED
                self._is_finished = True

    def _value_is_complete(self, token_text: str) -> bool:
        """現在の引数値が閉じられたかを判定します"""
        selected_function = self._selected_function
        if selected_function is None:
            return False
        parameters = selected_function.parameters or {}
        is_last_parameter = self._parameter_index == len(parameters) - 1
        delimiters = {"}" if is_last_parameter else ","}
        delimiters.update({", "} if not is_last_parameter else set())
        if token_text in delimiters:
            # 数値・真偽値は区切り記号の直前まで、文字列は閉じ引用符までを値として検証します
            value = self._value_text[: -len(token_text)]
            if value in {"true", "false"} or bool(
                re.fullmatch(r"-?(?:\d+)(?:\.\d+)?", value)
            ):
                return True
            if self._is_json_string(value):
                return True

        string_end = self._scan_json_string(self._value_text)
        return (
            string_end > 0
            and self._value_text[string_end:] in delimiters
            and self._is_json_string(self._value_text[:string_end])
        )

    @staticmethod
    def _is_json_string(value: str) -> bool:
        """完全な JSON 文字列リテラルかを判定します"""
        if (
            len(value) < 2
            or not value.startswith('"')
            or not value.endswith('"')
        ):
            return False
        return BaseConstrainedDecoder._scan_json_string(value) == len(value)

    def _is_json_string_prefix(self, value: str) -> bool:
        """末尾の引用符を加えれば JSON 文字列になるかを判定します"""
        return value.startswith('"') and (
            self._scan_json_string(value) == len(value)
        )

    @staticmethod
    def _scan_json_string(value: str) -> int:
        """JSON文字列の有効な接頭辞の終端位置を返します"""
        if not value.startswith('"'):
            return -1

        index = 1
        while index < len(value):
            character = value[index]
            if character == '"':
                return index + 1
            if ord(character) < 0x20:
                return -1
            if character != "\\":
                index += 1
                continue

            # バックスラッシュの後は、JSON が定める単一文字エスケープか
            # 4 桁の Unicode エスケープだけを受け付けます
            index += 1
            if index >= len(value):
                return len(value)
            escaped = value[index]
            if escaped == "u":
                if index + 4 >= len(value):
                    return -1 if '"' in value[index + 1:] else len(value)
                if any(
                    digit not in "0123456789abcdefABCDEF"
                    for digit in value[index + 1:index + 5]
                ):
                    return -1
                index += 4
            elif escaped not in '"\\/bfnrt':
                return -1
            index += 1

        return len(value)


__all__ = ["DecoderState", "BaseConstrainedDecoder"]
