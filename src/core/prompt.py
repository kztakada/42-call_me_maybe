"""プロンプト構築モジュール

関数定義データおよびユーザー入力プロンプトから、
指定された LLM フォーマット（デフォルト: ChatML）に準拠したプロンプトを構築します
"""

import json
from typing import Literal

from src.core.parser import FunctionDefinition

# サポートするプロンプトフォーマットの型定義
PromptFormat = Literal["chatml"]


class PromptBuilder:
    """関数定義とユーザー入力から LLM 用プロンプトを構築・保持するクラス

    Args:
        functions: 関数の定義モデルのリスト
        format_type: プロンプトのフォーマット形式（デフォルト: "chatml"）

    Raises:
        ValueError: サポートされていない format_type が指定された場合
    """

    def __init__(
        self,
        functions: list[FunctionDefinition],
        format_type: PromptFormat = "chatml",
    ) -> None:
        self.functions = functions
        self.format_type = format_type
        self.system_prompt = self._build_system_prompt()

    def _format_functions_json(self) -> str:
        """関数定義リストをインデント付き JSON 文字列に変換します

        Returns:
            整形された JSON スキーマ文字列
        """
        raw_list = [
            fn.model_dump(exclude_none=True)
            for fn in self.functions
        ]
        return json.dumps(raw_list, indent=2, ensure_ascii=False)

    def _build_system_prompt(self) -> str:
        """指定されたフォーマットに応じたシステムプロンプトを構築します

        Returns:
            構築されたシステムプロンプト文字列

        Raises:
            ValueError: サポートされていない format_type が指定された場合
        """
        functions_json = self._format_functions_json()

        if self.format_type == "chatml":
            instructions = (
                "You are a helpful assistant with access to the following "
                "functions:\n"
                f"{functions_json}\n\n"
                "To use a function, output a JSON object with 'name' and "
                "'parameters' keys.\n"
                "Do not include any text outside the JSON object."
            )
            return f"<|im_start|>system\n{instructions}<|im_end|>"

        # 今後ボーナス機能等で他フォーマット（例: llama3 等）を拡張する場合はここに追加
        raise ValueError(f"Unsupported prompt format: {self.format_type}")

    def build_prompt(self, user_prompt: str) -> str:
        """保持しているシステムプロンプトとユーザー入力から完全なプロンプトを生成します

        Args:
            user_prompt: ユーザーからの自然言語プロンプト

        Returns:
            LLM に投入可能な完全なプロンプト文字列

        Raises:
            ValueError: サポートされていない format_type が指定された場合
        """
        if self.format_type == "chatml":
            user_part = f"<|im_start|>user\n{user_prompt}<|im_end|>"
            assistant_part = "<|im_start|>assistant\n"
            return f"{self.system_prompt}\n{user_part}\n{assistant_part}"

        raise ValueError(f"Unsupported prompt format: {self.format_type}")


__all__ = ["PromptBuilder", "PromptFormat"]
