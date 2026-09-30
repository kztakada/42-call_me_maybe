"""推論パイプラインモジュール

プロンプト構築、LLM モデルによる推論、および制約付きデコーディングを統合し、
入力プロンプトから関数呼び出し JSON を生成するメインパイプラインを提供します
"""

from llm_sdk import Small_LLM_Model
from src.core.parser import (FunctionDefinition, InputPrompt,
                             FunctionCallOutput, parse_json_to_output)
from src.core.prompt import PromptBuilder
from src.core.decoder import BaseConstrainedDecoder


class FunctionCallingPipeline:
    """ファンクションコーディング推論パイプラインクラス

    Args:
        functions: 関数の定義モデルのリスト
        model: SDK の Small_LLM_Model インスタンス
    """

    def __init__(
        self,
        functions: list[FunctionDefinition],
        model: Small_LLM_Model,
    ) -> None:
        self.functions = functions
        self.model = model
        self.prompt_builder = PromptBuilder(functions)

    def run(self, input_prompts: list[InputPrompt]
            ) -> list[FunctionCallOutput]:
        """すべての入力プロンプトに対して推論を実行し、結果のリストを返します

        Args:
            input_prompts: 入力プロンプトモデルのリスト

        Returns:
            各プロンプトに対する FunctionCallOutput オブジェクトのリスト
        """
        results: list[FunctionCallOutput] = []
        for prompt_item in input_prompts:
            full_prompt = self.prompt_builder.build_prompt(
                prompt_item.prompt
            )
            output_json = self._generate_single(full_prompt)
            results.append(output_json.model_copy(
                update={"prompt": prompt_item.prompt}
            ))

        return results

    def _generate_single(self, full_prompt: str, max_new_tokens: int = 256
                         ) -> FunctionCallOutput:
        """1 つのプロンプトに対してデコーディングを実行します

        Args:
            full_prompt: システムプロンプトが含まれる ChatML プロンプト
            max_new_tokens: 生成する最大トークン数 (デフォルト: 256)

        Returns:
            生成された関数呼び出しの JSON 辞書
        """
        # 1. プロンプトを1次元のトークン ID リストに変換
        input_ids: list[int] = self.model.encode(
            full_prompt).squeeze(0).tolist()  # squeeze で batch 次元を除去
        prompt_length = len(input_ids)  # プロンプトの長さを保持

        # 2. プロンプト毎にデコーダー状態マシンを初期化
        decoder = BaseConstrainedDecoder(
            functions=self.functions,
            model=self.model,
        )

        # 3. 生成ループの駆動
        generated_count = 0
        while not decoder.is_finished() and generated_count < max_new_tokens:
            raw_logits = self.model.get_logits_from_input_ids(input_ids)
            next_token_id = decoder.step(raw_logits)
            input_ids.append(next_token_id)
            generated_count += 1

        # 4. 新規生成されたトークンのみを切り出してデコード
        generated_ids = input_ids[prompt_length:]
        generated_text = self.model.decode(generated_ids)

        # 5. 生成された JSON 文字列をパースして出力モデルに変換
        return parse_json_to_output(generated_text)


__all__ = ["FunctionCallingPipeline"]
