"""
pytestで全テストファイルから共通して利用するフィクスチャ（テスト用データや初期化処理）を定義する
tests/内では明示的にimportしなくても利用可能になる
"""

from typing import Any, Dict, List
import pytest


@pytest.fixture
def sample_function_definition() -> Dict[str, Any]:
    """テスト用の単一の関数定義データを返します。

    Returns:
        Dict[str, Any]: ダミーの関数定義データ。
    """
    return {
        "name": "get_current_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. Seattle, WA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    }


@pytest.fixture
def sample_test_prompts() -> List[str]:
    """テスト用のプロンプト文字列リストを返します。

    Returns:
        List[str]: プロンプトのリスト。
    """
    return [
        "What's the weather like in Tokyo?",
        "Calculate the 10th Fibonacci number.",
    ]
