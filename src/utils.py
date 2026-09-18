"""ユーティリティモジュール

プロジェクト全体で使用するユーティリティ関数を提供します
"""

import sys
from typing import NoReturn


def exit_with_error(message: str, exit_code: int = 1) -> NoReturn:
    """エラーメッセージを標準エラー出力に表示し、プログラムを終了します

    Args:
        message: 表示するエラーメッセージ
        exit_code: プログラム終了コード (デフォルト: 1)
    """
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(exit_code)
