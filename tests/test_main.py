"""__main__.pyのテスト"""

import pytest
import src.__main__ as main_module


def test_main() -> None:
    """関数が正常終了するか、コマンドライン引数や全体のパース〜結果出力の一連の流れ
    （結合テスト / E2E テスト）が正常に行われるかを検証します
    """
    # ここでは、実際のコマンドライン引数を模倣して、main 関数を呼び出すことができます。
    # 例えば、sys.argv をモックするか、pytest の capsys を使用して標準出力をキャプチャすることができます。
    # ただし、ここでは簡単な例として、main 関数が例外を投げずに終了することだけを確認します。

    try:
        # main() 関数を呼び出す（必要に応じて引数を渡す）
        main_module.main()
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")
