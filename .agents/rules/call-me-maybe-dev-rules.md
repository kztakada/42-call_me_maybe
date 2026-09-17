# 開発ルール & コーディング規約 ("call me maybe")

> **プロジェクト名**: call me maybe (LLMにおけるファンクションコーディング入門)  
> **対象**: AIエージェント (Claude, Codex, AntiGravity, Cursor, Windsurf 等) & 人間開発者  
> **準拠規格**: 42カリキュラム規格, PEP 8, PEP 257, 厳格な型指定 (Strict Typing)  
> **アーキテクチャ設計方針**: 必須機能 (Mandatory Part) とボーナス機能 (Bonus Part) の完全分離設計  

---

## 1. プロジェクト概要 & コアミッション

本プロジェクトは、小型LLM（**Qwen/Qwen3-0.6B**）を使用して、自然言語プロンプトを機械実行可能な構造化JSON関数呼び出しに変換する**ファンクションコーディングツール**を構築します。

* **核心要件**: システムは**制約デコーディング（Constrained Decoding）**を使用して、100%構文的に有効なJSONおよび厳格なスキーマ準拠を保証しなければなりません。
* **基本方針**: 必須機能（Mandatory）の 100% 確実な動作と安定性を最優先としつつ、ボーナス機能（Bonus）は拡張モジュールとして独立・カプセル化して実装します。

---

## 2. 技術スタック & 制約事項

### 2.1 言語 & 開発環境
* **Pythonバージョン**: Python 3.10 以降。
* **パッケージマネージャー**: `uv` の使用が必須（インストールおよび同期は `uv sync`）。
* **仮想環境**: すべてのコマンドは `uv run` を介して実行すること。

### 2.2 許可パッケージ & 禁止ライブラリ
* **許可される主要パッケージ**:
  * `numpy` (数値計算、ロジット操作、ベクトル計算用)
  * `pydantic` (すべてのデータモデルおよびバリデーションに必須)
  * `json` (パースおよびシリアライズ用標準ライブラリ)
  * Python標準モジュール (`typing`, `argparse`, `sys`, `pathlib`, `logging` 等)
* **禁止ライブラリ（厳格適用）**:
  * **高レベルAIフレームワーク禁止**: `dspy`, `outlines`, `lmformatenforcer`, `guidance` 等。
  * **メインコードでの直接的なMLフレームワーク使用禁止**: `pytorch` (`torch`), `transformers`, `huggingface_hub`, `accelerate`, `vllm`。
  * *注意*: LLMとのすべてのインタラクションは、提供された `llm_sdk.Small_LLM_Model` ラッパーを経由しなければなりません。

### 2.3 LLM SDK 利用規定
* LLMへのアクセスは、厳密に `llm_sdk.Small_LLM_Model` を経由すること:
  * `get_logits_from_input_ids(input_ids: List[int]) -> List[float]`
  * `get_path_to_vocab_file() -> str`
  * `encode(text: str) -> Tensor`
  * `decode(token_ids: List[int]) -> str`
* **禁止事項**: `llm_sdk` のプライベートメソッドや属性（`_` で始まるもの）に絶対アクセスしないこと。

---

## 3. 必須コーディング標準

### 3.1 型アノテーション & Mypy
* すべての関数引数、戻り値、クラス変数には、Pythonの `typing` モジュールを使用して明示的な型ヒントを記述すること。
* すべてのコードは、エラーなしで `mypy` の静的型チェックに合格すること。

### 3.2 PEP 8 & コードスタイル
* `flake8` を用いて **PEP 8** スタイルガイドラインを厳格に適用すること。
* Pythonコードのインデントには半角スペース 4 つを使用すること。

### 3.3 PEP 257 & Docstring ルール
* すべてのクラス、関数、メソッドには **PEP 257 準拠の docstring** を含めること。
* **推奨フォーマット**: **Google スタイル Docstring**。
  ```python
  def parse_functions(file_path: str) -> List[FunctionDefinition]:
      """JSONファイルから関数定義を読み込み、バリデーションを行います。

      Args:
          file_path: 関数スキーマを含むJSONファイルのパス。

      Returns:
          検証済みの関数定義モデルのリスト。

      Raises:
          FileNotFoundError: 指定されたファイルが存在しない場合。
          ValidationError: JSON構造がスキーマと一致しない場合。
      """
  ```
* **型ヒントの二重管理防止ルール**: シグネチャに型アノテーションが存在する場合、docstring内での型の重複記述（例: `file_path (str): 説明`）は避け、説明文のみ（`file_path: 説明`）を記述して冗長性を排除すること。

### 3.4 例外処理 & リソース管理
* 予期せぬクラッシュを防ぐため、関数は例外を適切に処理すること。評価時の未処理の例外は自動失格となります。
* すべてのリソース（ファイル、I/O）はコンテキストマネージャー（`with` 文）を使用して管理すること。

### 3.5 Pydantic によるデータ検証
* すべてのデータ構造、スキーマ、モデルは **`pydantic.BaseModel` を継承しなければならない**。

---

## 4. 自動化 & Makefile ルール

リポジトリルートには **`Makefile` を必ず含めること**。レシピのインデントにはタブ文字（Tab）を使用すること。

| ターゲット | 実行コマンド / 説明 |
| :--- | :--- |
| `install` | `uv sync` |
| `run` | `uv run python -m src` （必須機能のみをデフォルトで実行） |
| `run-bonus` | `uv run python -m src --bonus` （ボーナス機能を含めてデモ実行） |
| `test` | `uv run pytest -v` （単体・結合テストの全実行） |
| `debug` | `pdb` を使用したデバッグ実行（例: `uv run python -m pdb -m src`） |
| `clean` | `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `*.pyc` の削除 |
| `lint` | `uv run flake8 src` <br> `uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` |
| `lint-strict` | `uv run flake8 src` <br> `uv run mypy src --strict --ignore-missing-imports` <br> `uv run pydoclint src` |

---

## 5. ディレクトリ構造 & ボーナス分離アーキテクチャ

必須機能（`src/core/`）とボーナス機能（`src/bonus/`）を明確に分離し、エントリーポイント（`src/__main__.py`）で条件分岐する設計とします。

### 5.1 プロジェクト構成
```text
call_me_maybe/
├── src/
│   ├── __init__.py
│   ├── __main__.py          # エントリーポイント（CLI引数解析 & core/bonus切り替え）
│   ├── config.py            # 定数・設定管理
│   │
│   ├── core/                # 【必須機能】(Mandatory) — 100%の安定性を保証
│   │   ├── __init__.py
│   │   ├── parser.py        # 基本 Pydantic モデル & JSON ローダー
│   │   ├── decoder.py       # 基本 Constrained Decoder (BaseConstrainedDecoder)
│   │   └── pipeline.py      # 標準 LLM 生成パイプライン
│   │
│   └── bonus/               # 【ボーナス機能】(Bonus Part) — 独立拡張モジュール
│       ├── __init__.py
│       ├── custom_tokenizer.py # 自作・再実装トークナイザー
│       ├── visualizer.py    # デコーディング過程（Logitマスク/トークン選択）可視化
│       ├── optimizer.py     # キャッシング・バッチ処理最適化
│       ├── nested_decoder.py# ネスト構造・複雑スキーマ対応デコーダー
│       └── error_recovery.py# 高度なエラーハンドリング・リカバリ
│
├── llm_sdk/                 # 提供されたSDK（プライベート属性変更不可）
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/              # GITにコミットしないこと
│       └── function_calling_results.json
├── tests/                   # Pytest テストスイート
│   ├── __init__.py
│   ├── conftest.py          # pytestフィクスチャ定義
│   ├── test_parser.py       # Pydanticモデル単体テスト
│   ├── test_decoder.py      # 制約デコーダ単体テスト
│   └── test_bonus.py        # ボーナス機能拡張テスト
├── Makefile                 # 必須タスクランナー
├── pyproject.toml           # 依存関係 & ツール設定
├── uv.lock                  # ロックファイル
├── README.md                # ドキュメント (英語, 42ヘッダー付き, ボーナス機能マッピング表を含む)
└── .gitignore
```

### 5.2 コマンドラインインターフェース (CLI) 仕様
エントリーポイントは必須引数・オプショナル引数に加えて、ボーナス機能フラグを受け付けます。

```bash
uv run python -m src \
  [--functions_definition <functions_jsonへのパス>] \
  [--input <test_prompts_jsonへのパス>] \
  [--output <output_jsonへのパス>] \
  [--bonus] \
  [--visualize] \
  [--custom-tokenizer] \
  [--enable-nested]
```

* **デフォルト値**:
  * `--functions_definition`: `data/input/functions_definition.json`
  * `--input`: `data/input/function_calling_tests.json`
  * `--output`: `data/output/function_calling_results.json`
  * ボーナス関連フラグなしの場合は **`src/core/` のみ** を使用して安全に実行されます。

### 5.3 出力 JSON スキーマ
生成される出力ファイルは、各オブジェクトが**正確に**以下のキーを持つ JSON 配列でなければなりません:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

---

## 6. クラス設計 & オブジェクト指向戦略パターン (Strategy Pattern)

必須機能の基本クラス（`BaseConstrainedDecoder`）を継承してボーナス機能の拡張クラス（`NestedConstrainedDecoder` 等）を構築することで、コードの重複を防ぎ、レビュー時の切り替えを容易にします。

```python
# src/core/decoder.py (必須機能)
class BaseConstrainedDecoder:
    """基本の制約デコーディングクラス（単層オブジェクト・基本型の検証）。"""

    def mask_logits(self, logits: np.ndarray, current_prefix: str) -> np.ndarray:
        """無効なトークンのロジットを -inf にマスクします。"""
        ...


# src/bonus/nested_decoder.py (ボーナス機能)
from src.core.decoder import BaseConstrainedDecoder


class NestedConstrainedDecoder(BaseConstrainedDecoder):
    """【Bonus】ネストされた複雑なオブジェクト・配列に対応した拡張デコーダー。"""

    def mask_logits(self, logits: np.ndarray, current_prefix: str) -> np.ndarray:
        """ネスト構造を考慮した高度なロジットマスク処理。"""
        ...
```

---

## 7. レビュー対策 & AI エージェント開発ワークフロー

### 7.1 レビュアーへのアピール & ピアレビュー対策 (42 Defense)
1. **必須機能の完全動作**: フラグ未指定時に 100% 必須要件のみで動作し、失敗（Fail）リスクを排除する。
2. **`README.md` での透明性確保**: どのボーナス機能がどのファイルに実装され、どの CLI オプションで起動できるか明確な対応表を記述する。
3. **即時切り替え対応**: レビュアーから「ボーナスを外して必須だけで動かして」と言われた場合も、フラグを外すだけで即座にデモ可能とする。

### 7.2 AI エージェント開発チェックリスト
1. **事前確認**: `pyproject.toml` に `[tool.mypy]`, `[tool.pydoclint]`, `[tool.flake8]` の設定が含まれていることを確認する。
2. **実装手順**:
   * まず `src/core/` 配下に必須機能を実装・完成させる。
   * その後 `src/bonus/` 配下に拡張機能を独立して実装する。
   * すべての関数に完全な型ヒントと Google スタイルの docstring を記述する。
3. **検証ステップ**:
   * ターミナルで `make lint-strict` を実行し、`flake8`, `mypy`, `pydoclint` のエラーをすべてゼロにする。
   * `make test` (`uv run pytest`) を実行し、`tests/` 配下の全テストがパスすることを確認する。
   * コミット前に `make clean` を実行する。
