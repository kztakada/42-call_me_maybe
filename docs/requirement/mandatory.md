# 必須要件仕様書 (Mandatory Requirements Specification)

> **プロジェクト名**: call me maybe (Introduction to Function Calling in LLMs)  
> **対象プロジェクト**: 必須要件パート (Mandatory Part)  
> **根拠文書**: `call_me_maybe.pdf` (Version 1.5) & `call-me-maybe-dev-rules.md`  

---

## 1. 概要と目的

本課題の目的は、小型LLM（**Qwen/Qwen3-0.6B**）を使用して、ユーザーの自然言語プロンプトを機械実行可能な構造化された JSON 関数呼び出しに変換するツールを構築することです [1, 15]。

プロンプトエンジニアリングのみに頼るのではなく、生成トークンごとに確率分布（Logit）を直接制御する**制約デコーディング（Constrained Decoding）**を実装し、100% 有効でスキーマに完全に準拠した JSON を出力させます [8, 15, 22, 23]。

---

## 2. 技術スタックおよび制約事項

### 2.1 言語および環境
* **Python バージョン**: Python 3.10 以降 [9]。
* **パッケージマネージャー**: `uv` を使用（依存関係の導入は `uv sync`） [10, 13]。
* **仮想環境**: すべてのスクリプト実行は `uv run` 経由で行うこと [13, 14]。

### 2.2 許可および禁止ライブラリ
* **許可されるライブラリ** [12]:
  * `numpy`: 数値計算、ロジットベクトル操作、マスク処理用。
  * `pydantic`: すべてのクラスおよびデータ構造のバリデーションに必須 [12]。
  * `json`: 標準ライブラリ（パース・シリアライズ用）。
  * Python 標準ライブラリ (`typing`, `argparse`, `sys`, `pathlib`, `logging` など)。
* **厳格に禁止されるライブラリ** [12]:
  * **高レベル AI フレームワーク**: `dspy`, `outlines`, `lmformatenforcer`, `guidance` など [12]。
  * **直接的な ML フレームワークの呼び出し**: `pytorch` (`torch`), `transformers`, `huggingface_hub`, `accelerate`, `vllm` など [12]。

### 2.3 LLM SDK (`llm_sdk`) 利用規則
* LLM との対話は、提供されている `llm_sdk.Small_LLM_Model` クラスのみを介して行います [12, 18]。
* **使用可能な SDK メソッド** [18, 19]:
  * `get_logits_from_input_ids(input_ids: List[int]) -> List[float]`
  * `get_path_to_vocab_file() -> str`
  * `encode(text: str) -> Tensor`
  * `decode(token_ids: List[int]) -> str`（オプション）
* **禁止事項**: `llm_sdk` 内のプライベートメソッドやプライベート属性（アンダースコア `_` で始まるもの）へのアクセスは一切禁止されています [13]。

---

## 3. 入力・出力仕様

### 3.1 入力ファイル仕様
入力ファイルはデフォルトで `data/input/` ディレクトリに配置されます [14, 16]。

1. **`functions_definition.json`** [17]:
   * 利用可能な関数定義の JSON 配列。
   * 各要素に含まれるフィールド:
     * `name` (string): 関数名 (例: `fn_add_numbers`) [17]。
     * `description` (string): 関数の説明 [17]。
     * `parameters` (object): 引数名とその型 (`type`: `number`, `string`, `boolean` 等) [17, 25]。
     * `returns` (object): 戻り値の型 [17]。
2. **`function_calling_tests.json`** [16]:
   * テストプロンプトの JSON 配列 [16]。
   * 各要素: `{"prompt": "What is the sum of 2 and 3?"}` [16]。

> **注意**: 入力ファイルが存在しない、または不完全・破損した JSON である場合に備え、適切なエラー処理を実装しなければなりません [17, 26]。

### 3.2 出力ファイル仕様
出力は `data/output/function_calling_results.json` に書き出します [24]。

* **JSON 構造**: 各プロンプトに対する処理結果の JSON 配列 [24]。
* **必須キー**:
  * `prompt` (string): 元の自然言語プロンプト [24]。
  * `name` (string): 選択された関数名 [24]。
  * `parameters` (object): 型が一致する引数オブジェクト [24, 25]。
* **出力検証ルール (Validation Rules)** [25]:
  * 100% 有効でパース可能な JSON（末尾カンマやコメントは不可） [25]。
  * `functions_definition.json` に定義されたスキーマ（キー名・データ型）と完全一致 [25]。
  * 配列の外側に解説テキストやプロズ（prose）を一切出力しないこと [25]。

---

## 4. 制約デコーディング (Constrained Decoding) 要件

単にプロンプトで「JSON形式で出力してください」と頼む手法は禁止されています [23]。以下のアルゴリズムに基づき、トークン生成ごとに直接ロジットを制御します [21, 22]:

1. **Logit 取得**: 現在の `input_ids` を `get_logits_from_input_ids()` に渡し、全語彙の Logit ベクトルを取得 [18, 22]。
2. **語彙マッピング**: `get_path_to_vocab_file()` から取得した `vocab.json` を参照し、各トークン ID に対応する文字列を把握 [18, 23]。
3. **状態・スキーマ判定**: 現在の生成文字列プレフィックスに対して、次にどのトークンを追加しても「有効な JSON 構文」かつ「期待されるスキーマ（関数名・引数型）」を維持できるか判定 [22]。
4. **マスク適用**: 構文またはスキーマに違反するトークンの Logit を $-\infty$ (`-np.inf` や `-1e9`) に書き換える [22]。
5. **トークン選択**: マスク後の有効なトークンから最大確率のもの（`argmax`）を選択し、生成を更新 [22]。

---

## 5. 品質基準・自動化・ドキュメント要件

### 5.1 精度およびパフォーマンス要件
* **高精度**: 正しい関数選択および引数抽出において 90% 以上の精度を達成すること [26]。
* **100% JSON 有効性**: 生成されるすべての出力がエラーなくパース可能であること [22, 26]。
* **実行速度**: 標準的なハードウェア上で、すべてのテストプロンプトを 5 分未満で処理すること [26]。
* **堅牢性**: 予期せぬクラッシュを起こさず、未処理の例外で終了しないこと [9, 13, 26]。

### 5.2 コーディング規格
* **PEP 8**: `flake8 src` によるチェックをパスすること [9, 10]。
* **型ヒント & Mypy**: 全ての関数・変数に型アノテーションを記述し、`mypy` でエラーが出ないこと [9, 10]。
* **Docstring**: PEP 257（Google スタイル推奨）に準拠した docstring を記述すること [9]。
* **Pydantic**: 全てのクラス・モデルで `pydantic.BaseModel` を使用すること [12]。

### 5.3 Makefile 要件
プロジェクトルートに `Makefile` を配置し、以下のターゲットを実装すること（インデントはタブ文字） [10]:
* `install`: `uv sync` による依存関係インストール [10]。
* `run`: `uv run python -m src` によるメイン処理実行 [10, 14]。
* `debug`: `uv run python -m pdb -m src` によるデバッグ実行 [10]。
* `clean`: 一時キャッシュ（`__pycache__`, `.mypy_cache` 等）の削除 [10]。
* `lint`: `flake8 src` および指定されたフラグでの `mypy src` の実行 [10]。
* `lint-strict`: `flake8 src` および `mypy src --strict` の実行 [10]。

### 5.4 README.md 要件
* **言語**: 英語で記述すること [30]。
* **冒頭行**: イタリック体で `This project has been created as part of the 42 curriculum by <login>.` と記述すること [28]。
* **必須セクション**:
  * Description / Instructions / Resources (AI利用方法の明記を含む) [28]。
  * Algorithm explanation（制約デコーディングのアルゴリズム解説） [29]。
  * Design decisions / Performance analysis / Challenges faced / Testing strategy / Example usage [29]。
