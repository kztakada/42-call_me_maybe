# ボーナス要件仕様書 (Bonus Requirements Specification)

> **プロジェクト名**: call me maybe (Introduction to Function Calling in LLMs)  
> **対象プロジェクト**: ボーナス要件パート (Bonus Part)  
> **根拠文書**: `call_me_maybe.pdf` (Chapter VII Bonus Part) & `call-me-maybe-dev-rules.md`  

---

## 1. 概要および前提条件

ボーナスパート（Bonus Part）は、必須要件（Mandatory Part）を完璧に達成した上で、システムをより高度かつ実用的に拡張するための追加機能群です [30, 31]。

### ⚠️ ボーナス評価の絶対条件
1. **必須パートの完全性**: 必須パートが 100% 正常に動作し、`make lint-strict` およびテストをクリアしている場合のみ、ボーナスパートの評価が行われます [31]。
2. **実動と実演可能性**: README.md に説明を書くだけでなく、実際のコードとして実装されており、ピアレビュー（Defense）時に実演・デモ可能でなければなりません [31]。

---

## 2. ボーナス機能一覧および詳細要件

仕様書（Chapter VII）で提示されているボーナス機能と、本プロジェクトでの実装要件は以下の通りです [30]：

| ボーナス機能項目 | 内容および実装要件 | 担当モジュール (推奨) |
| :--- | :--- | :--- |
| **1. 複数 LLM モデルのサポート** | `Qwen/Qwen3-0.6B` 以外の LLM モデルでも動作する抽象化レイヤーの実装 [30]。 | `src/bonus/model_adapter.py` |
| **2. トークナイザーの再構築 (Recoding)** | メインコードで `llm_sdk` の `encode` / `decode` を直接使わず、`get_logits_from_input_ids` と `get_path_to_vocab_file` から独自にトークナイザー処理を実装する [30]。 | `src/bonus/custom_tokenizer.py` |
| **3. 高度なエラーリカバリ機構** | 入力データの崩れ、不完全なプロンプト、未知のスキーマに対して柔軟にフォールバック・補正を行う処理 [30]。 | `src/bonus/error_recovery.py` |
| **4. パフォーマンス最適化** | プレフィックスキャッシング（KV Cache 相当の再利用概念）やバッチ処理による生成スピードの高速化 [30]。 | `src/bonus/optimizer.py` |
| **5. 包括的なテストスイート** | 単一ユニットテストにとどまらず、エッジケース・境界値・異常系・複合スキーマを網羅するテスト群の構築 [30]。 | `tests/bonus/` |
| **6. 生成プロセスの可視化 (Visualization)** | トークン生成の各ステップで、Logit マスク前後の確率分布や候補トークンをリアルタイム表示する CLI/UI ツール [30]。 | `src/bonus/visualizer.py` |
| **7. 複雑なネスト構造引数のサポート** | ネストされたオブジェクト（Object inside Object）や配列（Array of Objects）を持つ複合スキーマの制約デコーディング対応 [30]。 | `src/bonus/nested_decoder.py` |
| **8. 公開エンコード/デコード実装** | 自作トークナイザーの `encode()` および `decode()` メソッドを公開クラスとして提供 [30]。 | `src/bonus/custom_tokenizer.py` |
| **9. 制約デコーディングとの統合実証** | 独自トークナイズ処理と制約デコーディング（Logit Masking）がどのように統合されているかを可視化・実証するデモ [30]。 | `src/bonus/demo.py` |

---

## 3. アーキテクチャおよび設計方針

必須パート（Mandatory）のコードを汚さず、レビュー時に安全に切り替えられるよう、ボーナス機能は**完全モジュール分離**および**CLIフラグによる切り替え**を行うアプローチを採用します。

### 3.1 ディレクトリ構成
```text
src/
├── core/                     # 必須要件コード (Mandatory)
│   ├── parser.py
│   ├── decoder.py
│   └── pipeline.py
└── bonus/                    # ボーナス要件コード (Bonus)
    ├── __init__.py
    ├── custom_tokenizer.py   # 自作トークナイザー
    ├── nested_decoder.py     # ネスト構造対応デコーダー
    ├── visualizer.py         # 生成過程可視化ツール
    ├── optimizer.py          # キャッシュ・最適化
    └── error_recovery.py     # エラーリカバリ
```

### 3.2 CLI フラグによる制御仕様
エントリーポイント（`__main__.py`）は統一し、以下の CLI オプションによってボーナス機能を有効化します：

```bash
# 基本実行 (必須パートのみ動作)
uv run python -m src

# ボーナス機能有効化実行
uv run python -m src --bonus --visualize --custom-tokenizer
```

* `--bonus`: ボーナス用デコーダーおよび機能拡張を有効化。
* `--visualize`: トークンごとの Logit マスク状況をターミナルにステップ実行表示。
* `--custom-tokenizer`: 自作トークナイザー（`custom_tokenizer.py`）を使用。

---

## 4. Makefile コマンドと評価（Peer Review）手順

評価者が簡単にボーナス機能をテストできるよう、`Makefile` に専用ターゲットを準備します。

### Makefile 追加ターゲット
```makefile
# ボーナス機能のデモ実行
run-bonus:
	uv run python -m src --bonus --visualize

# ボーナス機能を含めた包括テストの実行
test-bonus:
	uv run pytest tests/bonus/ -v
```

### 評価（Defense）時のプレゼンテーション手順
1. **必須機能のデモ**: `make run` および `make lint-strict` で必須パートの健全性を証明。
2. **ボーナス機能の切り替え**: `make run-bonus` を実行し、ターミナル上でトークン制限のリアルタイム可視化や自作トークナイザーの動作を実演。
3. **コード構造の説明**: `src/core/` と `src/bonus/` が疎結合に分離されており、必須処理へ悪影響を与えない設計になっていることを提示。
