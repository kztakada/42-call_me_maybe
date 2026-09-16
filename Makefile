.PHONY: install run debug clean lint lint-strict test test-full

# 依存関係のインストール（レビュアーが実行するコマンド）
install:
	uv sync

# メインスクリプトの実行（python -m src）
run:
	uv run python -m src

# デバッグモードでの実行
debug:
	uv run python -m pdb -m src 

# キャッシュや出力ファイルの削除
clean:
	rm -rf .venv .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 
	find . -type f -name "*.pyc" -delete
	rm -rf data/output/*

# lint チェック
lint:
	uv run flake8 src tests
	uv run mypy src tests --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src tests
	uv run mypy src tests --strict --ignore-missing-imports
	uv run pydoclint src tests

test:
	uv run pytest -v --maxfail=1 --disable-warnings

test-full:
	uv run pytest -v
