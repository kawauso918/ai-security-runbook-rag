# AIセキュリティ運用手順書アシスタント（RAG）

SOC運用×業務系RAGシステム。手順書を検索して質問に回答します。

## 機能

- **ハイブリッド検索**: BM25 + ベクトル検索の組み合わせ
- **根拠不足判定**: 検索スコアが低い場合は「該当する手順が見つかりませんでした」と返答
- **危険操作検知**: 削除・消去などの危険な操作を含む場合は警告を表示
- **会話履歴**: セッション内で5往復の会話を保持
- **ログ記録**: JSONL形式でログを記録（監査用）

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd ai-security-runbook-rag
```

### 2. 仮想環境の作成と有効化

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# または
.venv\Scripts\activate  # Windows
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

`.env.example`をコピーして`.env`を作成し、OpenAI APIキーを設定：

```bash
cp .env.example .env
# .envファイルを編集してOPENAI_API_KEYを設定
```

### 5. データの準備

`data/`フォルダに手順書ファイル（PDF/Markdown/テキスト）を配置します。

サンプルデータが既に含まれています：
- `data/sample_manual.md`
- `data/incident_response.md`

## 起動方法

```bash
streamlit run main.py
```

ブラウザで `http://localhost:8501` が開きます。

## 使い方

1. **インデックス構築**: サイドバーから「インデックス再構築」ボタンをクリック
2. **質問入力**: メインエリアのチャット入力欄に質問を入力
3. **回答確認**: 回答と引用元が表示されます

### サイドバー設定

- **データフォルダパス**: 手順書ファイルを格納しているフォルダのパス
- **検索結果数 (k)**: 検索結果として取得するチャンク数（デフォルト: 4）
- **BM25重み**: BM25検索の重み（デフォルト: 0.6）
- **ベクトル重み**: ベクトル検索の重み（自動計算: 1.0 - BM25重み）

## ファイル構成

```
ai-security-runbook-rag/
├── main.py              # Streamlitメインアプリ
├── initialize.py        # インデックス構築
├── retriever.py         # ハイブリッド検索
├── guardrails.py        # ガードレール（危険操作検知等）
├── logger.py            # ログ記録
├── components.py        # UIコンポーネント
├── utils.py             # ユーティリティ関数
├── constants.py         # 定数定義
├── requirements.txt     # 依存パッケージ
├── .env.example         # 環境変数テンプレート
├── .gitignore
├── data/                # 手順書ファイル格納フォルダ
├── logs/                # ログファイル（JSONL形式）
├── eval/                # 評価データ
└── chroma_db/           # ChromaDBデータ（自動生成）
```

## 完了条件の確認

以下の機能が正常に動作することを確認してください：

- ✅ 質問→回答→引用（ファイル名>見出し>抜粋）が表示される
- ✅ ハイブリッド検索（BM25 0.6 / vector 0.4、k=4）で動く
- ✅ スコア最高値<0.5 で「該当する手順が見つかりませんでした」と返る
- ✅ 危険操作キーワードで警告バナー＆文言が出る
- ✅ logs/ にJSONLでログが残る

## 技術スタック

- **UI**: Streamlit
- **RAGフレームワーク**: LangChain
- **ベクトルDB**: ChromaDB
- **検索**: BM25 (rank-bm25) + ベクトル検索
- **LLM**: OpenAI API (gpt-4o-mini)
- **日本語処理**: Sudachi

## ライセンス

（ライセンスを記載）

## 詳細設計

詳細設計書は `DESIGN.md` を参照してください。
