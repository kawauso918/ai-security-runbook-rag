# 短期拡張案：OCR対応 & Re-ranking 実装計画

**対象システム**: AIセキュリティ運用手順書アシスタント（RAG）
**作成日**: 2025-12-30
**対象機能**: (1) OCR対応、(2) Re-ranking
**想定実装期間**: 設計→実装→テストまで

---

## 📋 目次

1. [概要](#概要)
2. [1. OCR対応（スキャンPDF対応）](#1-ocr対応スキャンpdf対応)
   - [1.1 課題と目的](#11-課題と目的)
   - [1.2 アーキテクチャ](#12-アーキテクチャ)
   - [1.3 実装設計](#13-実装設計)
   - [1.4 セットアップ手順](#14-セットアップ手順)
   - [1.5 タスク分割](#15-タスク分割)
3. [2. Re-ranking（検索精度向上）](#2-re-ranking検索精度向上)
   - [2.1 課題と目的](#21-課題と目的)
   - [2.2 アーキテクチャ](#22-アーキテクチャ)
   - [2.3 実装設計](#23-実装設計)
   - [2.4 セットアップ手順](#24-セットアップ手順)
   - [2.5 タスク分割](#25-タスク分割)
4. [統合実装のタイムライン](#統合実装のタイムライン)
5. [評価方法](#評価方法)

---

## 概要

既存の「AIセキュリティ運用手順書アシスタント」に対して、**短期的な改善**として以下2つの機能を追加します。

### 拡張機能の概要

| 機能 | 目的 | 手法候補 | 優先度 |
|------|------|----------|--------|
| **OCR対応** | スキャンPDFからテキスト抽出を可能にする | Tesseract OCR（OSS）/ Azure Document Intelligence（クラウド） | 高 |
| **Re-ranking** | ハイブリッド検索後の結果を再ランキングして精度向上 | Cohere Rerank API / LLMベースReranking | 高 |

### 既存システムとの関係

```
[既存システム]
PDF/MD読み込み → チャンキング → ハイブリッド検索（BM25+Vector） → LLM生成 → 回答
                      ↓                          ↓
               [拡張1: OCR対応]          [拡張2: Re-ranking]
```

---

## 1. OCR対応（スキャンPDF対応）

### 1.1 課題と目的

**現状の課題**:
- スキャン画像のみのPDFからテキスト抽出ができない
- `pypdf`では画像PDFに対して空文字列が返る
- エラー表示のみで、ユーザーはOCR処理を選択できない

**目的**:
- スキャンPDFに対してOCR処理を適用し、テキスト抽出を可能にする
- 既存のPDF処理パイプライン（見出し推定→セクション化）にシームレスに統合する

**要件**:
- ✅ Tesseract OCR（ローカル、無料）を優先的に検討
- ✅ Azure Document Intelligence（クラウド、有料）を代替案として用意
- ✅ OCR処理の有効化/無効化をUI/設定で切り替え可能にする
- ✅ OCR処理中の進捗表示（ページ単位）

---

### 1.2 アーキテクチャ

#### 既存フロー vs 拡張フロー

**既存フロー**:
```
PDF読み込み（pypdf）
  ↓
テキスト抽出（extract_text()）
  ↓
テキストが空？ → YES → エラー表示
  ↓ NO
見出し推定 → セクション化 → チャンキング
```

**拡張フロー（OCR統合）**:
```
PDF読み込み（pypdf）
  ↓
テキスト抽出（extract_text()）
  ↓
テキストが空？ → YES → OCR処理（Tesseract/Azure）
  ↓ NO              ↓
  └─────────────────┘
  ↓
見出し推定 → セクション化 → チャンキング
```

#### 処理フローチャート

```
┌──────────────────────────────────────────────┐
│  load_documents(data_folder)                  │
│    - PDFファイルを検出                         │
│    - pdf_to_sections_with_ocr() を呼び出し   │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│  pdf_to_sections_with_ocr(pdf_path)          │
│    1. pypdfでテキスト抽出を試みる              │
│    2. テキストが極端に少ない場合:              │
│       - OCR有効？ → YES: OCR処理実行          │
│                   → NO: エラー返却             │
│    3. 既存の見出し推定ロジックに渡す           │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│  ocr_extract_text_from_pdf(pdf_path, method) │
│    method = 'tesseract' or 'azure'            │
│    - PDFをページ画像に変換（pdf2image）        │
│    - 各ページにOCR適用                         │
│    - 結果を [(page_no, text), ...] で返す     │
└──────────────────────────────────────────────┘
```

---

### 1.3 実装設計

#### 1.3.1 関数設計

##### 関数1: `ocr_extract_text_from_pdf()`

**目的**: PDFから画像を抽出してOCR処理

**I/O仕様**:
```python
def ocr_extract_text_from_pdf(
    pdf_path: str,
    method: str = "tesseract",  # 'tesseract' or 'azure'
    language: str = "jpn",
    progress_callback: Optional[callable] = None
) -> List[Tuple[int, str]]:
    """PDFから画像を抽出してOCR処理を実行

    Args:
        pdf_path: PDFファイルのパス
        method: OCR手法（'tesseract' or 'azure'）
        language: OCR言語（'jpn', 'eng'）
        progress_callback: 進捗コールバック（page_no, total_pages）

    Returns:
        [(page_no, extracted_text), ...] のリスト（page_noは1始まり）

    Raises:
        ImportError: Tesseractがインストールされていない
        ValueError: Azure APIキーが設定されていない
        Exception: OCR処理エラー
    """
```

**擬似コード**:
```python
def ocr_extract_text_from_pdf(pdf_path, method="tesseract", language="jpn", progress_callback=None):
    # PDFをページ画像に変換
    images = pdf2image.convert_from_path(pdf_path)

    pages_text = []
    total_pages = len(images)

    for i, image in enumerate(images, 1):
        if progress_callback:
            progress_callback(i, total_pages)

        if method == "tesseract":
            # Tesseract OCR
            text = pytesseract.image_to_string(image, lang=language)
        elif method == "azure":
            # Azure Document Intelligence
            text = _ocr_with_azure(image)
        else:
            raise ValueError(f"Unknown OCR method: {method}")

        pages_text.append((i, text))

    return pages_text
```

---

##### 関数2: `_ocr_with_tesseract()`

**目的**: Tesseract OCRを使ったテキスト抽出

**I/O仕様**:
```python
def _ocr_with_tesseract(
    image: PIL.Image.Image,
    language: str = "jpn"
) -> str:
    """Tesseract OCRでテキスト抽出

    Args:
        image: PIL Image オブジェクト
        language: OCR言語（'jpn', 'eng'）

    Returns:
        抽出されたテキスト
    """
```

**擬似コード**:
```python
def _ocr_with_tesseract(image, language="jpn"):
    try:
        import pytesseract
        text = pytesseract.image_to_string(image, lang=language)
        return text.strip()
    except ImportError:
        raise ImportError(
            "pytesseractがインストールされていません。\n"
            "インストール方法:\n"
            "  Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-jpn\n"
            "  macOS: brew install tesseract tesseract-lang\n"
            "  pip install pytesseract pdf2image"
        )
```

---

##### 関数3: `_ocr_with_azure()`

**目的**: Azure Document Intelligenceを使ったテキスト抽出

**I/O仕様**:
```python
def _ocr_with_azure(
    image: PIL.Image.Image,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """Azure Document Intelligenceでテキスト抽出

    Args:
        image: PIL Image オブジェクト
        endpoint: Azure endpoint（環境変数から取得可能）
        api_key: Azure APIキー（環境変数から取得可能）

    Returns:
        抽出されたテキスト
    """
```

**擬似コード**:
```python
def _ocr_with_azure(image, endpoint=None, api_key=None):
    import os
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential

    # 環境変数から取得
    endpoint = endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    api_key = api_key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not api_key:
        raise ValueError("Azure Document IntelligenceのエンドポイントとAPIキーが設定されていません")

    # Azure APIクライアント
    client = DocumentAnalysisClient(endpoint, AzureKeyCredential(api_key))

    # PIL ImageをBytesIOに変換
    import io
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # OCR実行
    poller = client.begin_analyze_document("prebuilt-read", img_bytes)
    result = poller.result()

    # テキスト抽出
    lines = []
    for page in result.pages:
        for line in page.lines:
            lines.append(line.content)

    return '\n'.join(lines)
```

---

##### 関数4: `pdf_to_sections_with_ocr()`

**目的**: 既存の`pdf_to_sections()`にOCR処理を統合

**I/O仕様**:
```python
def pdf_to_sections_with_ocr(
    pdf_path: str,
    ocr_enabled: bool = True,
    ocr_method: str = "tesseract",
    ocr_language: str = "jpn",
    progress_callback: Optional[callable] = None
) -> List[Dict]:
    """PDFを見出し推定→セクション化（OCR対応版）

    Args:
        pdf_path: PDFファイルのパス
        ocr_enabled: OCR処理を有効化
        ocr_method: OCR手法（'tesseract' or 'azure'）
        ocr_language: OCR言語
        progress_callback: 進捗コールバック

    Returns:
        Documentのリスト（既存のpdf_to_sections()と同じ形式）
    """
```

**擬似コード**:
```python
def pdf_to_sections_with_ocr(pdf_path, ocr_enabled=True, ocr_method="tesseract",
                              ocr_language="jpn", progress_callback=None):
    # 1. pypdfでテキスト抽出を試みる
    try:
        pages = extract_pdf_pages(pdf_path)
    except Exception as e:
        if ocr_enabled:
            # pypdf失敗 → OCR処理にフォールバック
            pages = ocr_extract_text_from_pdf(pdf_path, method=ocr_method,
                                              language=ocr_language,
                                              progress_callback=progress_callback)
        else:
            raise e

    # 2. テキストが極端に少ない場合 → OCR処理
    total_text_length = sum(len(text) for _, text in pages)
    if total_text_length < 100 and ocr_enabled:
        pages = ocr_extract_text_from_pdf(pdf_path, method=ocr_method,
                                          language=ocr_language,
                                          progress_callback=progress_callback)

    # 3. 既存の見出し推定→セクション化ロジック
    # （normalize_pdf_text, remove_repeated_lines, score_heading_line等）
    normalized_pages = [(p, normalize_pdf_text(t)) for p, t in pages]
    cleaned_pages = remove_repeated_lines(normalized_pages)
    sections = _extract_sections_from_pages(cleaned_pages, pdf_path)

    return sections
```

---

#### 1.3.2 定数追加（constants.py）

```python
# OCR設定
OCR_ENABLED = True  # デフォルトでOCRを有効化
OCR_METHOD = "tesseract"  # 'tesseract' or 'azure'
OCR_LANGUAGE = "jpn"  # 日本語
OCR_MIN_TEXT_LENGTH = 100  # この文字数未満の場合OCR処理を試みる
```

---

#### 1.3.3 UI統合（main.py / components.py）

**サイドバーにOCR設定を追加**:
```python
# main.py の render_sidebar() 内
st.sidebar.header("⚙️ OCR設定")
ocr_enabled = st.sidebar.checkbox("OCR処理を有効化", value=True,
                                   help="スキャンPDFからテキスト抽出を試みます")
if ocr_enabled:
    ocr_method = st.sidebar.selectbox("OCR手法", ["tesseract", "azure"],
                                       help="Tesseract（ローカル）またはAzure（クラウド）")
```

**プログレスバー表示**:
```python
# initialize.py の load_documents() 内
def progress_callback(page_no, total_pages):
    progress_bar.progress(page_no / total_pages)
    status_text.text(f"OCR処理中: {page_no}/{total_pages} ページ")

with st.spinner("PDFファイルを処理中..."):
    progress_bar = st.progress(0)
    status_text = st.empty()

    pdf_sections = pdf_to_sections_with_ocr(
        str(pdf_file),
        ocr_enabled=ocr_enabled,
        ocr_method=ocr_method,
        progress_callback=progress_callback
    )
```

---

### 1.4 セットアップ手順

#### Tesseract OCR（推奨：ローカル、無料）

**1. Tesseractのインストール**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-jpn

# macOS
brew install tesseract tesseract-lang

# Windows
# https://github.com/UB-Mannheim/tesseract/wiki からインストーラーをダウンロード
```

**2. Pythonパッケージのインストール**

```bash
pip install pytesseract pdf2image pillow
```

**3. Popplerのインストール（pdf2image用）**

```bash
# Ubuntu/Debian
sudo apt install poppler-utils

# macOS
brew install poppler

# Windows
# https://github.com/oschwartz10612/poppler-windows/releases/ からダウンロード
```

**4. 動作確認**

```python
import pytesseract
print(pytesseract.get_languages())  # ['eng', 'jpn', ...] が表示されればOK
```

---

#### Azure Document Intelligence（代替案：クラウド、有料）

**1. Azureリソースの作成**

- Azure Portalで「Document Intelligence」リソースを作成
- エンドポイントとAPIキーを取得

**2. Pythonパッケージのインストール**

```bash
pip install azure-ai-formrecognizer
```

**3. 環境変数の設定**

`.env`ファイルに追加:
```env
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your_api_key_here
```

**4. 動作確認**

```python
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os

endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
print("Azure Document Intelligence接続成功")
```

---

### 1.5 タスク分割

| タスクID | タスク内容 | 担当ファイル | 所要時間目安 | 優先度 |
|---------|-----------|------------|------------|--------|
| OCR-1 | `ocr_extract_text_from_pdf()` 実装（Tesseract版） | `utils.py` | - | P0 |
| OCR-2 | `_ocr_with_tesseract()` 実装 | `utils.py` | - | P0 |
| OCR-3 | `_ocr_with_azure()` 実装（オプション） | `utils.py` | - | P1 |
| OCR-4 | `pdf_to_sections_with_ocr()` 実装 | `utils.py` | - | P0 |
| OCR-5 | `constants.py` にOCR設定を追加 | `constants.py` | - | P0 |
| OCR-6 | `initialize.py` のPDF読み込み処理を統合 | `initialize.py` | - | P0 |
| OCR-7 | UIにOCR設定を追加（サイドバー） | `main.py` | - | P1 |
| OCR-8 | プログレスバー表示の実装 | `main.py` | - | P1 |
| OCR-9 | エラーハンドリングとユーザー向けメッセージ | `error_handler.py` | - | P1 |
| OCR-10 | テストケース作成（スキャンPDF準備） | `tests/` | - | P2 |
| OCR-11 | 評価（OCR精度、処理時間測定） | `eval/` | - | P2 |
| OCR-12 | ドキュメント更新（README.md） | `README.md` | - | P2 |

**実装順序**:
1. OCR-1, OCR-2（Tesseract版の基本実装）
2. OCR-4, OCR-5（既存パイプラインへの統合）
3. OCR-6（initialize.pyの修正）
4. OCR-7, OCR-8（UI統合）
5. OCR-9, OCR-10, OCR-11（テスト・評価）
6. OCR-3（Azure版はオプション）
7. OCR-12（ドキュメント更新）

---

## 2. Re-ranking（検索精度向上）

### 2.1 課題と目的

**現状の課題**:
- ハイブリッド検索（BM25 + Vector）の結果は、スコアの単純な重み付き和で決まる
- 意味的に関連性が低いが、キーワードマッチで高スコアになるケースがある
- ユーザーの質問意図と検索結果の関連性を再評価する仕組みがない

**目的**:
- 検索結果をLLMまたは専用モデルで**再ランキング**し、精度を向上させる
- 質問意図との関連性を深く評価し、無関係な結果を除外する

**要件**:
- ✅ Cohere Rerank API（商用、高精度）を優先的に検討
- ✅ LLMベースReranking（OpenAI、ローカルLLM）を代替案として用意
- ✅ Re-rankingの有効化/無効化をUI/設定で切り替え可能にする
- ✅ Re-ranking後のスコアをログに記録

---

### 2.2 アーキテクチャ

#### 既存フロー vs 拡張フロー

**既存フロー**:
```
質問入力
  ↓
ハイブリッド検索（BM25 + Vector）
  ↓
スコア統合（重み付き和）
  ↓
上位k件を取得
  ↓
LLM生成
```

**拡張フロー（Re-ranking統合）**:
```
質問入力
  ↓
ハイブリッド検索（BM25 + Vector）
  ↓
スコア統合（重み付き和）
  ↓
上位k'件を取得（k' > k、例: k'=10）
  ↓
Re-ranking（Cohere / LLM）
  ↓
上位k件に再絞り込み
  ↓
LLM生成
```

#### 処理フローチャート

```
┌──────────────────────────────────────────────┐
│  handle_query(user_query)                     │
│    - hybrid_search() で初期検索（k'件）       │
│    - rerank_search_results() で再ランキング  │
│    - 上位k件をLLMに渡す                       │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│  rerank_search_results(query, results, k)    │
│    method = 'cohere' or 'llm' or 'none'       │
│    - Cohere Rerank APIまたはLLMを使用         │
│    - 各結果のrelevance_scoreを計算           │
│    - スコア降順でソート                       │
│    - 上位k件を返す                            │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│  _rerank_with_cohere() / _rerank_with_llm()  │
│    - Cohere Rerank APIまたはLLM呼び出し      │
│    - relevance_scoreを取得                   │
└──────────────────────────────────────────────┘
```

---

### 2.3 実装設計

#### 2.3.1 関数設計

##### 関数1: `rerank_search_results()`

**目的**: 検索結果を再ランキング

**I/O仕様**:
```python
def rerank_search_results(
    query: str,
    search_results: List[Dict],
    k: int = 4,
    method: str = "cohere",  # 'cohere', 'llm', 'none'
    model: Optional[str] = None
) -> List[Dict]:
    """検索結果を再ランキング

    Args:
        query: ユーザーの質問
        search_results: ハイブリッド検索の結果
        k: 返却する結果数
        method: Re-ranking手法（'cohere', 'llm', 'none'）
        model: LLMモデル名（method='llm'の場合）

    Returns:
        再ランキング後の検索結果（上位k件）
        各結果に 'rerank_score' フィールドが追加される
    """
```

**擬似コード**:
```python
def rerank_search_results(query, search_results, k=4, method="cohere", model=None):
    if method == "none" or not search_results:
        return search_results[:k]

    if method == "cohere":
        # Cohere Rerank API
        reranked_results = _rerank_with_cohere(query, search_results, k)
    elif method == "llm":
        # LLMベースReranking
        reranked_results = _rerank_with_llm(query, search_results, k, model)
    else:
        raise ValueError(f"Unknown rerank method: {method}")

    return reranked_results
```

---

##### 関数2: `_rerank_with_cohere()`

**目的**: Cohere Rerank APIを使った再ランキング

**I/O仕様**:
```python
def _rerank_with_cohere(
    query: str,
    search_results: List[Dict],
    k: int = 4,
    api_key: Optional[str] = None
) -> List[Dict]:
    """Cohere Rerank APIで再ランキング

    Args:
        query: ユーザーの質問
        search_results: 検索結果
        k: 返却する結果数
        api_key: Cohere APIキー（環境変数から取得可能）

    Returns:
        再ランキング後の検索結果（rerank_scoreを含む）
    """
```

**擬似コード**:
```python
def _rerank_with_cohere(query, search_results, k=4, api_key=None):
    import os
    import cohere

    # 環境変数から取得
    api_key = api_key or os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("Cohere APIキーが設定されていません")

    # Cohere Client
    co = cohere.Client(api_key)

    # 検索結果のテキストを抽出
    documents = [result['text'] for result in search_results]

    # Rerank API呼び出し
    rerank_response = co.rerank(
        query=query,
        documents=documents,
        top_n=k,
        model="rerank-multilingual-v3.0"  # 日本語対応モデル
    )

    # 結果を再構築
    reranked_results = []
    for item in rerank_response.results:
        result = search_results[item.index].copy()
        result['rerank_score'] = item.relevance_score
        reranked_results.append(result)

    return reranked_results
```

---

##### 関数3: `_rerank_with_llm()`

**目的**: LLMを使った再ランキング

**I/O仕様**:
```python
def _rerank_with_llm(
    query: str,
    search_results: List[Dict],
    k: int = 4,
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """LLMで再ランキング

    Args:
        query: ユーザーの質問
        search_results: 検索結果
        k: 返却する結果数
        model: LLMモデル名

    Returns:
        再ランキング後の検索結果（rerank_scoreを含む）
    """
```

**擬似コード**:
```python
def _rerank_with_llm(query, search_results, k=4, model="gpt-4o-mini"):
    from openai import OpenAI

    client = OpenAI()

    # プロンプト構築
    prompt = f"""
以下の検索結果について、質問との関連性を0-100点で評価してください。

【質問】
{query}

【検索結果】
"""
    for i, result in enumerate(search_results):
        prompt += f"\n[結果{i+1}]\n{result['text'][:200]}...\n"

    prompt += """
【出力形式】
JSON形式で各結果のスコアを出力してください。
{
    "scores": [85, 70, 45, ...],  # 結果1, 結果2, 結果3, ... のスコア
    "reasoning": "評価理由"
}
"""

    # LLM呼び出し
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    # スコアを取得
    import json
    result_data = json.loads(response.choices[0].message.content)
    scores = result_data.get("scores", [])

    # 結果にスコアを付与
    for i, result in enumerate(search_results):
        result['rerank_score'] = scores[i] / 100.0 if i < len(scores) else 0.0

    # スコア降順でソート
    reranked_results = sorted(search_results, key=lambda x: x['rerank_score'], reverse=True)

    return reranked_results[:k]
```

---

#### 2.3.2 定数追加（constants.py）

```python
# Re-ranking設定
RERANK_ENABLED = True  # デフォルトでRe-rankingを有効化
RERANK_METHOD = "cohere"  # 'cohere', 'llm', 'none'
RERANK_LLM_MODEL = "gpt-4o-mini"  # LLMベースReranking用モデル
RERANK_TOP_K_BEFORE = 10  # Re-ranking前に取得する結果数（k' > k）
```

---

#### 2.3.3 パイプライン統合（main.py）

**handle_query() の修正**:
```python
def handle_query(user_query, session_state):
    # 1. ハイブリッド検索（k'件を取得）
    k_before_rerank = RERANK_TOP_K_BEFORE if session_state['rerank_enabled'] else session_state['k']
    search_results = hybrid_search(
        query=user_query,
        k=k_before_rerank,
        bm25_weight=session_state['bm25_weight'],
        vector_weight=session_state['vector_weight'],
        vectorstore=session_state['vectorstore'],
        bm25_index=session_state['bm25_index'],
        chunks_metadata=session_state['chunks_metadata']
    )

    # 2. Re-ranking
    if session_state['rerank_enabled']:
        search_results = rerank_search_results(
            query=user_query,
            search_results=search_results,
            k=session_state['k'],
            method=session_state['rerank_method']
        )

    # 3. 根拠不足判定
    if check_insufficient_evidence(search_results):
        return get_insufficient_evidence_response()

    # 4. LLM生成
    answer = generate_answer(user_query, search_results)

    return answer, search_results
```

---

#### 2.3.4 UI統合（main.py）

**サイドバーにRe-ranking設定を追加**:
```python
# main.py の render_sidebar() 内
st.sidebar.header("🔀 Re-ranking設定")
rerank_enabled = st.sidebar.checkbox("Re-rankingを有効化", value=True,
                                      help="検索結果を再ランキングして精度を向上させます")
if rerank_enabled:
    rerank_method = st.sidebar.selectbox("Re-ranking手法",
                                          ["cohere", "llm", "none"],
                                          help="Cohere（推奨）またはLLMベース")
```

---

### 2.4 セットアップ手順

#### Cohere Rerank API（推奨：高精度、有料）

**1. Cohere APIキーの取得**

- [Cohere Dashboard](https://dashboard.cohere.com/) でアカウント作成
- APIキーを取得（無料プランあり、月1,000リクエストまで）

**2. Pythonパッケージのインストール**

```bash
pip install cohere
```

**3. 環境変数の設定**

`.env`ファイルに追加:
```env
COHERE_API_KEY=your_cohere_api_key_here
```

**4. 動作確認**

```python
import cohere
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))
response = co.rerank(
    query="セキュリティインシデントの対応手順は？",
    documents=["手順1", "手順2"],
    top_n=2,
    model="rerank-multilingual-v3.0"
)
print(response.results)
```

---

#### LLMベースReranking（代替案：OpenAI、無料枠あり）

**1. 追加のセットアップ不要**

- 既存のOpenAI APIキーを使用
- `gpt-4o-mini`を推奨（コスト効率が良い）

**2. 動作確認**

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "テスト"}]
)
print(response.choices[0].message.content)
```

---

### 2.5 タスク分割

| タスクID | タスク内容 | 担当ファイル | 所要時間目安 | 優先度 |
|---------|-----------|------------|------------|--------|
| RR-1 | `rerank_search_results()` 実装 | `retriever.py` | - | P0 |
| RR-2 | `_rerank_with_cohere()` 実装 | `retriever.py` | - | P0 |
| RR-3 | `_rerank_with_llm()` 実装 | `retriever.py` | - | P1 |
| RR-4 | `constants.py` にRe-ranking設定を追加 | `constants.py` | - | P0 |
| RR-5 | `main.py` のhandle_query()を修正 | `main.py` | - | P0 |
| RR-6 | UIにRe-ranking設定を追加（サイドバー） | `main.py` | - | P1 |
| RR-7 | ログにrerank_scoreを記録 | `logger.py` | - | P1 |
| RR-8 | エラーハンドリング（API失敗時） | `error_handler.py` | - | P1 |
| RR-9 | テストケース作成（精度比較） | `tests/` | - | P2 |
| RR-10 | 評価（Re-ranking前後の精度測定） | `eval/` | - | P2 |
| RR-11 | ドキュメント更新（README.md） | `README.md` | - | P2 |

**実装順序**:
1. RR-1, RR-2（Cohere版の基本実装）
2. RR-4（定数追加）
3. RR-5（パイプライン統合）
4. RR-6（UI統合）
5. RR-7（ログ記録）
6. RR-8, RR-9, RR-10（テスト・評価）
7. RR-3（LLM版はオプション）
8. RR-11（ドキュメント更新）

---

## 統合実装のタイムライン

以下の順序で実装を進めることを推奨します：

### フェーズ1: Re-ranking実装（優先度高）

**理由**: Re-rankingは既存パイプラインへの統合が容易で、即座に精度向上が見込める

1. **RR-1〜RR-5**: Re-ranking基本実装とパイプライン統合
2. **RR-6〜RR-7**: UI統合とログ記録
3. **RR-9〜RR-10**: テスト・評価

**期間目安**: -

---

### フェーズ2: OCR実装（優先度中）

**理由**: OCRは環境構築（Tesseractインストール等）が必要だが、スキャンPDF対応の需要が高い

1. **OCR-1〜OCR-6**: OCR基本実装とパイプライン統合
2. **OCR-7〜OCR-8**: UI統合とプログレスバー
3. **OCR-10〜OCR-11**: テスト・評価

**期間目安**: -

---

### フェーズ3: ドキュメント更新・最終評価

1. **OCR-12, RR-11**: README.mdの更新
2. 統合テスト（OCR + Re-ranking両方を有効化したシナリオ）
3. 評価結果の分析とチューニング

**期間目安**: -

---

## 評価方法

### 評価指標

| 指標 | 評価内容 | 測定方法 |
|------|---------|---------|
| **OCR精度** | スキャンPDFからのテキスト抽出精度 | 手動確認（サンプル10ページ）、文字認識率 |
| **OCR処理時間** | ページあたりの処理時間 | ログ記録（latency_ms） |
| **Re-ranking精度** | 検索結果の関連性向上率 | LLM as a Judge評価（根拠性・正確性スコア） |
| **Re-rankingコスト** | API呼び出しコスト | ログ記録（cost_usd） |
| **統合精度** | OCR + Re-ranking両方を有効化した場合の回答品質 | LLM as a Judge評価（10問セット、合格ライン70点） |

---

### 評価データセット

**OCR評価用**:
- スキャンPDFサンプル（5ファイル、計50ページ）
- 日本語/英語混在のドキュメント
- 表・図を含むページ

**Re-ranking評価用**:
- 既存の`eval/eval_dataset.json`を拡張
- 曖昧な質問（キーワードマッチで誤検出しやすいケース）を追加
- 例: 「ログインできない」→「ログファイルの削除」が誤検出される

**評価実行コマンド**:
```bash
# Re-ranking有効/無効の比較
python eval/run_evaluation.py --rerank-enabled
python eval/run_evaluation.py --rerank-disabled

# OCR対応PDFの評価
python eval/run_evaluation.py --ocr-enabled --pdf-path data/scanned_sample.pdf
```

---

### 成功基準

| 項目 | 目標 |
|------|------|
| **OCR文字認識率** | 95%以上（日本語、明瞭なスキャン） |
| **OCR処理時間** | 1ページあたり5秒以内 |
| **Re-ranking精度向上** | LLM as a Judge評価で平均+5点以上 |
| **Re-rankingコスト** | 1質問あたり$0.01未満（Cohere無料枠内） |
| **統合評価** | 10問中8問以上が平均75点以上 |

---

## まとめ

この実装計画に従うことで、以下の改善が見込まれます：

1. **OCR対応**: スキャンPDFからテキスト抽出が可能になり、対応ドキュメントの範囲が拡大
2. **Re-ranking**: 検索精度が向上し、無関係な結果が減少、ユーザー満足度が向上
3. **拡張性**: 将来的な機能追加（マルチモーダル対応、外部API統合等）の基盤が整う

**次のステップ**:
- このドキュメントをREADME.mdに追記または別ファイルとして保存
- フェーズ1（Re-ranking）から実装を開始
- 各タスクの完了後、評価を実施してチューニング

---

**作成者**: Claude Sonnet 4.5
**バージョン**: 1.0
**最終更新**: 2025-12-30
