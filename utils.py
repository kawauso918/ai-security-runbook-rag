"""ユーティリティ関数"""

import re
import os
import json
import urllib.parse
import urllib.request
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sudachipy import dictionary, tokenizer

from constants import (
    PDF_MAX_HEADING_LEN, PDF_MIN_HEADING_LEN,
    HEADING_SCORE_THRESHOLD, HEADING_SCORE_THRESHOLD_STRICT,
    HEADING_PATTERNS, HEADING_KEYWORDS
)


def extract_headings(text: str) -> List[Dict]:
    """Markdownから見出しを抽出"""
    headings = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        # Markdown見出しを検出 (#, ##, ###)
        match = re.match(r'^(#{1,3})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            heading_text = match.group(2)
            headings.append({
                'level': level,
                'text': heading_text,
                'line_number': i
            })
    
    return headings


def chunk_by_headings(
    text: str,
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 50,
    adaptive: bool = True,
    doc_type: str = 'text'
) -> List[Dict]:
    """見出しベースでチャンキング（適応的チャンキング対応）

    Args:
        text: チャンキング対象テキスト
        file_path: ファイルパス
        chunk_size: 基本チャンクサイズ
        overlap: オーバーラップサイズ
        adaptive: 適応的チャンキングを有効化（デフォルト: True）
        doc_type: ドキュメントタイプ（'pdf', 'markdown', 'text'）

    Returns:
        チャンクのリスト
    """
    chunks = []
    headings = extract_headings(text)
    lines = text.split('\n')

    # 見出しがない場合は全体を1チャンクとして処理
    if not headings:
        chunk_text = '\n'.join(lines)
        if chunk_text.strip():
            chunks.append({
                'chunk_id': f"{file_path}_0",
                'text': chunk_text.strip(),
                'heading': '',
                'file': file_path,
                'updated_at': datetime.fromtimestamp(os.path.getmtime(file_path) if os.path.exists(file_path) else 0).isoformat(),
                'chunk_index': 0
            })
        return chunks

    # 見出しごとにセクション分割
    for i, heading in enumerate(headings):
        start_line = heading['line_number']
        end_line = headings[i + 1]['line_number'] if i + 1 < len(headings) else len(lines)

        section_lines = lines[start_line:end_line]
        section_text = '\n'.join(section_lines)

        # セクション内でチャンキング（適応的チャンキングパラメータを渡す）
        section_chunks = chunk_text_adaptive(
            section_text,
            chunk_size,
            overlap,
            adaptive=adaptive,
            doc_type=doc_type
        )

        for j, chunk_text in enumerate(section_chunks):
            chunk_id = f"{file_path}_{len(chunks)}"
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text.strip(),
                'heading': heading['text'],
                'file': file_path,
                'updated_at': datetime.fromtimestamp(os.path.getmtime(file_path) if os.path.exists(file_path) else 0).isoformat(),
                'chunk_index': len(chunks)
            })

    return chunks


def _chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    adaptive: bool = True,
    doc_type: str = 'text'
) -> List[str]:
    """テキストを指定サイズでチャンキング（適応的チャンキング対応）

    Args:
        text: チャンキング対象テキスト
        chunk_size: 基本チャンクサイズ
        overlap: オーバーラップサイズ
        adaptive: 適応的チャンキングを有効化（デフォルト: True）
        doc_type: ドキュメントタイプ（'pdf', 'markdown', 'text'）

    Returns:
        チャンクのリスト
    """
    if len(text) <= chunk_size:
        return [text]

    # 適応的チャンキングが有効な場合、密度に基づいてチャンクサイズを調整
    if adaptive:
        from constants import ADAPTIVE_CHUNKING_ENABLED
        if ADAPTIVE_CHUNKING_ENABLED:
            density = analyze_content_density(text)
            chunk_size = calculate_adaptive_chunk_size(chunk_size, density, doc_type)

    chunks = []
    start = 0

    while start < len(text):
        # 仮のチャンク終端位置を計算
        tentative_end = min(start + chunk_size, len(text))

        # セマンティック境界で切断位置を最適化
        if tentative_end < len(text):
            chunk_end = find_semantic_boundary(text, tentative_end, search_range=100)
        else:
            chunk_end = len(text)

        # チャンクを抽出
        chunk = text[start:chunk_end].strip()
        if chunk:
            chunks.append(chunk)

        # 次のチャンクの開始位置を計算（オーバーラップを考慮）
        if chunk_end < len(text) and overlap > 0:
            # オーバーラップ開始位置を計算
            overlap_start = max(start, chunk_end - overlap)
            # セマンティック境界でオーバーラップ開始位置を調整
            next_start = find_semantic_boundary(text, overlap_start, search_range=50)
        else:
            next_start = chunk_end

        start = next_start

        # 無限ループ防止
        if start >= len(text):
            break

    return chunks


def chunk_text_adaptive(
    text: str,
    chunk_size: int,
    overlap: int,
    adaptive: bool = True,
    doc_type: str = 'text'
) -> List[str]:
    """_chunk_textの後方互換ラッパー（adaptive非対応でも動作させる）"""
    try:
        return _chunk_text(
            text,
            chunk_size,
            overlap,
            adaptive=adaptive,
            doc_type=doc_type
        )
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" in message:
            return _chunk_text(text, chunk_size, overlap)
        raise


def analyze_content_density(text: str) -> float:
    """テキストの情報密度を分析

    Args:
        text: 分析対象テキスト

    Returns:
        密度スコア（0.0-1.0、高いほど情報が密）
    """
    if not text or len(text) < 50:
        return 0.5  # 短いテキストはデフォルト値

    # Sudachiトークン化
    tokens = tokenize_japanese(text, preserve_special_tokens=True)

    # 文数を計算
    sentences = text.count('。') + text.count('.') + text.count('\n')
    if sentences == 0:
        sentences = 1
    avg_sentence_len = len(text) / sentences

    # キーワード密度（文字あたりのトークン数）
    keyword_density = len(tokens) / len(text)

    # 句読点密度
    punct_count = sum(1 for c in text if c in '。、.,;:!?「」『』（）()【】[]')
    punct_density = punct_count / len(text)

    # 技術用語密度（英数字を含むトークン）
    tech_terms = len([t for t in tokens if any(c.isalnum() for c in t)])
    tech_density = tech_terms / max(len(tokens), 1)

    # 重み付き平均で密度を計算
    density = (
        keyword_density * 0.4 +
        min(punct_density * 3, 1.0) * 0.3 +
        tech_density * 0.3
    )

    return min(max(density, 0.0), 1.0)


def calculate_adaptive_chunk_size(
    base_size: int,
    density_score: float,
    doc_type: str
) -> int:
    """密度に基づいてチャンクサイズを動的調整

    Args:
        base_size: ドキュメントタイプ別の基本チャンクサイズ
        density_score: コンテンツ密度（0.0-1.0）
        doc_type: ドキュメントタイプ（'pdf', 'markdown', 'text'）

    Returns:
        調整後のチャンクサイズ（CHUNK_SIZE_MIN～MAX範囲内）
    """
    from constants import (
        DENSITY_LOW_THRESHOLD, DENSITY_HIGH_THRESHOLD,
        DENSITY_LOW_MULTIPLIER, DENSITY_HIGH_MULTIPLIER,
        CHUNK_SIZE_MIN, CHUNK_SIZE_MAX
    )

    # 密度に応じてサイズを調整
    if density_score < DENSITY_LOW_THRESHOLD:
        # 疎なコンテンツ: 大きめのチャンク
        adjusted_size = int(base_size * DENSITY_LOW_MULTIPLIER)
    elif density_score > DENSITY_HIGH_THRESHOLD:
        # 密なコンテンツ: 小さめのチャンク
        adjusted_size = int(base_size * DENSITY_HIGH_MULTIPLIER)
    else:
        # 通常の密度: 基本サイズを使用
        adjusted_size = base_size

    # 範囲内に制限
    return max(CHUNK_SIZE_MIN, min(adjusted_size, CHUNK_SIZE_MAX))


def find_semantic_boundary(
    text: str,
    target_pos: int,
    search_range: int = 100
) -> int:
    """目標位置付近で最適なセマンティック境界を検出

    優先順位:
    1. 段落区切り（\n\n）
    2. 文末（。.!?）
    3. 句点（、,）
    4. 空白

    Args:
        text: 検索対象テキスト
        target_pos: 目標位置
        search_range: 前後の検索範囲（文字数）

    Returns:
        最適な境界位置
    """
    from constants import PARAGRAPH_BOUNDARY_WEIGHT, SENTENCE_BOUNDARY_WEIGHT

    if target_pos >= len(text):
        return len(text)

    start = max(0, target_pos - search_range)
    end = min(len(text), target_pos + search_range)

    best_pos = target_pos
    best_score = 0.0

    for i in range(start, end):
        score = 0.0
        distance = abs(i - target_pos)
        distance_penalty = 1.0 - (distance / search_range)

        # 境界の種類をチェック
        if i > 0 and i < len(text):
            # 段落区切り
            if text[i-1:i+1] == '\n\n':
                score = PARAGRAPH_BOUNDARY_WEIGHT
            # 文末
            elif text[i] in '。.!?':
                score = SENTENCE_BOUNDARY_WEIGHT
            # 句点
            elif text[i] in '、,;':
                score = 1.0
            # 空白
            elif text[i] in ' \n\t':
                score = 0.5

        # 距離ペナルティを適用
        final_score = score * distance_penalty

        if final_score > best_score:
            best_score = final_score
            best_pos = i + 1 if score > 0 else i

    return best_pos


def normalize_score(score: float, min_score: float, max_score: float) -> float:
    """スコアを0-1範囲に正規化"""
    if max_score == min_score:
        return 1.0
    normalized = (score - min_score) / (max_score - min_score)
    return max(0.0, min(1.0, normalized))


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
    """トークン使用量からコスト計算（USD）"""
    from constants import TOKEN_PRICES
    
    if model not in TOKEN_PRICES:
        return 0.0
    
    prices = TOKEN_PRICES[model]
    prompt_cost = (prompt_tokens / 1000) * prices["prompt"]
    completion_cost = (completion_tokens / 1000) * prices["completion"]
    
    return prompt_cost + completion_cost


def format_timestamp(dt: datetime) -> str:
    """ISO形式タイムスタンプ"""
    return dt.isoformat()


def tokenize_japanese(text: str, preserve_special_tokens: bool = True) -> List[str]:
    """日本語テキストをトークン化（Sudachi使用）

    Args:
        text: トークン化するテキスト
        preserve_special_tokens: 特殊トークン（コマンド、ID等）を原文保持するか

    Returns:
        トークンのリスト
    """
    if preserve_special_tokens:
        # 特殊トークンを保護するためのプレースホルダーに置換
        special_patterns = [
            # コマンド系（rm -rf, sudo, etc.）
            (r'\brm\s+-rf\b', '__RM_RF__'),
            (r'\bsudo\s+\w+', '__SUDO_CMD__'),
            (r'\b(?:chmod|chown|kill|shutdown|reboot)\s+[\w\-]+', '__DANGEROUS_CMD__'),
            # アラートID系（英数字+ハイフン、例: ALERT-001, CVE-2021-1234）
            (r'\b[A-Z]{2,}[\-_][A-Z0-9\-_]{3,}\b', '__ALERT_ID__'),
            # IPアドレス
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '__IP_ADDR__'),
            # ファイルパス（Linux系）
            (r'/[\w\-/\.]+', '__FILE_PATH__'),
            # メールアドレス
            (r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '__EMAIL__'),
        ]

        # 特殊トークンを保存（置換前の値を保持）
        replacements = {}
        protected_text = text

        for pattern, placeholder in special_patterns:
            matches = re.finditer(pattern, protected_text)
            for match in matches:
                original = match.group(0)
                # プレースホルダー + 一意のキー
                key = f"{placeholder}_{len(replacements)}"
                replacements[key] = original
                protected_text = protected_text.replace(original, key, 1)

        # Sudachiでトークン化
        try:
            tokenizer_obj = dictionary.Dictionary().create()
            mode = tokenizer.Tokenizer.SplitMode.C
            tokens = [m.surface() for m in tokenizer_obj.tokenize(protected_text, mode)]

            # プレースホルダーを元に戻す
            restored_tokens = []
            for token in tokens:
                if token in replacements:
                    restored_tokens.append(replacements[token])
                else:
                    restored_tokens.append(token)

            return restored_tokens
        except Exception:
            # Sudachiが使えない場合は単純に文字列を分割
            return protected_text.split()
    else:
        # 通常のトークン化
        try:
            tokenizer_obj = dictionary.Dictionary().create()
            mode = tokenizer.Tokenizer.SplitMode.C
            tokens = [m.surface() for m in tokenizer_obj.tokenize(text, mode)]
            return tokens
        except Exception:
            return list(text)


def external_search_duckduckgo(
    query: str,
    max_results: int = 5,
    timeout_sec: int = 8
) -> List[Dict[str, str]]:
    """DuckDuckGo Instant Answer APIで外部検索（簡易）

    Returns:
        [{'title': str, 'snippet': str, 'url': str}, ...]
    """
    if not query.strip():
        return []

    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "1"
    }
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            data = json.load(response)
    except Exception:
        return []

    results = []

    abstract_text = data.get("AbstractText")
    abstract_url = data.get("AbstractURL")
    heading = data.get("Heading")
    if abstract_text:
        results.append({
            "title": heading or "DuckDuckGo",
            "snippet": abstract_text,
            "url": abstract_url or "https://duckduckgo.com/?q=" + urllib.parse.quote(query)
        })

    related = data.get("RelatedTopics", [])
    for item in related:
        if isinstance(item, dict) and "Text" in item and "FirstURL" in item:
            results.append({
                "title": item.get("Text", "").split(" - ")[0] or "DuckDuckGo",
                "snippet": item.get("Text", ""),
                "url": item.get("FirstURL", "")
            })
        elif isinstance(item, dict) and "Topics" in item:
            for sub in item.get("Topics", []):
                if "Text" in sub and "FirstURL" in sub:
                    results.append({
                        "title": sub.get("Text", "").split(" - ")[0] or "DuckDuckGo",
                        "snippet": sub.get("Text", ""),
                        "url": sub.get("FirstURL", "")
                    })

        if len(results) >= max_results:
            break

    return results[:max_results]


# ==================== PDF処理関数 ====================

def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """PDFからページごとにテキストを抽出
    
    Args:
        pdf_path: PDFファイルのパス
    
    Returns:
        [(page_no, text), ...] のリスト（page_noは1始まり）
    """
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            raise ValueError("パスワード保護されたPDFファイルです")
        
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text is None:
                text = ""
            pages.append((i, text))
        
        return pages
    except ImportError:
        raise ImportError("pypdfがインストールされていません。pip install pypdf を実行してください。")
    except Exception as e:
        raise Exception(f"PDF読み込みエラー: {e}")


def normalize_pdf_text(text: str) -> str:
    """PDF抽出テキストのノイズを軽減
    
    Args:
        text: PDFから抽出したテキスト
    
    Returns:
        正規化されたテキスト
    """
    # 連続スペースを1つに
    text = re.sub(r' +', ' ', text)
    
    # 行末ハイフン分割を結合（"-\n" を削除）
    text = re.sub(r'-\n+', '', text)
    
    # 連続改行を2つまでに圧縮
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 先頭・末尾の空白・改行を削除
    text = text.strip()
    
    return text


def score_heading_line(line: str) -> int:
    """行が見出しかどうかをスコアリング
    
    Args:
        line: 評価する行
    
    Returns:
        スコア（高いほど見出しの可能性が高い）
    """
    line = line.strip()
    if not line:
        return 0
    
    score = 0
    
    # 文字数チェック
    char_count = len(line)
    if PDF_MIN_HEADING_LEN <= char_count <= PDF_MAX_HEADING_LEN:
        score += 1
    else:
        return 0  # 文字数が範囲外なら見出しではない
    
    # 句点で終わらない（見出しは通常句点なし）
    if not line.endswith('。'):
        score += 1
    
    # 記号が少ない（記号が多い行は見出しではない）
    symbol_count = len(re.findall(r'[!-/:-@\[-`{-~]', line))
    if symbol_count < char_count * 0.2:  # 記号が20%未満
        score += 1
    
    # パターンマッチング
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, line):
            score += 3
            break
    
    # キーワードチェック
    for keyword in HEADING_KEYWORDS:
        if keyword in line:
            score += 1
            break
    
    return score


def remove_repeated_lines(pages_texts: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """ヘッダー/フッターっぽい反復行を除外（簡易版）
    
    Args:
        pages_texts: [(page_no, text), ...] のリスト
    
    Returns:
        反復行を除外した [(page_no, text), ...] のリスト
    """
    if len(pages_texts) < 3:
        return pages_texts
    
    # 各行の出現回数をカウント
    all_lines = []
    for _, text in pages_texts:
        lines = text.split('\n')
        all_lines.extend([line.strip() for line in lines if line.strip()])
    
    line_counts = Counter(all_lines)
    
    # 3回以上出現する行を除外対象とする（簡易）
    repeated_lines = {line for line, count in line_counts.items() if count >= 3}
    
    # 除外して再構築
    cleaned_pages = []
    for page_no, text in pages_texts:
        lines = text.split('\n')
        cleaned_lines = [
            line for line in lines
            if line.strip() not in repeated_lines or len(line.strip()) > 50
        ]
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_pages.append((page_no, cleaned_text))
    
    return cleaned_pages


def pdf_to_sections(pdf_path: str) -> List[Dict]:
    """PDFを見出し推定→セクション化してDocumentリストに変換
    
    Args:
        pdf_path: PDFファイルのパス
    
    Returns:
        Documentのリスト。各Documentは以下の構造:
        {
            'file_path': str,
            'content': str,  # セクション本文
            'heading': str,  # 見出し（フォールバック時は "PAGE {n}"）
            'page_start': int,  # 開始ページ（1始まり）
            'page_end': int,  # 終了ページ（1始まり）
            'updated_at': str  # ISO形式
        }
    """
    # PDFからページごとにテキスト抽出
    pages = extract_pdf_pages(pdf_path)
    
    if not pages:
        return []
    
    # テキストがほぼ取れない場合（画像のみPDFの可能性）
    total_text_length = sum(len(text) for _, text in pages)
    if total_text_length == 0:
        raise ValueError("このPDFはテキスト抽出できませんでした（スキャンPDFの可能性）")
    
    # テキスト正規化
    normalized_pages = []
    for page_no, text in pages:
        normalized_text = normalize_pdf_text(text)
        normalized_pages.append((page_no, normalized_text))

    if total_text_length < 100:
        # 短いPDFは見出し抽出をせず、ページ単位でセクション化
        sections = []
        for page_no, text in normalized_pages:
            if text.strip():
                sections.append({
                    'file_path': pdf_path,
                    'content': text.strip(),
                    'heading': f"PAGE {page_no}",
                    'page_start': page_no,
                    'page_end': page_no,
                    'updated_at': datetime.fromtimestamp(os.path.getmtime(pdf_path) if os.path.exists(pdf_path) else 0).isoformat()
                })
        return sections
    
    # 反復行（ヘッダー/フッター）を除外
    cleaned_pages = remove_repeated_lines(normalized_pages)
    
    # 全ページのテキストを結合して行に分解
    all_lines = []
    line_to_page = {}  # 行がどのページに属するか
    
    for page_no, text in cleaned_pages:
        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                all_lines.append(line_stripped)
                line_to_page[len(all_lines) - 1] = page_no
    
    if not all_lines:
        # フォールバック: ページ単位でセクション化
        sections = []
        for page_no, text in cleaned_pages:
            if text.strip():
                sections.append({
                    'file_path': pdf_path,
                    'content': text.strip(),
                    'heading': f"PAGE {page_no}",
                    'page_start': page_no,
                    'page_end': page_no,
                    'updated_at': datetime.fromtimestamp(os.path.getmtime(pdf_path) if os.path.exists(pdf_path) else 0).isoformat()
                })
        return sections
    
    # 見出し候補をスコアリング
    heading_scores = []
    for i, line in enumerate(all_lines):
        score = score_heading_line(line)
        heading_scores.append((i, line, score))
    
    # 見出し候補が多すぎる場合は閾値を上げる
    high_score_count = sum(1 for _, _, score in heading_scores if score >= HEADING_SCORE_THRESHOLD)
    threshold = HEADING_SCORE_THRESHOLD_STRICT if high_score_count > len(all_lines) * 0.3 else HEADING_SCORE_THRESHOLD
    
    # 見出しとして採用
    headings = [
        (i, line) for i, line, score in heading_scores
        if score >= threshold
    ]
    
    # 見出しが少なすぎる場合はフォールバック
    if len(headings) < 2:
        # フォールバック: ページ単位でセクション化
        sections = []
        for page_no, text in cleaned_pages:
            if text.strip():
                sections.append({
                    'file_path': pdf_path,
                    'content': text.strip(),
                    'heading': f"PAGE {page_no}",
                    'page_start': page_no,
                    'page_end': page_no,
                    'updated_at': datetime.fromtimestamp(os.path.getmtime(pdf_path) if os.path.exists(pdf_path) else 0).isoformat()
                })
        return sections
    
    # 見出しごとにセクション分割
    sections = []
    for idx, (heading_idx, heading_text) in enumerate(headings):
        # セクションの開始行
        start_line_idx = heading_idx
        
        # セクションの終了行（次の見出しの前まで）
        if idx + 1 < len(headings):
            end_line_idx = headings[idx + 1][0]
        else:
            end_line_idx = len(all_lines)
        
        # セクション本文を取得
        section_lines = all_lines[start_line_idx:end_line_idx]
        # 見出し行を除く
        section_content = '\n'.join(section_lines[1:]) if len(section_lines) > 1 else '\n'.join(section_lines)
        
        # ページ範囲を取得
        page_start = line_to_page.get(start_line_idx, 1)
        page_end = line_to_page.get(end_line_idx - 1, page_start)
        
        if section_content.strip():
            sections.append({
                'file_path': pdf_path,
                'content': section_content.strip(),
                'heading': heading_text,
                'page_start': page_start,
                'page_end': page_end,
                'updated_at': datetime.fromtimestamp(os.path.getmtime(pdf_path) if os.path.exists(pdf_path) else 0).isoformat()
            })

    return sections


# ==================== OCR処理関数 ====================

def ocr_extract_text_from_pdf(
    pdf_path: str,
    method: str = "tesseract",
    language: str = "jpn",
    progress_callback: callable = None
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
        ImportError: 必要なライブラリがインストールされていない
        ValueError: Azure APIキーが設定されていない
        Exception: OCR処理エラー
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2imageがインストールされていません。\n"
            "インストール方法:\n"
            "  pip install pdf2image\n"
            "  また、popplerのインストールも必要です:\n"
            "    Ubuntu/Debian: sudo apt install poppler-utils\n"
            "    macOS: brew install poppler\n"
            "    Windows: https://github.com/oschwartz10612/poppler-windows/releases/"
        )

    # PDFをページ画像に変換
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        raise Exception(f"PDF画像変換エラー: {e}")

    pages_text = []
    total_pages = len(images)

    for i, image in enumerate(images, 1):
        if progress_callback:
            progress_callback(i, total_pages)

        if method == "tesseract":
            text = _ocr_with_tesseract(image, language)
        elif method == "azure":
            text = _ocr_with_azure(image)
        else:
            raise ValueError(f"Unknown OCR method: {method}")

        pages_text.append((i, text))

    return pages_text


def _ocr_with_tesseract(image, language: str = "jpn") -> str:
    """Tesseract OCRでテキスト抽出

    Args:
        image: PIL Image オブジェクト
        language: OCR言語（'jpn', 'eng'）

    Returns:
        抽出されたテキスト
    """
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseractがインストールされていません。\n"
            "インストール方法:\n"
            "  pip install pytesseract\n"
            "  また、Tesseract OCRのインストールも必要です:\n"
            "    Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-jpn\n"
            "    macOS: brew install tesseract tesseract-lang\n"
            "    Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        )

    try:
        text = pytesseract.image_to_string(image, lang=language)
        return text.strip()
    except Exception as e:
        raise Exception(f"Tesseract OCRエラー: {e}")


def _ocr_with_azure(image, endpoint: str = None, api_key: str = None) -> str:
    """Azure Document Intelligenceでテキスト抽出

    Args:
        image: PIL Image オブジェクト
        endpoint: Azure endpoint（環境変数から取得可能）
        api_key: Azure APIキー（環境変数から取得可能）

    Returns:
        抽出されたテキスト
    """
    import io

    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        raise ImportError(
            "azure-ai-formrecognizerがインストールされていません。\n"
            "インストール方法: pip install azure-ai-formrecognizer"
        )

    # 環境変数から取得
    endpoint = endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    api_key = api_key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not api_key:
        raise ValueError(
            "Azure Document IntelligenceのエンドポイントとAPIキーが設定されていません。\n"
            ".envファイルに以下を追加してください:\n"
            "  AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/\n"
            "  AZURE_DOCUMENT_INTELLIGENCE_KEY=your_api_key_here"
        )

    # Azure APIクライアント
    client = DocumentAnalysisClient(endpoint, AzureKeyCredential(api_key))

    # PIL ImageをBytesIOに変換
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # OCR実行
    try:
        poller = client.begin_analyze_document("prebuilt-read", img_bytes)
        result = poller.result()

        # テキスト抽出
        lines = []
        for page in result.pages:
            for line in page.lines:
                lines.append(line.content)

        return '\n'.join(lines)
    except Exception as e:
        raise Exception(f"Azure Document Intelligence OCRエラー: {e}")


def pdf_to_sections_with_ocr(
    pdf_path: str,
    ocr_enabled: bool = True,
    ocr_method: str = "tesseract",
    ocr_language: str = "jpn",
    progress_callback: callable = None
) -> List[Dict]:
    """PDFを見出し推定→セクション化（OCR対応版）

    Args:
        pdf_path: PDFファイルのパス
        ocr_enabled: OCR処理を有効化
        ocr_method: OCR手法（'tesseract' or 'azure'）
        ocr_language: OCR言語
        progress_callback: 進捗コールバック

    Returns:
        Documentのリスト（pdf_to_sections()と同じ形式）
    """
    from constants import OCR_MIN_TEXT_LENGTH

    # 1. pypdfでテキスト抽出を試みる
    try:
        pages = extract_pdf_pages(pdf_path)
        total_text_length = sum(len(text) for _, text in pages)

        # 2. テキストが極端に少ない場合 → OCR処理
        if total_text_length < OCR_MIN_TEXT_LENGTH and ocr_enabled:
            print(f"⚠️ テキスト抽出量が少ない（{total_text_length}文字）ため、OCR処理を実行します...")
            pages = ocr_extract_text_from_pdf(
                pdf_path,
                method=ocr_method,
                language=ocr_language,
                progress_callback=progress_callback
            )
    except Exception as e:
        if ocr_enabled:
            # pypdf失敗 → OCR処理にフォールバック
            print(f"⚠️ pypdfでのテキスト抽出失敗（{e}）。OCR処理を実行します...")
            pages = ocr_extract_text_from_pdf(
                pdf_path,
                method=ocr_method,
                language=ocr_language,
                progress_callback=progress_callback
            )
        else:
            raise e

    # 3. 既存の見出し推定→セクション化ロジック
    if not pages:
        return []

    total_text_length = sum(len(text) for _, text in pages)
    if total_text_length == 0:
        raise ValueError("このPDFはテキスト抽出できませんでした（スキャンPDFの可能性）")

    # normalize_pdf_text, remove_repeated_lines等を使用
    normalized_pages = [(p, normalize_pdf_text(t)) for p, t in pages]

    if total_text_length < 100:
        # 短いPDFは見出し抽出をせず、ページ単位でセクション化
        sections = []
        for page_no, text in normalized_pages:
            if text.strip():
                sections.append({
                    'file_path': pdf_path,
                    'content': text.strip(),
                    'heading': f"PAGE {page_no}",
                    'page_start': page_no,
                    'page_end': page_no,
                    'updated_at': datetime.fromtimestamp(os.path.getmtime(pdf_path) if os.path.exists(pdf_path) else 0).isoformat()
                })
        return sections

    cleaned_pages = remove_repeated_lines(normalized_pages)

    # 既存のpdf_to_sections()ロジックと同じ
    all_lines = []
    line_to_page = {}

    for page_no, text in cleaned_pages:
        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                all_lines.append(line_stripped)
                line_to_page[len(all_lines) - 1] = page_no

    if not all_lines:
        sections = []
        for page_no, text in cleaned_pages:
            if text.strip():
                sections.append({
                    'file_path': pdf_path,
                    'content': text.strip(),
                    'heading': f"PAGE {page_no}",
                    'page_start': page_no,
                    'page_end': page_no,
                    'updated_at': datetime.fromtimestamp(os.path.getmtime(pdf_path) if os.path.exists(pdf_path) else 0).isoformat()
                })
        return sections

    # 見出し候補をスコアリング
    heading_scores = []
    for i, line in enumerate(all_lines):
        score = score_heading_line(line)
        heading_scores.append((i, line, score))

    from constants import HEADING_SCORE_THRESHOLD, HEADING_SCORE_THRESHOLD_STRICT

    high_score_count = sum(1 for _, _, score in heading_scores if score >= HEADING_SCORE_THRESHOLD)
    threshold = HEADING_SCORE_THRESHOLD_STRICT if high_score_count > len(all_lines) * 0.3 else HEADING_SCORE_THRESHOLD

    headings = [
        (i, line) for i, line, score in heading_scores
        if score >= threshold
    ]

    if len(headings) < 2:
        sections = []
        for page_no, text in cleaned_pages:
            if text.strip():
                sections.append({
                    'file_path': pdf_path,
                    'content': text.strip(),
                    'heading': f"PAGE {page_no}",
                    'page_start': page_no,
                    'page_end': page_no,
                    'updated_at': datetime.fromtimestamp(os.path.getmtime(pdf_path) if os.path.exists(pdf_path) else 0).isoformat()
                })
        return sections

    sections = []
    for idx, (heading_idx, heading_text) in enumerate(headings):
        start_line_idx = heading_idx

        if idx + 1 < len(headings):
            end_line_idx = headings[idx + 1][0]
        else:
            end_line_idx = len(all_lines)

        section_lines = all_lines[start_line_idx:end_line_idx]
        section_content = '\n'.join(section_lines[1:]) if len(section_lines) > 1 else '\n'.join(section_lines)

        page_start = line_to_page.get(start_line_idx, 1)
        page_end = line_to_page.get(end_line_idx - 1, page_start)

        if section_content.strip():
            sections.append({
                'file_path': pdf_path,
                'content': section_content.strip(),
                'heading': heading_text,
                'page_start': page_start,
                'page_end': page_end,
                'updated_at': datetime.fromtimestamp(os.path.getmtime(pdf_path) if os.path.exists(pdf_path) else 0).isoformat()
            })

    return sections
