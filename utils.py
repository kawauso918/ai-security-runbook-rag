"""ユーティリティ関数"""

import re
import os
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


def chunk_by_headings(text: str, file_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """見出しベースでチャンキング"""
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
        
        # セクション内でチャンキング
        section_chunks = _chunk_text(section_text, chunk_size, overlap)
        
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


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """テキストを指定サイズでチャンキング"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # オーバーラップ処理
        if end < len(text) and overlap > 0:
            # 次のチャンクの開始位置を調整（文の途中で切れないように）
            next_start = end - overlap
            # 可能な限り文の区切りで調整
            for i in range(next_start, end):
                if text[i] in ['。', '\n', '.', '!', '?']:
                    next_start = i + 1
                    break
        
        chunks.append(chunk)
        start = end - overlap if overlap > 0 else end
        
        if start >= len(text):
            break
    
    return chunks


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
    if total_text_length < 100:
        raise ValueError("このPDFはテキスト抽出できませんでした（スキャンPDFの可能性）")
    
    # テキスト正規化
    normalized_pages = []
    for page_no, text in pages:
        normalized_text = normalize_pdf_text(text)
        normalized_pages.append((page_no, normalized_text))
    
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



