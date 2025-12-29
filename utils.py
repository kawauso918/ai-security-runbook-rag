"""ユーティリティ関数"""

import re
import os
from datetime import datetime
from typing import List, Dict, Tuple
from sudachipy import dictionary, tokenizer


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



