"""ログ記録処理"""

import json
import os
from datetime import datetime
from typing import List, Dict
from pathlib import Path

from constants import LOGS_FOLDER


def log_query(
    query: str,
    search_results: List[Dict],
    answer: str,
    processing_time: float,
    token_usage: Dict,
    cost: float,
    search_config: Dict,
    flags: Dict,
    session_id: str = None
) -> None:
    """クエリログをJSONL形式で記録"""
    
    # ログフォルダが存在しない場合は作成
    log_dir = Path(LOGS_FOLDER)
    log_dir.mkdir(exist_ok=True)
    
    # 日付別ファイル名
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"query_{today}.jsonl"
    
    # 検索結果をログ用に整形
    log_search_results = []
    for result in search_results:
        log_search_results.append({
            'chunk_id': result.get('chunk_id', ''),
            'score': result.get('score', 0.0),
            'bm25_score': result.get('bm25_score', 0.0),
            'vector_score': result.get('vector_score', 0.0),
            'file': result.get('file', ''),
            'heading': result.get('heading', ''),
            'excerpt': result.get('text', '')[:200]  # 最初の200文字
        })
    
    # トップスコア取得
    top_score = max([r.get('score', 0.0) for r in search_results]) if search_results else 0.0

    # ログエントリを作成
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id or 'unknown',
        'query': query,
        'mode': 'hybrid',  # 検索モード
        'k': search_config.get('k', 0),
        'weights': {
            'bm25': search_config.get('bm25_weight', 0.0),
            'vector': search_config.get('vector_weight', 0.0)
        },
        'top_score': top_score,
        'sources': log_search_results,
        'answer': answer,
        'latency_ms': int(processing_time * 1000),  # ミリ秒に変換
        'tokens': token_usage,
        'cost': cost,
        'flags': flags
    }
    
    # JSONL形式で追記
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error writing log: {e}")

