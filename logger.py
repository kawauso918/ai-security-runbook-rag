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
    warning_reason: str = None,
    top_score: float = None,
    session_id: str = None
) -> None:
    """クエリログをJSONL形式で記録
    
    Args:
        query: ユーザーの質問
        search_results: 検索結果のリスト
        answer: 生成された回答
        processing_time: 処理時間（秒）
        token_usage: トークン使用量
        cost: コスト（USD）
        search_config: 検索設定
        flags: フラグ（insufficient_evidence, dangerous_operation, ambiguous_query）
        warning_reason: 警告理由（'insufficient_evidence', 'dangerous_operation', 'ambiguous_query', None）
        top_score: 検索結果の最高スコア
        session_id: セッションID
    """
    
    # ログフォルダが存在しない場合は作成
    log_dir = Path(LOGS_FOLDER)
    log_dir.mkdir(exist_ok=True)
    
    # 日付別ファイル名
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"query_{today}.jsonl"
    
    # 検索結果をログ用に整形
    log_search_results = []
    for result in search_results:
        log_result = {
            'chunk_id': result.get('chunk_id', ''),
            'score': result.get('score', 0.0),
            'bm25_score': result.get('bm25_score', 0.0),
            'vector_score': result.get('vector_score', 0.0),
            'file': result.get('file', ''),
            'heading': result.get('heading', ''),
            'excerpt': result.get('text', '')[:200]  # 最初の200文字
        }
        # Re-rankingスコアがある場合は記録
        if 'rerank_score' in result:
            log_result['rerank_score'] = result.get('rerank_score', 0.0)
        log_search_results.append(log_result)
    
    # トップスコア取得（引数で指定されていない場合は計算）
    if top_score is None:
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
        'rerank': {
            'enabled': search_config.get('rerank_enabled', False),
            'method': search_config.get('rerank_method', 'none')
        },
        'top_score': top_score,
        'sources': log_search_results,
        'answer': answer,
        'latency_ms': int(processing_time * 1000),  # ミリ秒に変換
        'tokens': token_usage,
        'cost': cost,
        'flags': flags,
        'warning_reason': warning_reason  # 警告理由を追加
    }
    
    # JSONL形式で追記
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error writing log: {e}")


