"""ガードレール処理（危険操作検知、根拠不足判定、曖昧質問検知）"""

from typing import List, Dict
from constants import DANGEROUS_KEYWORDS, DEFAULT_EVIDENCE_THRESHOLD


def detect_dangerous_operations(query: str, answer: str) -> bool:
    """危険操作検知"""
    text_to_check = (query + " " + answer).lower()
    
    for keyword in DANGEROUS_KEYWORDS:
        if keyword.lower() in text_to_check:
            return True
    
    return False


def check_insufficient_evidence(
    search_results: List[Dict],
    threshold: float = DEFAULT_EVIDENCE_THRESHOLD
) -> bool:
    """根拠不足判定"""
    if not search_results:
        return True
    
    max_score = max(result.get('score', 0.0) for result in search_results)
    return max_score < threshold


def detect_ambiguous_query(query: str) -> bool:
    """曖昧質問検知"""
    # 5文字未満は曖昧
    if len(query.strip()) < 5:
        return True
    
    # 疑問詞のみの場合は曖昧
    ambiguous_patterns = ['何', 'どう', 'なぜ', 'なんで', 'どこ', 'いつ', '誰']
    if len(query.strip()) < 10 and any(pattern in query for pattern in ambiguous_patterns):
        return True
    
    return False


def get_insufficient_evidence_response() -> str:
    """根拠不足時の応答テンプレート"""
    return """該当する手順が見つかりませんでした。

以下の点をご確認ください：
- 質問のキーワードを変えて再度お試しください
- 手順書に該当する内容が含まれているか確認してください"""


def get_ambiguous_query_response() -> str:
    """曖昧質問時の応答テンプレート"""
    return """質問が曖昧な可能性があります。以下の情報を追加していただけると、より正確な回答ができます：

- 対象システムや環境
- 具体的な操作や手順名
- エラーメッセージや状況

例：「ログインできない」→「Active Directoryへのログインができない。エラーコード: 0x80090308」"""

