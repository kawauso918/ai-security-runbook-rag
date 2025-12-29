"""ガードレール処理（危険操作検知、根拠不足判定、曖昧質問検知）"""

from typing import List, Dict, Tuple, Optional
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
) -> Tuple[bool, float]:
    """根拠不足判定
    
    Args:
        search_results: 検索結果のリスト
        threshold: 閾値（デフォルト0.5）
    
    Returns:
        (判定結果, 最高スコア) のタプル
    """
    if not search_results:
        return True, 0.0
    
    top_score = max(result.get('score', 0.0) for result in search_results)
    return top_score < threshold, top_score


def detect_ambiguous_query(query: str) -> bool:
    """曖昧質問検知"""
    query_stripped = query.strip()
    
    # 5文字未満は曖昧
    if len(query_stripped) < 5:
        return True
    
    # 疑問詞のみ、または文脈が不足している場合
    ambiguous_patterns = ['これ', 'それ', 'あれ', 'どう', '何', 'なぜ', 'なんで', 'どこ', 'いつ', '誰']
    
    # 短い質問で疑問詞のみの場合は曖昧
    if len(query_stripped) < 15:
        # 疑問詞が含まれ、かつ具体的なキーワードが少ない
        has_ambiguous_word = any(pattern in query_stripped for pattern in ambiguous_patterns)
        # 具体的なキーワード（英数字、固有名詞など）が少ない
        has_specific_keywords = any(
            char.isalnum() and len(word) > 3 
            for word in query_stripped.split() 
            for char in word
        )
        
        if has_ambiguous_word and not has_specific_keywords:
            return True
    
    return False


def get_insufficient_evidence_response() -> str:
    """根拠不足時の応答テンプレート"""
    return "該当する手順が見つかりませんでした。"


def get_ambiguous_query_response() -> str:
    """曖昧質問時の応答テンプレート"""
    return """質問の情報が不足しています。以下の情報を追加していただけると、より正確な回答ができます：

**必要な情報：**
- アラート種別（例: マルウェア検知、不正アクセス、データ漏洩など）
- アラートID（例: ALERT-2024-001）
- 影響範囲（例: 特定のサーバー、ネットワーク全体など）
- 発生時刻（例: 2024-12-29 14:00）
- 実施済み対応（例: 端末の隔離、ログの確認など）

**質問例：**
「マルウェア検知アラート（ALERT-2024-001）が発生しました。影響範囲はWebサーバー3台で、発生時刻は2024-12-29 14:00です。現在、該当サーバーをネットワークから隔離済みです。次の対応手順を教えてください。」"""


def get_dangerous_operation_warning() -> str:
    """危険操作警告メッセージ"""
    return "⚠️ この操作は承認・確認が必要です"


def get_evidence_preservation_note() -> str:
    """証跡保全の促しメッセージ"""
    return "\n\n**重要**: この操作を実施する前に、必ず証跡（ログ、スクリーンショット、関連ファイル）を保全してください。"


def apply_guardrails(
    query: str,
    search_results: List[Dict],
    answer: Optional[str] = None
) -> Dict:
    """ガードレールを適用
    
    Args:
        query: ユーザーの質問
        search_results: 検索結果のリスト
        answer: LLMが生成した回答（危険操作検知に使用、Noneの場合は質問のみチェック）
    
    Returns:
        {
            'should_respond': bool,  # 回答を返すか（Falseの場合は固定文を返す）
            'answer': str,  # 返すべき回答
            'citations': List[Dict],  # 引用（根拠不足の場合は空）
            'warning_reason': Optional[str],  # 警告理由（'insufficient_evidence', 'dangerous_operation', 'ambiguous_query', None）
            'flags': {
                'insufficient_evidence': bool,
                'dangerous_operation': bool,
                'ambiguous_query': bool
            },
            'top_score': float  # 検索結果の最高スコア
        }
    """
    # 1. 根拠不足判定（最優先）
    insufficient_evidence, top_score = check_insufficient_evidence(search_results)
    
    if insufficient_evidence:
        return {
            'should_respond': False,
            'answer': get_insufficient_evidence_response(),
            'citations': [],  # 根拠不足の場合は引用を出さない
            'warning_reason': 'insufficient_evidence',
            'flags': {
                'insufficient_evidence': True,
                'dangerous_operation': False,
                'ambiguous_query': False
            },
            'top_score': top_score
        }
    
    # 2. 曖昧質問検知（LLM呼び出し前に判定）
    ambiguous = detect_ambiguous_query(query)
    
    if ambiguous:
        return {
            'should_respond': False,
            'answer': get_ambiguous_query_response(),
            'citations': [],  # 曖昧質問の場合は引用を出さない
            'warning_reason': 'ambiguous_query',
            'flags': {
                'insufficient_evidence': False,
                'dangerous_operation': False,
                'ambiguous_query': True
            },
            'top_score': top_score
        }
    
    # 3. 危険操作検知（回答生成後にもチェック）
    dangerous = False
    if answer:
        dangerous = detect_dangerous_operations(query, answer)
    
    # 危険操作が検知された場合、警告を追加
    final_answer = answer if answer else ""
    if dangerous:
        warning = get_dangerous_operation_warning()
        evidence_note = get_evidence_preservation_note()
        
        # 回答冒頭に警告、末尾に証跡保全の促しを追加
        final_answer = f"{warning}\n\n{final_answer}{evidence_note}"
    
    return {
        'should_respond': True,
        'answer': final_answer,
        'citations': search_results,  # 正常な場合は引用を返す
        'warning_reason': 'dangerous_operation' if dangerous else None,
        'flags': {
            'insufficient_evidence': False,
            'dangerous_operation': dangerous,
            'ambiguous_query': False
        },
        'top_score': top_score
    }
