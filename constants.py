"""定数定義"""

# デフォルト設定
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_K = 4
DEFAULT_BM25_WEIGHT = 0.6
DEFAULT_VECTOR_WEIGHT = 0.4
DEFAULT_EVIDENCE_THRESHOLD = 0.5
MAX_CONVERSATION_HISTORY = 5

# 危険操作キーワード
DANGEROUS_KEYWORDS = [
    '削除', 'delete', 'rm -rf', 'drop', 'truncate',
    '証跡', '消去', 'フォーマット', 'format', '初期化', 'reset', 'purge',
    '無効化', 'disable', '停止', 'shutdown', 'kill',
    'clear', 'remove', 'erase', 'wipe'
]

# LLM設定
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"  # Judge用モデル

# パス
DEFAULT_DATA_FOLDER = "./data"
CHROMA_DB_PATH = "./chroma_db"
LOGS_FOLDER = "./logs"
EVAL_FOLDER = "./eval"

# トークン価格（USD per 1K tokens）
TOKEN_PRICES = {
    "gpt-4o-mini": {
        "prompt": 0.150,
        "completion": 0.600
    }
}

# Judge評価設定
JUDGE_EVALUATION_CRITERIA = [
    "根拠性",
    "正確性",
    "網羅性",
    "安全性",
    "引用明示",
    "簡潔性"
]
JUDGE_PASS_THRESHOLD = 70.0  # 合格ライン（平均70点以上）
JUDGE_PASS_RATE = 0.7  # MVP合格率（10問中7問以上）

# PDF処理設定
PDF_MAX_HEADING_LEN = 60  # 見出し候補の最大文字数
PDF_MIN_HEADING_LEN = 4   # 見出し候補の最小文字数
HEADING_SCORE_THRESHOLD = 3  # 見出しとして採用する最小スコア
HEADING_SCORE_THRESHOLD_STRICT = 5  # 連続見出しが多い場合の厳格な閾値

# 見出しパターン（正規表現）
HEADING_PATTERNS = [
    r'^第[0-9一二三四五六七八九十]+章',  # 第1章、第二章など
    r'^\d+(\.\d+)*\s+',  # 1, 1.2, 2.3.4 など
    r'^\(\d+\)',  # (1)
    r'^（\d+）',  # （1）
    r'^\d+\)',  # 1)
    r'^\d+\-\d+',  # 1-1
    r'^【.+】',  # 【見出し】
    r'^\[.+\]',  # [見出し]
    r'^＜.+＞',  # ＜見出し＞
]

# 見出しキーワード（加点用）
HEADING_KEYWORDS = [
    "手順", "概要", "目的", "対応", "確認", "影響",
    "エスカレーション", "復旧", "封じ込め", "一次対応",
    "二次対応", "判断基準", "初動", "調査", "報告",
    "対策", "予防", "検知", "通知", "連絡"
]


