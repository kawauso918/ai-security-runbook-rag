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


