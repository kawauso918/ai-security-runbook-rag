"""定数定義"""

# デフォルト設定
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_K = 4
DEFAULT_BM25_WEIGHT = 0.6
DEFAULT_VECTOR_WEIGHT = 0.4
DEFAULT_EVIDENCE_THRESHOLD = 0.5
MAX_CONVERSATION_HISTORY = 5
DEFAULT_EXTERNAL_SEARCH_ENABLED = True
DEFAULT_EXTERNAL_SEARCH_MAX_RESULTS = 5
DEFAULT_EXTERNAL_SEARCH_TIMEOUT_SEC = 8

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

# === 適応的チャンキング設定 ===
ADAPTIVE_CHUNKING_ENABLED = True

# ドキュメントタイプ別基本チャンクサイズ
CHUNK_SIZE_PDF = 800
CHUNK_SIZE_MARKDOWN = 600
CHUNK_SIZE_TEXT = 400

# チャンクサイズ範囲
CHUNK_SIZE_MIN = 300
CHUNK_SIZE_MAX = 1000

# コンテンツ密度閾値
DENSITY_LOW_THRESHOLD = 0.3   # 疎なコンテンツ（キーワード少）
DENSITY_HIGH_THRESHOLD = 0.7  # 密なコンテンツ（キーワード多）

# 密度による調整倍率
DENSITY_LOW_MULTIPLIER = 1.3   # 疎なコンテンツは大きめ
DENSITY_HIGH_MULTIPLIER = 0.8  # 密なコンテンツは小さめ

# セマンティック境界の重み
PARAGRAPH_BOUNDARY_WEIGHT = 2.0  # 段落区切り優先
SENTENCE_BOUNDARY_WEIGHT = 1.5   # 文区切り優先

# === Re-ranking設定 ===
RERANK_ENABLED = False  # デフォルトではRe-rankingを無効化（オプトイン）
RERANK_METHOD = "cohere"  # 'cohere', 'llm', 'none'
RERANK_LLM_MODEL = "gpt-4o-mini"  # LLMベースReranking用モデル
RERANK_TOP_K_BEFORE = 10  # Re-ranking前に取得する結果数（k' > k）

# === OCR設定 ===
OCR_ENABLED = False  # デフォルトでOCRを無効化（オプトイン）
OCR_METHOD = "tesseract"  # 'tesseract' or 'azure'
OCR_LANGUAGE = "jpn"  # 日本語
OCR_MIN_TEXT_LENGTH = 100  # この文字数未満の場合OCR処理を試みる
