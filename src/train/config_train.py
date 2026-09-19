# ============================================================
# PHASE 2 NEW CONFIG — HLT-Chunking + SQLite + BM25Okapi + Numpy Search
# ============================================================
import os

# ================= 1. CẤU HÌNH THƯ MỤC =================
DEFAULT_INPUT_PATH = r"/kaggle/input/datasets/nguynquckhanh/selected-context/selected-contexts"
PROCESSED_DATA_DIR = r"/kaggle/working/train/processed_data"
EMBEDED_DATA_DIR   = r"/kaggle/working/train/embeded_data"

# ── EMBEDED_DATA_INPUT: nơi CELL 7 đọc vector ──────────────
EMBEDED_DATA_INPUT = EMBEDED_DATA_DIR   # ← CHẠY TOÀN BỘ 1 NOTEBOOK
# EMBEDED_DATA_INPUT = r"/kaggle/input/datasets/nguynquckhanh/embeded-data/embeded_data"  # ← PHASE 2 (đã upload)

QUESTIONS_PUBLIC_FILE = r"/kaggle/input/datasets/nguynquckhanh/public-test/public-official.json"

JUST_CHECK_QUERIES = None
JUST_CHECK_CONTEXT = None

# ================= 2. CHUNKING (HLT — 300 từ/chunk) =================
MAX_WORDS_PER_CHUNK = 300
OVERLAP_WORDS       = 50

# ================= 3. EMBEDDING & SEMANTIC SEARCH =================
EMBEDDING_MODEL         = "BAAI/bge-m3"
EMBEDDING_DIM           = 1024
MAX_POSITION_EMBEDDINGS = 8192
EMBEDDING_BATCH_SIZE    = 16
SUBMISSION_SEMANTIC_TOP_K = r"/kaggle/working/train/submission_semantic_topK.json"

# ================= 4. BM25 =================
SQLITE_DB_PATH        = r"/kaggle/working/train/legal_corpus.db"
BM25_PKL_PATH         = r"/kaggle/working/train/bm25_index.pkl"
SUBMISSION_BM25_TOP_K = r"/kaggle/working/train/submission_bm25_topK.json"
RETRIEVAL_TOP_K       = 500

# ================= 5. RRF =================
SUBMISSION_HYBRID_TOP_K = r"/kaggle/working/train/submission_hybrid_topK.json"
MAX_CHUNKS_PER_DOC = 50
RRF_TOP_K = 500

# ================= 6. RERANKER =================
RERANKER_MODEL      = 'BAAI/bge-reranker-v2-m3'
RERANKER_BATCH_SIZE = 4
MAX_LENGTH          = 8192
SUBMISSION_FINAL    = r"/kaggle/working/train/submission.json"