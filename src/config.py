import os
import sys
from pathlib import Path

# Đảm bảo in UTF-8 trên Windows PowerShell
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. QUẢN LÝ ĐƯỜNG DẪN DỰ ÁN CỐ ĐỊNH VÀ ĐỘNG 
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Môi trường Docker của BTC
BTC_DOCKER_INPUT = Path("/app/input")
IS_BTC_DOCKER = BTC_DOCKER_INPUT.exists()

if IS_BTC_DOCKER:
    REF_DIR = Path("/app/input/ref")
    RES_DIR = Path("/app/input/res")
    OUTPUT_DIR = Path("/app/output")
    
    TRAIN_PATH = REF_DIR / "train.json"
    PUBLIC_TEST_PATH = RES_DIR / "public-official.json"
    SUBMISSION_PATH = OUTPUT_DIR / "submission.json"
    
    DATA_RAW_DIR = REF_DIR
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    INDICES_DIR = PROJECT_ROOT / "indices"
    LOGS_DIR = PROJECT_ROOT / "logs"
else:
    DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    INDICES_DIR = PROJECT_ROOT / "indices"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    TRAIN_PATH = DATA_RAW_DIR / "train.json"
    WARMUP_PATH = DATA_RAW_DIR / "warmup.json"
    PUBLIC_TEST_PATH = DATA_RAW_DIR / "public-official.json"
    SELECTED_CONTEXTS_DIR = DATA_RAW_DIR / "selected-contexts"
    SUBMISSION_PATH = PROJECT_ROOT / "submission.json"

# Đường dẫn DB SQLite & BM25 Index
DB_PATH = INDICES_DIR / "legal_corpus.db"
BM25_INDEX_PATH = INDICES_DIR / "bm25_index.pkl"
PROCESSED_CORPUS_JSONL = DATA_PROCESSED_DIR / "processed_corpus.jsonl"
LOG_FILE_PATH = LOGS_DIR / "pipeline.log"

# 2. HẰNG SỐ CẤU HÌNH PIPELINE & BM25
MAX_WORDS_PER_CHUNK = 600
OVERLAP_WORDS = 50

TOP_K_CANDIDATE_CHUNKS = 50   # Lấy Top 50 chunks từ BM25 để Max Pooling
TOP_K_SUBMISSION_DOCS = 5     # Ràng buộc tối đa 5 document_id của BTC
