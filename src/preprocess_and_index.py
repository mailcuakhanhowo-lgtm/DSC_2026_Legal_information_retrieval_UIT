import os
import sys
import re
import glob
import json
import time
import pickle
import sqlite3
import logging
import unicodedata
from pathlib import Path

# Them src vao sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

# Thiet lap Logging
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
config.INDICES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(config.LOG_FILE_PATH),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# =========================================================================
# CHUNKING LOGIC (BAO TON NGUYEN VEN 100% THEO YEU CAU)
# =========================================================================
def filter_and_clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(
        r'(Nơi nhận:|KT\. BỘ TRƯỞNG).*?(?=(?:\n\s*(?:QUY CHẾ|PHỤ LỤC|Chương \d+|Điều \d+\.))|\Z)', 
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def smart_legal_chunker(text: str, max_words: int = 600, overlap_words: int = 50) -> list:
    if not text or not text.strip():
        return []
    pattern = r'(?=(?:\n|^)\s*(?:' \
              r'Điều\s+\d+[a-zA-Z]*[\.\:]|' \
              r'(?:Chương|MỤC|Mục|PHỤ LỤC|Phụ lục)\s+(?:[IVXLCDM\d]+|[A-Z])|' \
              r'\d+(?:\.\d+)*[\.\)]\s*|' \
              r'[IVXLCDM]+\.\s+|' \
              r'[a-zà-ỹ]\)' \
              r'))'
    sections = re.split(pattern, text)
    final_chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        words = sec.split(' ')
        word_count = len(words)
        if word_count <= max_words:
            final_chunks.append(sec)
        else:
            step = max_words - overlap_words
            if step <= 0:
                step = max_words
            for i in range(0, word_count, step):
                chunk_words = words[i:i + max_words]
                chunk_text = " ".join(chunk_words).strip()
                if chunk_text:
                    final_chunks.append(chunk_text)
    return final_chunks

# =========================================================================
# TACH TU TIENG VIET CHO BM25
# =========================================================================
def segment_text(text: str) -> str:
    if HAS_PYVI:
        try:
            return ViTokenizer.tokenize(text)
        except Exception:
            return text
    return text

def tokenize_for_bm25(text: str) -> list:
    segmented = segment_text(text)
    clean_str = re.sub(r'[^\w\s_]', ' ', segmented.lower())
    return [w for w in clean_str.split() if len(w) > 1]

# =========================================================================
# PHASE 1: OFFLINE PREPROCESSING & INDEXING PIPELINE
# =========================================================================
def main():
    start_time = time.time()
    logging.info("BAT DAU GIAI DOAN 1: OFFLINE PREPROCESSING & INDEXING (BM25 + SQLITE)")

    corpus_dir = config.SELECTED_CONTEXTS_DIR
    if not corpus_dir.exists():
        logging.error(f"Khong tim thay thu muc kho van ban: {corpus_dir}")
        sys.exit(1)

    json_files = glob.glob(os.path.join(corpus_dir, "*.json"))
    total_files_count = len(json_files)
    logging.info(f"Da tim thay {total_files_count} file JSON trong kho ngu lieu.")

    # 1. KET NOI VA TAO BANG SQLITE
    db_path = config.DB_PATH
    if db_path.exists():
        try:
            os.remove(db_path)
        except Exception:
            pass
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            title TEXT,
            markdown_content TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks (doc_id)")
    conn.commit()

    # 2. XU LY VAN BAN & NAP SQLITE + BM25
    sqlite_batch = []
    bm25_corpus_tokens = []
    chunk_ids = []
    chunk_doc_map = {}

    total_docs = 0
    total_chunks = 0

    for idx, file_path in enumerate(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
        except Exception as e:
            logging.warning(f"Loi doc file {file_path}: {e}")
            continue

        doc_id = str(doc.get("id", "")).strip()
        title = doc.get("name", doc.get("title", "")).strip()
        passage = doc.get("passage", "").strip()

        if not doc_id or not passage:
            continue

        # Clean & Chunking
        cleaned_text = filter_and_clean_text(passage)
        raw_chunks = smart_legal_chunker(
            cleaned_text, 
            max_words=config.MAX_WORDS_PER_CHUNK, 
            overlap_words=config.OVERLAP_WORDS
        )

        # Header Injection
        for i, chunk_raw in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_{i}"
            markdown_content = f"# {title}\n\n{chunk_raw}" if title else chunk_raw

            sqlite_batch.append((chunk_id, doc_id, title, markdown_content))
            
            # Chuan bi Tokens cho BM25
            tokens = tokenize_for_bm25(markdown_content)
            bm25_corpus_tokens.append(tokens)
            chunk_ids.append(chunk_id)
            chunk_doc_map[chunk_id] = doc_id
            
            total_chunks += 1
            
        total_docs += 1

        if (idx + 1) % 1000 == 0 or (idx + 1) == total_files_count:
            logging.info(f"Tien do: Da xu ly {idx + 1}/{total_files_count} file ({total_chunks} chunks)...")

    # Nap du lieu hang loat vao SQLite (executemany)
    logging.info(f"Dang luu {total_chunks} chunks vao SQLite database ({db_path})...")
    cursor.executemany(
        "INSERT INTO chunks (chunk_id, doc_id, title, markdown_content) VALUES (?, ?, ?, ?)",
        sqlite_batch
    )
    conn.commit()
    conn.close()

    # 3. HUAN LUYEN BM25OKAPI VA LUU PICKLE
    logging.info(f"Dang huan luyen mo hinh BM25Okapi tren {total_chunks} chunks...")
    try:
        from rank_bm25 import BM25Okapi
        bm25_model = BM25Okapi(bm25_corpus_tokens)
    except ImportError:
        logging.error("Thu vien 'rank-bm25' chua duoc cai dat. Vui long go: pip install rank-bm25")
        sys.exit(1)

    bm25_payload = {
        "bm25": bm25_model,
        "chunk_ids": chunk_ids,
        "chunk_doc_map": chunk_doc_map,
        "total_chunks": total_chunks,
        "total_docs": total_docs
    }

    bm25_path = config.BM25_INDEX_PATH
    logging.info(f"Dang luu BM25 Index vao: {bm25_path}")
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed_time = time.time() - start_time
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    bm25_size_mb = os.path.getsize(bm25_path) / (1024 * 1024)

    logging.info("=" * 60)
    logging.info("HOAN THANH TIEN XU LY VA BUILD INDEX THANH CONG!")
    logging.info(f"Tong so van ban xu ly: {total_docs}")
    logging.info(f"Tong so chunks da tao: {total_chunks}")
    logging.info(f"Dung luong SQLite DB: {db_size_mb:.2f} MB")
    logging.info(f"Dung luong BM25 Pickle Index: {bm25_size_mb:.2f} MB")
    logging.info(f"Tong thoi gian thuc thi: {elapsed_time:.2f} giay")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
