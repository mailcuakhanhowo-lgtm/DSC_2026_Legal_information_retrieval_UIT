# KAGGLE PHASE 2 — BỎ QUA EMBEDDING + MILVUS, TÌM KIẾM TRỰC TIẾP BẰNG NUMPY

> **Điều kiện:** Sếp đã upload 2 Dataset lên Kaggle:
> 1. **embeded-data**: Chứa 16 file `embeddings_part_*.npy` + 16 file `metadata_part_*.json`
> 2. **public-test**: Chứa file `public-official.json`
> 3. Dataset **selected-context** vẫn giữ nguyên từ lần trước

> ✅ **Không cần Milvus!** Tìm kiếm vector trực tiếp bằng numpy (dot product) — cùng độ chính xác, tiết kiệm ~20 phút build DB.


> **Trước khi chạy:** Chạy cell debug dưới đây để tìm đường dẫn thực tế rồi điền vào CELL 1.
> ```python
> import os
> def tree(path, prefix="", depth=3):
>     if depth == 0: return
>     try: items = os.listdir(path)
>     except: return
>     for i, item in enumerate(items[:5]):
>         full = os.path.join(path, item)
>         print(f"{prefix}{'└──' if i==len(items)-1 else '├──'} {item}")
>         if os.path.isdir(full):
>             tree(full, prefix + ("    " if i==len(items)-1 else "│   "), depth-1)
> tree("/kaggle/input/datasets/nguynquckhanh")
> ```

---

### CELL 0: CÀI ĐẶT THƯ VIỆN
```python
# rank-bm25: thêm mới để dùng BM25Okapi (tìm kiếm từ khóa chính xác hơn SQLite FTS5)
# pyvi: tách từ tiếng Việt cho BM25 (đã có sẵn nhưng nâng version)
!pip install -q -U accelerate>=1.14.0 bitsandbytes>=0.50.1 flagembedding>=1.3.0 pyvi>=0.1.1 "sentence-transformers>=6.0.0" "transformers==5.15.0" rank-bm25
print("✅ Cài đặt xong!")
```

---

### CELL 1: CẤU HÌNH TRUNG TÂM
```python
%%writefile config_train.py
# ============================================================
# PHASE 2 NEW CONFIG — HLT-Chunking + SQLite + BM25Okapi + Numpy Search
# ============================================================
import os

# ================= 1. CẤU HÌNH THƯ MỤC =================
DEFAULT_INPUT_PATH = r"/kaggle/input/datasets/nguynquckhanh/selected-context/selected-contexts"
PROCESSED_DATA_DIR = r"/kaggle/working/train/processed_data"
EMBEDED_DATA_DIR   = r"/kaggle/working/train/embeded_data"

# ── EMBEDED_DATA_INPUT: nơi CELL 7 đọc vector ──────────────
# Chạy TOÀN BỘ 1 notebook (CELL 4 vừa tạo ra) → dùng EMBEDED_DATA_DIR
# Chạy PHASE 2 (đã upload .npy lên Dataset)    → dùng path Dataset bên dưới
EMBEDED_DATA_INPUT = EMBEDED_DATA_DIR   # ← CHẠY TOÀN BỘ 1 NOTEBOOK
# EMBEDED_DATA_INPUT = r"/kaggle/input/datasets/nguynquckhanh/embeded-data/embeded_data"  # ← PHASE 2 (đã upload)

QUESTIONS_PUBLIC_FILE = r"/kaggle/input/datasets/nguynquckhanh/public-test/public-official.json"

JUST_CHECK_QUERIES = None
JUST_CHECK_CONTEXT = None

# ================= 2. CHUNKING (HLT — 300 từ/chunk) =================
MAX_WORDS_PER_CHUNK = 300   # ↓ từ 6000 — khớp với giới hạn Embedding model
OVERLAP_WORDS       = 50

# ================= 3. LLM =================
LLM_MODEL_NAME     = "vilm/vinallama-2.7b-chat"
LLM_CACHE_DIR      = r"/kaggle/working/train/local_llm_data/.hf_cache"
LLM_INPUT_QUERIES  = QUESTIONS_PUBLIC_FILE
LLM_OUTPUT_QUERIES = r"/kaggle/working/train/rewritten_queries.json"

LLM_SYSTEM_PROMPT = (
    "Bạn là chuyên gia trích xuất thông tin pháp lý. Nhiệm vụ của bạn là đọc câu hỏi và trả về ĐÚNG MỘT chuỗi JSON.\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    '1. "cau_hoi_phap_ly": Tóm tắt ý chính của câu hỏi thành một cụm danh từ hoặc câu khẳng định pháp lý ngắn gọn.\n'
    '2. "tu_khoa": Mảng chứa các từ khóa quan trọng nhất có trong câu (từ 2 đến 5 từ).\n'
    "- KHÔNG dùng lại nội dung của phần Ví dụ.\n"
    "- TUYỆT ĐỐI KHÔNG tự bịa thêm số hiệu Luật, Nghị định, Điều, Khoản nếu câu hỏi gốc không nhắc đến.\n\n"
    "--- VÍ DỤ MẪU ---\n"
    'Người dùng: "làm sổ đỏ lần đầu mất bao nhiêu tiền"\n'
    'Đầu ra: {"cau_hoi_phap_ly": "Quy định về lệ phí cấp giấy chứng nhận quyền sử dụng đất lần đầu", "tu_khoa": ["lệ phí", "cấp giấy chứng nhận", "quyền sử dụng đất", "lần đầu"]}\n'
    "--- HẾT VÍ DỤ ---\n\n"
    "Bắt đầu xử lý câu hỏi sau (Chỉ xuất JSON):"
)

# ================= 4. EMBEDDING & SEMANTIC SEARCH =================
EMBEDDING_MODEL         = "BAAI/bge-m3"
EMBEDDING_DIM           = 1024
MAX_POSITION_EMBEDDINGS = 8192   # ↑ BGE-M3 hỗ trợ tối đa 8192 token
EMBEDDING_BATCH_SIZE    = 16
SUBMISSION_SEMANTIC_TOP_K = r"/kaggle/working/train/submission_semantic_topK.json"

# ================= 5. BM25 (rank-bm25 pkl — thay thế SQLite FTS5) =================
SQLITE_DB_PATH        = r"/kaggle/working/train/legal_corpus.db"   # corpus lưu chunk text
BM25_PKL_PATH         = r"/kaggle/working/train/bm25_index.pkl"    # BM25Okapi index
SUBMISSION_BM25_TOP_K = r"/kaggle/working/train/submission_bm25_topK.json"
RETRIEVAL_TOP_K       = 500

# ================= 6. RRF =================
SUBMISSION_HYBRID_TOP_K = r"/kaggle/working/train/submission_hybrid_topK.json"
MAX_CHUNKS_PER_DOC = 50
RRF_TOP_K = 500

# ================= 7. RERANKER =================
RERANKER_MODEL      = 'BAAI/bge-reranker-v2-m3'
RERANKER_BATCH_SIZE = 4
MAX_LENGTH          = 8192
SUBMISSION_FINAL    = r"/kaggle/working/train/submission.json"
```

---

#### CELL 2: HLT-CHUNKING → SQLITE (~60–70 phút)
```python
%%writefile preprocess_train.py
# ============================================================
# CELL 2: HLT-CHUNKING V2 + SQLITE STREAMING
# Thuật toán cắt 4 tầng: Chương > Mục > Điều > Khoản > Điểm
# Lưu ra SQLite (legal_corpus.db) thay vì thư mục .jsonl
# ============================================================
import os, re, gc, json, glob, time, sqlite3, unicodedata
from tqdm import tqdm
import config_train as config

# ── 1. LÀM SẠCH VĂN BẢN ────────────────────────────────────
def clean_legal_text(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'(Nơi nhận:|KT\. BỘ TRƯỞNG).*?(?=(?:\n\s*(?:QUY CHẾ|PHỤ LỤC|Chương \d+|Điều \d+\.))|\\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'(?<![\.:\;!?])\s*\n\s*(?=[A-Za-zà-ỹ0-9])', ' ', text)
    text = re.sub(r'(?<=[a-zà-ỹ0-9\.:\;!?])\s+(Điều\s+\d+\.?\s+[A-ZÀ-Ỹ])',   r'\n\n\1', text)
    text = re.sub(r'(?<=[a-zà-ỹ0-9\.:\;!?])\s+(Chương\s+[IVXLCDM\d]+\.?\s+[A-ZÀ-Ỹ])', r'\n\n\1', text)
    text = re.sub(r'(?<=[a-zà-ỹ0-9\.:\;!?])\s+(Mục\s+\d+\.?\s+[A-ZÀ-Ỹ])',     r'\n\n\1', text)
    text = re.sub(r'(?<!Điều)(?<!Chương)(?<!Mục)(?<!Phần)(?<!Khoản)(?<!Điểm)(?<=[a-zà-ỹ0-9\.:\;!?])\s+(\d+\.\s+[A-ZÀ-Ỹ])', r'\n\n\1', text)
    text = re.sub(r'(?<![Đđ]iểm)(?<![Kk]hoản)(?<=[a-zà-ỹ0-9\.:\;!?])\s+([a-zà-ỹ]\)\s+[A-ZÀ-Ỹa-zà-ỹ])', r'\n\n\1', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ── 2. HLT-CHUNKER (4 tầng phân cấp) ───────────────────────
def hlt_legal_chunker(text: str, max_words: int = 300, overlap_words: int = 50) -> list:
    if not text or not text.strip(): return []
    lines = text.split('\n')
    final_chunks = []
    current_chapter = current_section = current_article = current_clause_lead = ""
    current_chunk = ""

    def flush_chunk():
        nonlocal current_chunk
        if current_chunk:
            final_chunks.append(current_chunk.strip())
            current_chunk = ""

    def get_full_prefix():
        parts = [p for p in [current_chapter, current_section, current_article, current_clause_lead] if p]
        return " > ".join(parts) + "\n" if parts else ""

    for line in lines:
        line = line.strip()
        if not line: continue

        is_chapter = re.match(r'^(Chương\s+[IVXLCDM\d]+|PHẦN\s+[IVXLCDM\d]+)', line, re.IGNORECASE)
        is_section = re.match(r'^(Mục\s+\d+|PHỤ LỤC)',                           line, re.IGNORECASE)
        is_article = re.match(r'^(Điều\s+\d+)',                                   line, re.IGNORECASE)
        is_clause  = re.match(r'^\d+\.\s+', line)
        is_point   = re.match(r'^([a-zà-ỹ]\)|-|\+)\s+', line)

        if is_chapter:
            flush_chunk()
            current_chapter = line
            current_section = current_article = current_clause_lead = ""
        elif is_section:
            flush_chunk()
            current_section = line
            current_article = current_clause_lead = ""
        elif is_article:
            flush_chunk()
            current_article = line
            current_clause_lead = ""
            prefix = ""
            if current_chapter: prefix += f"{current_chapter}\n"
            if current_section: prefix += f"{current_section}\n"
            current_chunk = f"{prefix}{line}"
        elif is_clause:
            flush_chunk()
            current_clause_lead = line
            current_chunk = f"{get_full_prefix()}{line}"
        elif is_point:
            flush_chunk()
            current_chunk = f"{get_full_prefix()}{line}"
        else:
            if line.endswith(':'): current_clause_lead = line
            if not current_chunk:  current_chunk = f"{get_full_prefix()}{line}"
            else:                  current_chunk += " " + line

    flush_chunk()

    final_limited = []
    for chunk in final_chunks:
        words = chunk.split()
        if len(words) <= max_words:
            final_limited.append(chunk)
        else:
            step = max(1, max_words - overlap_words)
            for i in range(0, len(words), step):
                c = " ".join(words[i:i + max_words]).strip()
                if c: final_limited.append(c)
    return final_limited

# ── 3. MAIN ─────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("CELL 2: HLT-CHUNKING → SQLITE (STREAMING MEMORY-SAFE)")
    print("=" * 70)

    input_dir = config.DEFAULT_INPUT_PATH
    db_path   = config.SQLITE_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    json_files = list(glob.glob(os.path.join(input_dir, "*.json")))
    if hasattr(config, 'JUST_CHECK_CONTEXT') and config.JUST_CHECK_CONTEXT is not None:
        json_files = json_files[:config.JUST_CHECK_CONTEXT]
    total_files = len(json_files)
    print(f"Tổng số file văn bản: {total_files:,}")

    # Khởi tạo SQLite
    if os.path.exists(db_path): os.remove(db_path)
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id         TEXT PRIMARY KEY,
            doc_id           TEXT,
            title            TEXT,
            markdown_content TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks (doc_id)")
    conn.commit()

    BATCH_SIZE   = 5000
    sqlite_batch = []
    total_chunks = 0
    t0 = time.time()

    for idx, file_path in enumerate(tqdm(json_files, desc="HLT-Chunking")):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
        except:
            continue

        doc_id  = str(doc.get("id", "")).strip()
        title   = doc.get("name", doc.get("title", "")).strip()
        passage = doc.get("passage", "").strip()
        if not doc_id: continue

        cleaned     = clean_legal_text(passage) if passage else f"Văn bản pháp luật: {title}"
        title_words = title.split()
        short_title = " ".join(title_words[:40]) + "..." if len(title_words) > 40 else title
        dynamic_max = max(100, config.MAX_WORDS_PER_CHUNK - len(short_title.split()))
        raw_chunks  = hlt_legal_chunker(cleaned, max_words=dynamic_max, overlap_words=config.OVERLAP_WORDS)
        if not raw_chunks:
            raw_chunks = [f"# {title}\n\n{cleaned}"]

        for i, c_raw in enumerate(raw_chunks):
            c_id       = f"{doc_id}_{i}"
            md_content = f"# {title}\n[Mã văn bản: {doc_id}]\n\n{c_raw}"
            sqlite_batch.append((c_id, doc_id, short_title, md_content))
            total_chunks += 1

        # Flush mỗi 5000 chunks để giữ RAM thấp
        if len(sqlite_batch) >= BATCH_SIZE:
            cursor.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", sqlite_batch)
            conn.commit()
            sqlite_batch.clear()

    if sqlite_batch:
        cursor.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", sqlite_batch)
        conn.commit()
        sqlite_batch.clear()

    conn.close()
    gc.collect()
    print(f"\n🎉 HOÀN THÀNH trong {time.time()-t0:.2f}s!")
    print(f"   Xử lý {total_files:,} văn bản → {total_chunks:,} chunks")
    print(f"   SQLite DB: {db_path} ({os.path.getsize(db_path)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
```

---

### CELL 3: BUILD BM25 INDEX (pyvi + BM25Okapi → pkl, ~10–15 phút)
```python
%%writefile build_bm25_train.py
# ============================================================
# CELL 3: BUILD BM25OKAPI INDEX (TỪ SQLITE)
# ============================================================
import sqlite3, json, time, os, pickle
import multiprocessing as mp
from tqdm import tqdm
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi
import config_train as config

# --- Hàm worker chạy song song ---
# Phải khai báo ở top-level module để multiprocessing hoạt động trên Windows/Kaggle
def _tokenize_worker(row):
    cid, did, text = row
    tokens = ViTokenizer.tokenize(text).split()
    return cid, did, tokens

def main():
    print("=========================================================")
    print("📖 BUILD BM25OKAPI INDEX (THAY THẾ FTS5) 📖")
    print("=========================================================\n")
    
    db_path = config.SQLITE_DB_PATH
    if not os.path.exists(db_path):
        print(f"❌ Không tìm thấy {db_path}. Hãy chạy CELL 2 trước!")
        return

    # [1] Kết nối SQLite và đếm tổng chunk
    print(f"[1] Kết nối CSDL: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]
    print(f"    Tổng số chunks: {total_chunks:,}")

    # Load toàn bộ data vào RAM (khoảng ~2-3GB cho 1.6M chunks, Kaggle RAM 30GB vô tư)
    cursor.execute("SELECT chunk_id, doc_id, markdown_content FROM chunks ORDER BY rowid")
    rows = cursor.fetchall()
    conn.close()

    # [2] Tokenize bằng pyvi (Multiprocessing)
    print(f"\n[2] Tokenize {total_chunks:,} chunks bằng pyvi ViTokenizer (3 cores)...")
    t0 = time.time()
    
    corpus_tokens = []
    chunk_ids     = []
    chunk_doc_map = {}

    # Sử dụng 3 processes (Kaggle CPU có 4 cores, chừa lại 1 core cho hệ điều hành)
    with mp.Pool(processes=3) as pool:
        # imap xử lý theo chunksize=5000 giúp chia việc đều và ít overhead
        results = list(tqdm(pool.imap(_tokenize_worker, rows, chunksize=5000), 
                            total=total_chunks, desc="Tokenizing"))
                            
    for cid, did, tokens in results:
        chunk_ids.append(cid)
        chunk_doc_map[cid] = did
        corpus_tokens.append(tokens)

    t1 = time.time()
    print(f"    ✅ Tokenize xong trong {t1 - t0:.2f} giây.")

    # [3] Build BM25
    print("\n[3] Khởi tạo mô hình BM25Okapi...")
    t2 = time.time()
    bm25 = BM25Okapi(corpus_tokens)
    t3 = time.time()
    print(f"    ✅ Build BM25 xong trong {t3 - t2:.2f} giây.")

    # [4] Lưu xuống file
    out_pkl = config.BM25_PKL_PATH
    print(f"\n[4] Lưu index BM25 ra file: {out_pkl}...")
    t4 = time.time()
    data_to_save = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "chunk_doc_map": chunk_doc_map,
        "total_chunks": total_chunks
    }
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(data_to_save, f)
    t5 = time.time()
    print(f"    ✅ Lưu xong trong {t5 - t4:.2f} giây. Kích thước file: {os.path.getsize(out_pkl)/(1024*1024):.2f} MB")

    print("\n🎉 HOÀN THÀNH BUILD BM25!")

if __name__ == "__main__":
    main()
```

---

### CELL 4: EMBED VECTORS (~60–90 phút, chỉ cần chạy 1 lần rồi upload .npy lên Kaggle Dataset)
```python
%%writefile embed_train.py
# ============================================================
# CELL 4: EMBED VECTORS — Đọc từ SQLite (legal_corpus.db)
# Xuất ra .npy + .json parts để upload lên Kaggle Dataset
# ============================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc, json, sqlite3, time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import config_train as config

def embed_corpus_local_numpy():
    print("=" * 70)
    print("CELL 4: EMBED VECTORS (đọc từ SQLite → xuất .npy)")
    print("=" * 70)

    db_path = config.SQLITE_DB_PATH
    out_dir = config.EMBEDED_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(db_path):
        print(f"❌ Không tìm thấy SQLite DB: {db_path}")
        print("   Hãy chạy CELL 2 trước!")
        return

    # [1] Đọc toàn bộ chunks từ SQLite
    print(f"\n[1] Đọc chunks từ SQLite: {db_path} ...")
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]
    print(f"    Tổng số chunks: {total_chunks:,}")

    cursor.execute("SELECT chunk_id, title, markdown_content FROM chunks ORDER BY rowid")
    rows = cursor.fetchall()
    conn.close()

    all_texts    = [row[2] for row in rows]   # markdown_content
    all_metadata = [
        {"chunk_id": row[0], "title": row[1], "link": ""}
        for row in rows
    ]
    del rows
    gc.collect()
    print(f"    Đã nạp {total_chunks:,} chunks vào RAM!")

    # [2] Khởi tạo model Multi-GPU
    print("\n[2] Khởi tạo Embedding Model (Multi-GPU Pool)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    pool  = model.start_multi_process_pool()

    # [3] Encode và lưu từng phần 100k chunks
    print("\n[3] Vector hóa & lưu từng phần ...")
    chunk_size = 100000
    num_parts  = (total_chunks + chunk_size - 1) // chunk_size

    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx   = min((i + 1) * chunk_size, total_chunks)
        print(f"\n🚀 Phần {i+1}/{num_parts}: [{start_idx:,} → {end_idx:,}]")

        embeddings = model.encode(
            all_texts[start_idx:end_idx],
            pool=pool,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        emb_path  = os.path.join(out_dir, f"embeddings_part_{i+1}.npy")
        meta_path = os.path.join(out_dir, f"metadata_part_{i+1}.json")
        np.save(emb_path, np.array(embeddings).astype(np.float32))
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(all_metadata[start_idx:end_idx], f, ensure_ascii=False)
        print(f"    ✅ Đã lưu: {emb_path}")

    model.stop_multi_process_pool(pool)
    print(f"\n🎉 XONG! Vector đã lưu tại: {out_dir}")
    print("👉 Upload toàn bộ file .npy và .json lên Kaggle Dataset!")

if __name__ == "__main__":
    embed_corpus_local_numpy()
```

---


### CELL 5: LLM QUERY REWRITE (Có thể comment # dòng chạy ở CELL 10 nếu muốn bỏ qua)
```python
%%writefile preprocess_queries_train.py
import os
import json
import time
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import config_train as config

class DualStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, timeout_seconds=5.0):
        self.tokenizer = tokenizer
        self.timeout = timeout_seconds
        self.start_time = time.time()
    def reset(self):
        self.start_time = time.time()
    def __call__(self, input_ids, scores, **kwargs):
        if time.time() - self.start_time > self.timeout:
            return True
        if '}' in self.tokenizer.decode(input_ids[0][-1]):
            return True
        return False

def parse_llm_json(raw_text, original_query):
    fallback = {"original_query": original_query, "cau_hoi_phap_ly": original_query, "tu_khoa": []}
    try:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match: return fallback
        json_str = re.sub(r',\s*}', '}', match.group(0))
        json_str = re.sub(r',\s*\]', ']', json_str)
        data = json.loads(json_str)
        return {"original_query": original_query,
                "cau_hoi_phap_ly": data.get("cau_hoi_phap_ly","").strip() or original_query,
                "tu_khoa": data.get("tu_khoa",[]) if isinstance(data.get("tu_khoa",[]), list) else []}
    except Exception:
        return fallback

def main():
    print(f"[1] Đang tải {config.LLM_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME, cache_dir=config.LLM_CACHE_DIR)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_NAME, torch_dtype=torch.float16, device_map="auto", cache_dir=config.LLM_CACHE_DIR
    )
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        questions_raw = json.load(f)
    query_ids = list(questions_raw.keys())
    if hasattr(config,'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        query_ids = query_ids[:config.JUST_CHECK_QUERIES]

    done_data = {}
    if os.path.exists(config.LLM_OUTPUT_QUERIES):
        with open(config.LLM_OUTPUT_QUERIES,'r',encoding='utf-8') as f:
            done_data = json.load(f)

    stopping_criteria = DualStoppingCriteria(tokenizer, 6.0)
    criteria_list = StoppingCriteriaList([stopping_criteria])

    for i, qid in enumerate(tqdm(query_ids, desc="Dịch truy vấn")):
        if qid in done_data: continue
        original_query = questions_raw[qid].get('question','') if isinstance(questions_raw[qid],dict) else questions_raw[qid]
        try:
            messages = [{"role":"system","content":config.LLM_SYSTEM_PROMPT},
                        {"role":"user","content":f"Phân tích câu hỏi sau:\n{original_query}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prefill = '{\n  "cau_hoi_phap_ly": "'
            prompt += prefill
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            stopping_criteria.reset()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False,
                                         stopping_criteria=criteria_list, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            full_response = prefill + generated
            if '}' not in full_response: full_response += '"\n}'
            done_data[qid] = parse_llm_json(full_response, original_query)
        except Exception as e:
            done_data[qid] = {"original_query":original_query,"cau_hoi_phap_ly":original_query,"tu_khoa":[]}
        finally:
            torch.cuda.empty_cache()
        if (i+1) % 20 == 0:
            os.makedirs(os.path.dirname(config.LLM_OUTPUT_QUERIES), exist_ok=True)
            with open(config.LLM_OUTPUT_QUERIES,'w',encoding='utf-8') as f:
                json.dump(done_data, f, ensure_ascii=False, indent=4)

    with open(config.LLM_OUTPUT_QUERIES,'w',encoding='utf-8') as f:
        json.dump(done_data, f, ensure_ascii=False, indent=4)
    print(f"✅ HOÀN TẤT! Lưu tại: {config.LLM_OUTPUT_QUERIES}")

if __name__ == "__main__":
    main()
```

---

### CELL 6: SEARCH BM25 (BM25Okapi pkl + pyvi)
```python
%%writefile search_bm25_low_ram_train.py
# ============================================================
# CELL 6: BM25 SEARCH — Load pkl, tokenize query bằng pyvi
# → BM25Okapi.get_scores() → Top-K chunk_ids
# ============================================================
import json, re, os, gc, pickle
from tqdm import tqdm
from pyvi import ViTokenizer
import config_train as config

def tokenize_query(text: str) -> list:
    """Tokenize query bằng pyvi — khớp với cách build BM25 ở CELL 3."""
    try:
        seg = ViTokenizer.tokenize(text)
    except:
        seg = text
    clean_str = re.sub(r'[^\w\s_]', ' ', seg.lower())
    return [w for w in clean_str.split() if len(w) > 1]

def main():
    print("=" * 70)
    print("CELL 6: BM25 SEARCH (BM25Okapi + pyvi)")
    print("=" * 70)

    bm25_path = config.BM25_PKL_PATH
    if not os.path.exists(bm25_path):
        print(f"❌ Không tìm thấy BM25 pkl: {bm25_path}")
        print("   Hãy chạy CELL 3 trước!")
        return

    # [1] Load BM25 index từ pkl
    print(f"\n[1] Load BM25 index từ: {bm25_path} ...")
    with open(bm25_path, "rb") as f:
        bm25_data  = pickle.load(f)
    bm25_model = bm25_data["bm25"]
    chunk_ids  = bm25_data["chunk_ids"]
    print(f"    Tổng chunks trong index: {bm25_data['total_chunks']:,}")

    # [2] Load câu hỏi
    print(f"\n[2] Đọc câu hỏi từ: {config.QUESTIONS_PUBLIC_FILE} ...")
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        q_data = json.load(f)

    llm_data = {}
    if os.path.exists(config.LLM_OUTPUT_QUERIES):
        print("    ✅ Tìm thấy LLM Rewritten Queries, sẽ dùng tu_khoa bổ sung.")
        with open(config.LLM_OUTPUT_QUERIES, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)

    query_ids = list(q_data.keys())
    if hasattr(config, 'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        query_ids = query_ids[:config.JUST_CHECK_QUERIES]
    print(f"    Số câu hỏi: {len(query_ids):,}")

    # [3] BM25 Search từng câu hỏi
    print(f"\n[3] BM25 Search Top-{config.RETRIEVAL_TOP_K} ...")
    import numpy as np
    submission = {}
    top_k = config.RETRIEVAL_TOP_K

    for qid in tqdm(query_ids, desc="BM25 Search"):
        q_raw    = q_data[qid]
        original = q_raw.get('question', '') if isinstance(q_raw, dict) else q_raw

        # Ghép tu_khoa từ LLM nếu có
        query_text = original
        if qid in llm_data:
            tu_khoa = llm_data[qid].get("tu_khoa", [])
            if isinstance(tu_khoa, list) and tu_khoa:
                query_text = original + " " + " ".join(tu_khoa)

        tokens = tokenize_query(query_text)
        if not tokens:
            submission[qid] = {"answer": []}
            continue

        scores     = bm25_model.get_scores(tokens)
        top_k_idx  = np.argpartition(scores, -top_k)[-top_k:]
        top_k_idx  = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        submission[qid] = {"answer": [chunk_ids[i] for i in top_k_idx]}

    del bm25_model, chunk_ids, bm25_data
    gc.collect()

    os.makedirs(os.path.dirname(config.SUBMISSION_BM25_TOP_K), exist_ok=True)
    with open(config.SUBMISSION_BM25_TOP_K, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
    print(f"\n🎉 Hoàn tất BM25 Search! Lưu tại: {config.SUBMISSION_BM25_TOP_K}")

if __name__ == "__main__":
    main()
```

---

### CELL 7: SEARCH SEMANTIC (Trực tiếp từ .npy — không cần Milvus!)
```python
%%writefile search_public_low_ram_train.py
# Tìm kiếm vector trực tiếp từ .npy + .json đã upload
# Không cần Milvus, tốc độ và độ chính xác tương đương (cùng brute-force FLAT)
import json, os, glob, gc
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import config_train as config

def main():
    print("=========================================================")
    print("🔍 SEMANTIC SEARCH TRỰC TIẾP TỪ .NPY (KHÔNG CẦN MILVUS) 🔍")
    print("=========================================================\n")

    embed_dir = config.EMBEDED_DATA_INPUT

    # Hỗ trợ cả .json lẫn .txt
    npy_files  = sorted(glob.glob(os.path.join(embed_dir, "embeddings_part_*.npy")))
    json_files = sorted(
        glob.glob(os.path.join(embed_dir, "metadata_part_*.json")) +
        glob.glob(os.path.join(embed_dir, "metadata_part_*.txt"))
    )

    if not npy_files or len(npy_files) != len(json_files):
        print(f"❌ Lỗi: {len(npy_files)} .npy vs {len(json_files)} metadata — không khớp!")
        print(f"Kiểm tra lại: {embed_dir}")
        return

    print(f"📦 Tìm thấy {len(npy_files)} phần vector. Đang đọc chunk_id...")

    # [1] Load tất cả chunk_id vào RAM (nhỏ, chỉ ~50MB)
    all_chunk_ids = []
    part_sizes = []
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        all_chunk_ids.extend([m["chunk_id"] for m in meta])
        part_sizes.append(len(meta))
    total_chunks = len(all_chunk_ids)
    print(f"✅ Tổng số chunk: {total_chunks:,}")

    # [2] Tải câu hỏi
    print(f"\n[2] Đang đọc câu hỏi từ {config.QUESTIONS_PUBLIC_FILE}...")
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        q_data = json.load(f)

    llm_data = {}
    if os.path.exists(config.LLM_OUTPUT_QUERIES):
        print("✅ Tìm thấy LLM Rewritten Queries.")
        with open(config.LLM_OUTPUT_QUERIES, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)

    query_ids = list(q_data.keys())
    if hasattr(config, 'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        query_ids = query_ids[:config.JUST_CHECK_QUERIES]
    print(f"Số câu hỏi cần tìm: {len(query_ids)}")

    # [3] Encode toàn bộ query một lần → (num_queries, 1024)
    print("\n[3] Đang encode tất cả câu hỏi bằng BGE-M3...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

    query_texts = []
    for qid in query_ids:
        q_raw = q_data[qid]
        original = q_raw.get('question', '') if isinstance(q_raw, dict) else q_raw
        text = original
        if qid in llm_data and "cau_hoi_phap_ly" in llm_data[qid]:
            # Kết hợp cả câu hỏi gốc và ý định pháp lý từ LLM
            text = f"{original}. {llm_data[qid]['cau_hoi_phap_ly']}"
        query_texts.append(text)

    query_matrix = model.encode(query_texts, normalize_embeddings=True,
                                batch_size=32, show_progress_bar=True)
    query_matrix = query_matrix.astype(np.float32)  # (num_queries, 1024)

    # Giải phóng VRAM ngay sau khi encode xong
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ Encode xong! Shape: {query_matrix.shape}")

    # [4] Tính điểm theo từng phần .npy, tích lũy top-K
    # Strategy: mỗi part tính scores cho TẤT CẢ query cùng lúc → O(num_parts) lần đọc disk
    print(f"\n[4] Đang tính điểm similarity cho {total_chunks:,} chunks...")
    n_queries = len(query_ids)
    top_k = config.RETRIEVAL_TOP_K

    # Lưu score tích lũy theo từng query — dùng list of numpy arrays để ghép sau
    accumulated_scores = [[] for _ in range(n_queries)]

    for npy_f in tqdm(npy_files, desc="Quét .npy parts"):
        embs = np.load(npy_f).astype(np.float32)  # (N, 1024)
        # Tính dot product: (num_queries, 1024) @ (1024, N) → (num_queries, N)
        part_scores = query_matrix @ embs.T
        for qi in range(n_queries):
            accumulated_scores[qi].append(part_scores[qi].copy())
        del embs, part_scores
        gc.collect()

    # [5] Tổng hợp top-K cho từng query
    print("\n[5] Tổng hợp kết quả Top-K...")
    submission = {}
    for qi, qid in enumerate(tqdm(query_ids, desc="Tổng hợp Top-K")):
        full_scores = np.concatenate(accumulated_scores[qi])  # (total_chunks,)
        # argpartition nhanh hơn argsort khi chỉ cần top-K
        top_k_indices = np.argpartition(full_scores, -top_k)[-top_k:]
        top_k_indices = top_k_indices[np.argsort(full_scores[top_k_indices])[::-1]]
        submission[qid] = {"answer": [all_chunk_ids[i] for i in top_k_indices]}

    os.makedirs(os.path.dirname(config.SUBMISSION_SEMANTIC_TOP_K), exist_ok=True)
    with open(config.SUBMISSION_SEMANTIC_TOP_K, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
    print(f"\n🎉 HOÀN TẤT SEMANTIC SEARCH! Lưu tại: {config.SUBMISSION_SEMANTIC_TOP_K}")

if __name__ == "__main__":
    main()
```

---

### CELL 8: RRF FUSION
```python
%%writefile rrf_fusion_train.py
import json, os
from tqdm import tqdm
import config_train as config

K_RRF = 60
TOP_K_FINAL = config.RRF_TOP_K

def fuse_rrf():
    print("🧬 RRF FUSION SEMANTIC + BM25\n")
    if not os.path.exists(config.SUBMISSION_SEMANTIC_TOP_K) or not os.path.exists(config.SUBMISSION_BM25_TOP_K):
        print("❌ Thiếu file đầu vào!"); return

    with open(config.SUBMISSION_SEMANTIC_TOP_K,'r',encoding='utf-8') as f: semantic_data = json.load(f)
    with open(config.SUBMISSION_BM25_TOP_K,'r',encoding='utf-8') as f: bm25_data = json.load(f)

    hybrid_submission = {}
    for qid in tqdm(semantic_data.keys(), desc="Fusion RRF"):
        rrf_scores = {}
        for rank, chunk_id in enumerate(semantic_data.get(qid,{}).get("answer",[])):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id,0.0) + 1.0/(K_RRF+rank+1)
        for rank, chunk_id in enumerate(bm25_data.get(qid,{}).get("answer",[])):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id,0.0) + 1.0/(K_RRF+rank+1)
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_chunks = []
        doc_chunk_count = {}
        for chunk, score in sorted_chunks:
            doc_id = chunk.rsplit('_', 1)[0]   # format: {doc_id}_{i}
            if doc_chunk_count.get(doc_id,0) < getattr(config,'MAX_CHUNKS_PER_DOC',50):
                top_k_chunks.append(chunk)
                doc_chunk_count[doc_id] = doc_chunk_count.get(doc_id,0) + 1
            if len(top_k_chunks) == TOP_K_FINAL: break
        hybrid_submission[qid] = {"answer": top_k_chunks}

    os.makedirs(os.path.dirname(config.SUBMISSION_HYBRID_TOP_K), exist_ok=True)
    with open(config.SUBMISSION_HYBRID_TOP_K,'w',encoding='utf-8') as f:
        json.dump(hybrid_submission, f, ensure_ascii=False, indent=4)
    print(f"🎉 HOÀN TẤT! Lưu tại: {config.SUBMISSION_HYBRID_TOP_K}")

if __name__ == "__main__":
    fuse_rrf()
```

---

### CELL 9: RERANKER
```python
%%writefile rerank_public_train.py
import json
import os
import sys
import sqlite3
from tqdm import tqdm
import multiprocessing as mp
import config_train as config

TOP_K_FILE     = config.SUBMISSION_HYBRID_TOP_K
RERANKER_MODEL = config.RERANKER_MODEL
BATCH_SIZE     = config.RERANKER_BATCH_SIZE

def load_data():
    print("[1] Đang tải danh sách câu hỏi...")
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        questions_raw = json.load(f)

    q_dict = {k: (v.get("question", "") if isinstance(v, dict) else v) for k, v in questions_raw.items()}

    rewritten_data = {}
    if os.path.exists(config.LLM_OUTPUT_QUERIES):
        print("✅ Tìm thấy LLM Queries, sẽ dùng cho Reranker.")
        with open(config.LLM_OUTPUT_QUERIES, 'r', encoding='utf-8') as f:
            rewritten_data = json.load(f)

    print("[2] Đang tải kết quả Top 500 thô (Retrieval)...")
    with open(TOP_K_FILE, 'r', encoding='utf-8') as f:
        top_k_data = json.load(f)

    if hasattr(config, 'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        top_k_data = {k: top_k_data[k] for k in list(top_k_data.keys())[:config.JUST_CHECK_QUERIES]}

    needed_chunks = set()
    for qid, data in top_k_data.items():
        needed_chunks.update(data['answer'])

    print(f"[3] Cần đọc nội dung của {len(needed_chunks):,} chunks từ SQLite...")
    doc_text_dict = {}

    # ── ĐỌC TỪ SQLITE THAY VÌ QUÉT THƯ MỤC .JSONL ──────────
    db_path = config.SQLITE_DB_PATH
    if not os.path.exists(db_path):
        print(f"❌ Không tìm thấy SQLite DB: {db_path}. Hãy chạy CELL 2 trước!")
        return q_dict, top_k_data, {}, rewritten_data

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query theo batch để tránh "too many variables" với SQLite
    needed_list = list(needed_chunks)
    BATCH = 500
    for i in range(0, len(needed_list), BATCH):
        batch = needed_list[i:i + BATCH]
        placeholders = ",".join("?" * len(batch))
        cursor.execute(
            f"SELECT chunk_id, markdown_content FROM chunks WHERE chunk_id IN ({placeholders})",
            batch
        )
        for chunk_id, md_content in cursor.fetchall():
            doc_text_dict[chunk_id] = md_content

    conn.close()
    print(f"    Đã đọc {len(doc_text_dict):,}/{len(needed_chunks):,} chunks từ SQLite.")
    return q_dict, top_k_data, doc_text_dict, rewritten_data

def worker_rerank(gpu_id, query_ids, q_dict, top_k_data, doc_text_dict, rewritten_data):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from FlagEmbedding import FlagReranker
    print(f"\n[GPU {gpu_id}] Đang nạp mô hình lên VRAM (fp32)...")
    reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
    
    submission_part = {}
    
    for qid in tqdm(query_ids, desc=f"GPU {gpu_id} Reranking", position=gpu_id, leave=True):
        data = top_k_data[qid]
        question_text = q_dict.get(qid, "")

        if qid in rewritten_data and "cau_hoi_phap_ly" in rewritten_data[qid]:
            # Nhấn mạnh câu gốc là chính, câu của LLM chỉ để tham khảo thêm
            combined_query = f"Câu hỏi chính: {question_text}\n(Thông tin tham khảo thêm: {rewritten_data[qid]['cau_hoi_phap_ly']})"
        else:
            combined_query = question_text

        pairs = []
        valid_chunks = []
        for chunk_id in data['answer']:
            if not chunk_id: continue
            text = doc_text_dict.get(chunk_id, "")
            pairs.append([combined_query, text])
            valid_chunks.append(chunk_id)

        if not pairs:
            submission_part[qid] = {"answer": []}
            continue

        try:
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    scores = reranker.compute_score(pairs, batch_size=BATCH_SIZE, max_length=config.MAX_LENGTH)
                finally:
                    sys.stderr = old_stderr
        except Exception as e:
            print(f"\n  [GPU {gpu_id}] Lỗi rerank câu {qid}: {e}")
            scores = [0.0] * len(pairs)

        if hasattr(scores, "tolist"): scores = scores.tolist()
        if not isinstance(scores, list): scores = [float(scores)]
        else: scores = [float(s) for s in scores]

        chunk_scores = sorted(zip(valid_chunks, scores), key=lambda x: x[1], reverse=True)

        top5_doc_ids = []
        seen_docs = set()
        for chunk, score in chunk_scores:
            doc_id = chunk.rsplit('_', 1)[0]   # format: {doc_id}_{i}
            if doc_id not in seen_docs:
                top5_doc_ids.append(doc_id)
                seen_docs.add(doc_id)
            if len(top5_doc_ids) >= 5: break

        submission_part[qid] = {"answer": top5_doc_ids}

    return submission_part

def main():
    print("=========================================================")
    print("🏆 BẮT ĐẦU CHẤM ĐIỂM (MULTIPROCESSING 2 GPU - FP32 - 8192 TOKENS) 🏆")
    print("=========================================================\n")

    q_dict, top_k_data, doc_text_dict, rewritten_data = load_data()
    
    all_qids = list(top_k_data.keys())
    mid = len(all_qids) // 2
    qids_0 = all_qids[:mid]
    qids_1 = all_qids[mid:]
    
    print(f"\n[4] Khởi tạo Pool: GPU 0 làm {len(qids_0)} câu | GPU 1 làm {len(qids_1)} câu")
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(2) as pool:
        args = [
            (0, qids_0, q_dict, top_k_data, doc_text_dict, rewritten_data),
            (1, qids_1, q_dict, top_k_data, doc_text_dict, rewritten_data)
        ]
        results = pool.starmap(worker_rerank, args)
        
    print("\n[5] Đang tổng hợp kết quả từ 2 GPU...")
    final_submission = {}
    final_submission.update(results[0])
    final_submission.update(results[1])

    os.makedirs(os.path.dirname(config.SUBMISSION_FINAL), exist_ok=True)
    with open(config.SUBMISSION_FINAL, 'w', encoding='utf-8') as f:
        json.dump(final_submission, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 HOÀN TẤT PIPELINE! Kết quả được lưu tại: {config.SUBMISSION_FINAL}")

if __name__ == "__main__":
    main()
```

---

### CELL 10: CHẠY TOÀN BỘ
```python
# ============================================================
# PIPELINE MỚI (HLT-Chunking + BM25Okapi + SQLite + Numpy)
#
# ── MODE A: CHẠY TOÀN BỘ 1 NOTEBOOK (mặc định — đang dùng)
#    CELL 1 Config: EMBEDED_DATA_INPUT = EMBEDED_DATA_DIR  ← đã set sẵn
#    Tổng thời gian ước tính: ~3.5 đến 5 tiếng
#
# ── MODE B: PHASE 2 (đã upload .npy mới lên Dataset)
#    CELL 1 Config: bỏ comment dòng EMBEDED_DATA_INPUT = r"/kaggle/input/..."
#                   comment dòng EMBEDED_DATA_INPUT = EMBEDED_DATA_DIR
#    CELL 10: comment lại dòng !python -u embed_train.py
# ============================================================

print("⏳ BƯỚC 1: XỬ LÝ DỮ LIỆU & BUILD BM25 (~80 phút)")
!python -u preprocess_train.py
!python -u build_bm25_train.py

print("\n⏳ BƯỚC 2: VECTOR HÓA EMBEDDING (~90–120 phút)")
# MODE A: BẮT BUỘC chạy (chunk đã đổi từ 6000 → 300 từ, .npy cũ không dùng được)
!python -u embed_train.py
# MODE B: Comment dòng trên nếu đã upload .npy mới lên Dataset

print("\n⏳ BƯỚC 3: DỊCH CÂU HỎI LLM (~15 phút)")
# Comment # dòng dưới nếu muốn bỏ qua LLM
!python -u preprocess_queries_train.py

print("\n⏳ BƯỚC 4: TÌM KIẾM BM25 & SEMANTIC (~30 phút)")
!python -u search_bm25_low_ram_train.py
!python -u search_public_low_ram_train.py

print("\n⏳ BƯỚC 5: RRF FUSION + RERANKER (~60–90 phút)")
!python -u rrf_fusion_train.py
!python -u rerank_public_train.py

print("\n✅ PIPELINE HOÀN TẤT! Tải file /kaggle/working/train/submission.json về nộp!")
```

