# KAGGLE NOTEBOOK PIPELINE: DSC 2026 LEGAL RETRIEVAL

Dưới đây là toàn bộ code của Baseline. Bạn hãy mở Kaggle Notebook, tạo các Code Cell và sao chép toàn bộ nội dung trong các khối code dưới đây (bao gồm cả dòng `%%writefile`) dán vào từng Cell tương ứng.

> **⚠️ QUAN TRỌNG:** Trong Cell 1 (`config_train.py`), bạn cần sửa đường dẫn `/kaggle/input/dsc-2026-dataset/` thành TÊN DATASET mà bạn đã tải lên Kaggle.

---

### CELL 0: CÀI ĐẶT THƯ VIỆN
> Khối này dùng để cài đặt thư viện. Hãy chạy nó đầu tiên khi vừa bật Kaggle lên.
```bash
# Cài đặt thư viện dựa trên pyproject.toml (Giữ lại Torch gốc của Kaggle để không hỏng CUDA)
!pip install -q -U accelerate>=1.14.0 bitsandbytes>=0.50.1 flagembedding>=1.3.0 milvus-lite>=3.2.0 pymilvus>=2.4.0 pyvi>=0.1.1 sentence-transformers>=6.0.0 transformers==5.15.0
```

---

### CELL 1: CONFIG
```python
%%writefile config_train.py
# ============================================================
# CELL 1: CẤU HÌNH TRUNG TÂM (config_train.py)
# Chỉnh đường dẫn dataset tại đây nếu cần
# ============================================================
import os

# ================= 1. CẤU HÌNH THƯ MỤC =================
# Đường dẫn thực tế trên Kaggle (đã debug xác nhận)
DEFAULT_INPUT_PATH = r"/kaggle/input/datasets/nguynquckhanh/selected-context/selected-contexts"
PROCESSED_DATA_DIR = r"/kaggle/working/train/processed_data"
EMBEDED_DATA_DIR = r"/kaggle/working/train/embeded_data"
QUESTIONS_PUBLIC_FILE = r"/kaggle/input/datasets/nguynquckhanh/public-test/public-official.json"

# CHẾ ĐỘ CHẠY
JUST_CHECK_QUERIES = None   # Để None để chạy toàn bộ file test
JUST_CHECK_CONTEXT = None   # Để None để chạy toàn bộ kho luật

# ================= 2. THAM SỐ CHIA ĐOẠN (CHUNKING) =================
MAX_WORDS_PER_CHUNK = 6000
OVERLAP_WORDS = 50

# ================= 3. TRUY VẤN GIẢ ĐỊNH & JSON EXTRACTION =================
LLM_MODEL_NAME = "vilm/vinallama-2.7b-chat"
LLM_CACHE_DIR = r"/kaggle/working/train/local_llm_data/.hf_cache"
LLM_INPUT_QUERIES = r"/kaggle/input/datasets/nguynquckhanh/public-test/public-official.json"
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
    'Đầu ra:\n'
    '{\n'
    '  "cau_hoi_phap_ly": "Quy định về lệ phí cấp giấy chứng nhận quyền sử dụng đất lần đầu",\n'
    '  "tu_khoa": ["lệ phí", "cấp giấy chứng nhận", "quyền sử dụng đất", "lần đầu"]\n'
    '}\n'
    "--- HẾT VÍ DỤ ---\n\n"
    "Bắt đầu xử lý câu hỏi sau (Chỉ xuất JSON):"
)

# ================= 4. LẬP CHỈ MỤC & VÒNG LOẠI TÌM KIẾM (HYBRID SEARCH) =================
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
MAX_POSITION_EMBEDDINGS = 2048
EMBEDDING_BATCH_SIZE = 4
MILVUS_DB_PATH = r"/kaggle/working/train/embeded_data/milvus_legal_bgem3.db"
MILVUS_COLLECTION = "legal_chunks_bgem3"
SUBMISSION_SEMANTIC_TOP_K = r"/kaggle/working/train/submission_semantic_topK.json"

BM25_DB_PATH = r"/kaggle/working/train/embeded_data/bm25_index.db"
SUBMISSION_BM25_TOP_K = r"/kaggle/working/train/submission_bm25_topK.json"
RETRIEVAL_TOP_K = 500

# ================= 5. LAI GHÉP & CHẮT LỌC (RRF FUSION) =================
SUBMISSION_HYBRID_TOP_K = r"/kaggle/working/train/submission_hybrid_topK.json"
MAX_CHUNKS_PER_DOC = 50
RRF_TOP_K = 500

# ================= 6. VÒNG CHUNG KẾT & NỘP BÀI (RERANKING) =================
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
RERANKER_BATCH_SIZE = 32
SUBMISSION_FINAL = r"/kaggle/working/train/submission.json"
```

---

### CELL 2: PREPROCESS TEXT
```python
%%writefile preprocess_train.py
# ============================================================
# CELL 2: TIỀN XỬ LÝ VĂN BẢN (preprocess_train.py)
# Băm nhỏ toàn bộ văn bản luật thành các Chunks
# ============================================================
import sys
import os
import re
import json
import glob
import unicodedata
import logging
from tqdm import tqdm
import config_train as config

logging.basicConfig(filename='pipeline_errors.log', level=logging.WARNING,
                    format='%(asctime)s - %(message)s')

def filter_and_clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    match_start = re.search(r'(QUYẾT ĐỊNH:|NGHỊ QUYẾT:|CĂN CỨ:|Điều 1\.)', text)
    if match_start:
        text = text[match_start.start():]
    text = re.sub(
        r'(Nơi nhận:|KT\. BỘ TRƯỞNG).*?(?=(?:\n\s*(?:QUY CHẾ|PHỤ LỤC|Chương \d+|Điều \d+\.))|\\Z)',
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'(?<![\.\:\;\!\?])\s*\n\s*(?=[A-Za-zà-ỹ0-9])', ' ', text)
    text = re.sub(r'(?<=[a-zà-ỹ0-9\.\:\;\!\?])\s+(Điều\s+\d+\.\s+[A-ZÀ-Ỹ])', r'\n\n\1', text)
    text = re.sub(r'(?<=[a-zà-ỹ0-9\.\:\;\!\?])\s+(Chương\s+[IVXLCDM]+\.?\s+[A-ZÀ-Ỹ])', r'\n\n\1', text)
    text = re.sub(r'(?<!Điều)(?<!Chương)(?<!Mục)(?<!Phần)(?<!Khoản)(?<!Điểm)(?<=[a-zà-ỹ0-9\.\:\;\!\?])\s+(\d+\.\s+[A-ZÀ-Ỹ])', r'\n\n\1', text)
    text = re.sub(r'(?<![Đđ]iểm)(?<![Kk]hoản)(?<=[a-zà-ỹ0-9\.\:\;\!\?])\s+([a-zà-ỹ]\)\s+[A-ZÀ-Ỹa-zà-ỹ])', r'\n\n\1', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def stateful_legal_chunker(text: str, max_words: int = 300, overlap_words: int = 50) -> list:
    if not text or not text.strip():
        return []
    lines = text.split('\n')
    final_chunks = []
    current_article = ""
    current_clause_lead = ""
    current_chunk = ""

    def flush_chunk():
        nonlocal current_chunk
        if current_chunk:
            final_chunks.append(current_chunk.strip())
            current_chunk = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_article = re.match(r'^(Điều\s+\d+|Chương\s+[IVXLCDM]+|PHỤ LỤC|MỤC)', line)
        is_clause = re.match(r'^\d+\.\s+', line)
        is_point = re.match(r'^([a-zà-ỹ]\)|-|\+)\s+', line)

        if is_article:
            flush_chunk()
            current_article = line
            current_clause_lead = ""
            current_chunk = line
        elif is_clause:
            flush_chunk()
            current_clause_lead = line
            current_chunk = f"{current_article} {line}" if current_article else line
        elif is_point:
            flush_chunk()
            chunk_prefix = ""
            if current_article:
                chunk_prefix += current_article + " "
            if current_clause_lead:
                chunk_prefix += current_clause_lead + " "
            current_chunk = chunk_prefix + line
        else:
            if line.endswith(':'):
                current_clause_lead = line
            if not current_chunk:
                chunk_prefix = ""
                if current_article:
                    chunk_prefix += current_article + " "
                if current_clause_lead:
                    chunk_prefix += current_clause_lead + " "
                current_chunk = chunk_prefix + line
            else:
                current_chunk += " " + line

    flush_chunk()

    final_limited_chunks = []
    for chunk in final_chunks:
        words = chunk.split()
        if len(words) <= max_words:
            final_limited_chunks.append(chunk)
        else:
            step = max_words - overlap_words
            if step <= 0:
                step = max_words
            for i in range(0, len(words), step):
                chunk_text = " ".join(words[i:i + max_words]).strip()
                if chunk_text:
                    final_limited_chunks.append(chunk_text)
    return final_limited_chunks

def process_corpus_to_md(input_dir=config.DEFAULT_INPUT_PATH, output_dir=config.PROCESSED_DATA_DIR,
                          base_max_words=config.MAX_WORDS_PER_CHUNK, overlap=config.OVERLAP_WORDS):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    json_files_list = list(glob.iglob(os.path.join(input_dir, "*.json")))

    if hasattr(config, 'JUST_CHECK_CONTEXT') and config.JUST_CHECK_CONTEXT is not None:
        json_files_list = json_files_list[:config.JUST_CHECK_CONTEXT]
        print(f"\n[TEST MODE] Chỉ xử lý {len(json_files_list)} văn bản đầu tiên...")

    print(f"Tìm thấy {len(json_files_list)} file. Bắt đầu xử lý...")
    total_docs = 0
    total_chunks = 0

    for file_path in tqdm(json_files_list, desc="Tiền xử lý", unit="file"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
        except Exception as e:
            logging.warning(f"Lỗi đọc file {file_path}: {e}")
            continue

        doc_id  = str(doc.get("id", "")).strip()
        title   = doc.get("name", doc.get("title", "")).strip()
        passage = doc.get("passage", "").strip()

        if not doc_id or not passage:
            logging.warning(f"Bỏ qua {file_path}: thiếu 'id' hoặc 'passage'.")
            continue

        doc_dir = os.path.join(output_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)

        cleaned_text = filter_and_clean_text(passage)

        title_words = title.split()
        if len(title_words) > 60:
            title = " ".join(title_words[:60]) + "..."
        dynamic_max_words = max(100, base_max_words - len(title.split()))

        raw_chunks = stateful_legal_chunker(cleaned_text, max_words=dynamic_max_words, overlap_words=overlap)

        jsonl_path = os.path.join(doc_dir, "chunks.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            for i, chunk_raw in enumerate(raw_chunks):
                chunk_id = f"chunk_{i}"
                text_with_header = f"# {title}\n\n{chunk_raw}" if title else chunk_raw
                record = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": text_with_header
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_chunks += 1
        total_docs += 1

    print(f"\n🎉 HOÀN THÀNH! Xử lý {total_docs} văn bản, tạo {total_chunks} chunks.")
    print(f"Thư mục lưu: {output_dir}")

if __name__ == "__main__":
    process_corpus_to_md()
```

---

### CELL 3: BUILD BM25
```python
%%writefile build_bm25_train.py
import os
import glob
import json
import sqlite3
import re
from tqdm import tqdm
import config_train as config

def remove_vietnamese_accents(text):
    s1 = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ"
    s0 = "AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy"
    s = ""
    for c in text:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def tokenize_vietnamese(text):
    text = text.lower()
    text = remove_vietnamese_accents(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return " ".join(tokens)

def build_fts5_db():
    print("--- BẮT ĐẦU XÂY DỰNG TỪ ĐIỂN BM25 (SQLITE FTS5) ---")
    
    db_path = config.BM25_DB_PATH
    data_dir = config.PROCESSED_DATA_DIR
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Đã xóa CSDL cũ: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE VIRTUAL TABLE bm25_docs USING fts5(
            doc_id UNINDEXED, 
            chunk_id UNINDEXED, 
            title UNINDEXED,
            content,
            tokenize="unicode61 remove_diacritics 1"
        )
    ''')

    doc_dirs = glob.glob(os.path.join(data_dir, "*"))
    
    chunk_files = []
    for d in doc_dirs:
        if os.path.isdir(d):
            jsonl_path = os.path.join(d, "chunks.jsonl")
            if os.path.exists(jsonl_path):
                chunk_files.append(jsonl_path)
                
    print(f"Tìm thấy {len(chunk_files)} file chunks. Đang nạp vào Database...")

    insert_count = 0
    for chunk_file in tqdm(chunk_files, desc="Đang nạp FTS5"):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                d_id = data.get('doc_id')
                c_id = data.get('chunk_id')
                title = data.get('title', '')
                text = data.get('text', '')

                tokenized_text = tokenize_vietnamese(title + " " + text)

                cursor.execute(
                    "INSERT INTO bm25_docs (doc_id, chunk_id, title, content) VALUES (?, ?, ?, ?)",
                    (d_id, c_id, title, tokenized_text)
                )
                insert_count += 1

    conn.commit()
    conn.close()
    
    print(f"🎉 Đã nạp thành công {insert_count} chunks vào SQLite BM25 FTS5!")
    print(f"Cơ sở dữ liệu được lưu tại: {db_path}")

if __name__ == "__main__":
    build_fts5_db()
```

---

### CELL 4: EMBED VECTORS
```python
%%writefile embed_train.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # CHỐNG PHÂN MẢNH VRAM

import glob
import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import config_train as config

def embed_corpus_local_numpy():
    print("--- BẮT ĐẦU NẶN VECTOR (EMBEDDING) CHO TOÀN BỘ VĂN BẢN ---")
    data_dir = config.PROCESSED_DATA_DIR
    out_dir = config.EMBEDED_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("\n--- BƯỚC 1: ĐỌC DỮ LIỆU TỪ JSONL ---")
    doc_dirs = glob.glob(os.path.join(data_dir, "*"))
    chunk_files = [os.path.join(d, "chunks.jsonl") for d in doc_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "chunks.jsonl"))]

    all_texts = []
    all_metadata = []

    print(f"Đang nạp toàn bộ nội dung từ {len(chunk_files)} file văn bản vào RAM...")
    for file_path in tqdm(chunk_files, desc="Đọc files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                doc_id = data.get("doc_id")
                chunk_id = data.get("chunk_id")
                text = data.get("text", "")
                
                all_texts.append(text)
                full_chunk_id = f"{doc_id}_{chunk_id}"
                all_metadata.append({
                    "chunk_id": full_chunk_id,
                    "title": data.get("title", ""),
                    "link": ""
                })

    total_chunks = len(all_texts)
    print(f"🎉 Đã nạp xong {total_chunks} đoạn văn bản từ {len(chunk_files)} file!")

    print("\n--- BƯỚC 2: KHỞI TẠO MÔ HÌNH AI ---")
    print(f"💻 Đang kích hoạt thiết bị: Multi-GPU (Tự động phát hiện)")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    print("🚀 Bật chế độ Đa GPU (Multi-Process Pool)...")
    pool = model.start_multi_process_pool()

    print("\n--- BƯỚC 3: VECTOR HÓA & LƯU TỪNG PHẦN ---")
    chunk_size = 100000 
    num_parts = (total_chunks + chunk_size - 1) // chunk_size

    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_chunks)
        print(f"\n🚀 Đang xử lý Phần {i+1}: Từ {start_idx} đến {end_idx}...")
        
        batch_texts = all_texts[start_idx:end_idx]
        batch_meta = all_metadata[start_idx:end_idx]

        # Cập nhật hàm encode mới nhất theo Warning và thêm show_progress_bar=True
        embeddings = model.encode(
            batch_texts, 
            pool=pool,
            batch_size=config.EMBEDDING_BATCH_SIZE, 
            normalize_embeddings=True,
            show_progress_bar=True
        )

        emb_path = os.path.join(out_dir, f"embeddings_part_{i+1}.npy")
        meta_path = os.path.join(out_dir, f"metadata_part_{i+1}.json")

        np.save(emb_path, np.array(embeddings).astype(np.float32))
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(batch_meta, f, ensure_ascii=False)
            
        print(f"✅ Đã lưu Phần {i+1} xong! (File: {emb_path})")

    # Dừng pool giải phóng VRAM sau khi xong
    model.stop_multi_process_pool(pool)
    print("\n🎉 XONG TOÀN BỘ QUÁ TRÌNH NẶN VECTOR! CHUẨN BỊ NẠP VÀO DB ĐƯỢC RỒI ĐÓ SẾP!")

if __name__ == "__main__":
    embed_corpus_local_numpy()
```

---

### CELL 5: BUILD MILVUS
```python
%%writefile build_local_milvus_train.py
import os
import glob
import json
import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient, DataType
import config_train as config

def build_milvus_local():
    print("=========================================================")
    print("🏰 BẮT ĐẦU NẠP DỮ LIỆU VÀO MILVUS LOCAL (VECTOR DB) 🏰")
    print("=========================================================\n")

    db_path = config.MILVUS_DB_PATH
    collection_name = config.MILVUS_COLLECTION
    embed_dir = config.EMBEDED_DATA_DIR

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"🗑️ Đã xóa Database cũ tại: {db_path}")

    print(f"🔌 Đang kết nối tới Milvus Lite ({db_path})...")
    client = MilvusClient(db_path)

    if client.has_collection(collection_name):
        print(f"⚠️ Bảng {collection_name} đã tồn tại. Đang xóa để tạo mới...")
        client.drop_collection(collection_name)

    print(f"🔨 Đang tạo bảng {collection_name} trong Milvus...")
    
    schema = MilvusClient.create_schema(
        auto_id=False, 
        enable_dynamic_field=True
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding", 
        metric_type="IP",
        index_type="FLAT",
        index_name="vector_index"
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    print("✅ Đã tạo bảng thành công!\n")

    npy_files = sorted(glob.glob(os.path.join(embed_dir, "embeddings_part_*.npy")))
    json_files = sorted(glob.glob(os.path.join(embed_dir, "metadata_part_*.json")))

    if not npy_files or len(npy_files) != len(json_files):
        print("❌ Lỗi: Không tìm thấy cục Vector (.npy) nào, hoặc số lượng file npy và json không khớp!")
        return

    print(f"📦 Đã tìm thấy {len(npy_files)} cục Vector (.npy).")
    
    global_id = 0
    total_inserted = 0

    for i, (npy_f, json_f) in enumerate(zip(npy_files, json_files)):
        print(f"\n⏳ Đang xử lý cục: part_{i+1}...")
        
        embeddings = np.load(npy_f)
        with open(json_f, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        num_vectors = len(embeddings)
        print(f"   - Đang chèn {num_vectors} dòng vào Milvus...")
        
        batch_size = 5000
        for j in tqdm(range(0, num_vectors, batch_size), desc="Chèn vào DB"):
            batch_emb = embeddings[j:j+batch_size]
            batch_meta = metadata[j:j+batch_size]
            
            insert_data = []
            for k in range(len(batch_emb)):
                insert_data.append({
                    "id": global_id,
                    "chunk_id": batch_meta[k]["chunk_id"],
                    "embedding": batch_emb[k]
                })
                global_id += 1
                
            client.insert(collection_name=collection_name, data=insert_data)
            total_inserted += len(insert_data)

    print(f"\n🎉 THÀNH CÔNG RỰC RỠ! Đã nhồi tổng cộng {total_inserted} Vector vào Database!")
    print(f"📁 Database hiện tại nằm gọn tại: {db_path}")

if __name__ == "__main__":
    build_milvus_local()
```

---

### CELL 6: LLM QUERY REWRITE
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
        last_token_str = self.tokenizer.decode(input_ids[0][-1])
        if '}' in last_token_str:
            return True
        return False

def parse_llm_json(raw_text, original_query):
    fallback_result = {
        "original_query": original_query,
        "cau_hoi_phap_ly": original_query,
        "tu_khoa": []
    }
    
    try:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            return fallback_result
            
        json_str = match.group(0)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        data = json.loads(json_str)
        
        cau_hoi_phap_ly = data.get("cau_hoi_phap_ly", "").strip()
        tu_khoa = data.get("tu_khoa", [])
        
        if not cau_hoi_phap_ly:
            cau_hoi_phap_ly = original_query
        if not isinstance(tu_khoa, list):
            tu_khoa = []
            
        return {
            "original_query": original_query,
            "cau_hoi_phap_ly": cau_hoi_phap_ly,
            "tu_khoa": tu_khoa
        }
    except Exception:
        return fallback_result

def main():
    print(f"[1] Đang tải {config.LLM_MODEL_NAME} lên VRAM...")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME, cache_dir=config.LLM_CACHE_DIR)
    
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"

    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=config.LLM_CACHE_DIR
    )
    
    print(f"[2] Đang đọc danh sách câu hỏi từ {config.QUESTIONS_PUBLIC_FILE}...")
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        questions_raw = json.load(f)
        
    query_ids = list(questions_raw.keys())
    if hasattr(config, 'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        query_ids = query_ids[:config.JUST_CHECK_QUERIES]
        
    print(f"Tổng số câu hỏi cần dịch: {len(query_ids)}")
    
    done_data = {}
    if os.path.exists(config.LLM_OUTPUT_QUERIES):
        with open(config.LLM_OUTPUT_QUERIES, 'r', encoding='utf-8') as f:
            done_data = json.load(f)
            
    stopping_criteria = DualStoppingCriteria(tokenizer, timeout_seconds=6.0)
    criteria_list = StoppingCriteriaList([stopping_criteria])
    
    print("\n[3] BẮT ĐẦU DỊCH TRUY VẤN (QUERY REWRITING)...")
    
    for i, qid in enumerate(tqdm(query_ids, desc="Dịch truy vấn")):
        if qid in done_data:
            continue
            
        original_query = questions_raw[qid].get('question', '') if isinstance(questions_raw[qid], dict) else questions_raw[qid]
        
        try:
            messages = [
                {"role": "system", "content": config.LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f"Phân tích câu hỏi sau:\n{original_query}"}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prefill_text = '{\n  "cau_hoi_phap_ly": "'
            prompt += prefill_text
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            stopping_criteria.reset()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    stopping_criteria=criteria_list,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            full_response = prefill_text + generated_text
            if '}' not in full_response:
                full_response += '"\n}'
                
            final_data = parse_llm_json(full_response, original_query)
            done_data[qid] = final_data
            
        except Exception as e:
            print(f" Lỗi câu {qid}: {e} -> Fallback")
            done_data[qid] = {
                "original_query": original_query,
                "cau_hoi_phap_ly": original_query,
                "tu_khoa": []
            }
            
        finally:
            torch.cuda.empty_cache()
            
        if (i + 1) % 20 == 0:
            os.makedirs(os.path.dirname(config.LLM_OUTPUT_QUERIES), exist_ok=True)
            with open(config.LLM_OUTPUT_QUERIES, 'w', encoding='utf-8') as f:
                json.dump(done_data, f, ensure_ascii=False, indent=4)
                
    with open(config.LLM_OUTPUT_QUERIES, 'w', encoding='utf-8') as f:
        json.dump(done_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ HOÀN TẤT DỊCH TRUY VẤN! File lưu tại: {config.LLM_OUTPUT_QUERIES}")

if __name__ == "__main__":
    main()
```

---

### CELL 7: SEARCH BM25
```python
%%writefile search_bm25_low_ram_train.py
import json
import sqlite3
import re
import os
from tqdm import tqdm
import config_train as config

def remove_vietnamese_accents(text):
    s1 = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ"
    s0 = "AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy"
    s = ""
    for c in text:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def tokenize_query(query):
    query = str(query).lower()
    query = remove_vietnamese_accents(query)
    query = re.sub(r'[^\w\s]', ' ', query)
    tokens = [t for t in query.split() if t.strip()]
    return ' OR '.join(tokens)

def main():
    print("--- TÌM KIẾM TỪ KHÓA BẰNG BM25 (SQLITE) ---")
    db_path = config.BM25_DB_PATH
    if not os.path.exists(db_path):
        print("❌ Chưa có Database BM25!")
        return

    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        q_data = json.load(f)

    llm_data = {}
    if os.path.exists(config.LLM_OUTPUT_QUERIES):
        print("✅ Đã tìm thấy file LLM Rewritten Queries, sẽ sử dụng Từ Khóa bổ sung.")
        with open(config.LLM_OUTPUT_QUERIES, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)

    query_ids = list(q_data.keys())
    if hasattr(config, 'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        query_ids = query_ids[:config.JUST_CHECK_QUERIES]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    submission = {}
    
    for qid in tqdm(query_ids, desc="Searching BM25"):
        q_raw = q_data[qid]
        original_question = q_raw.get('question', '') if isinstance(q_raw, dict) else q_raw
        
        query_text = original_question
        if qid in llm_data:
            tu_khoa = llm_data[qid].get("tu_khoa", [])
            if isinstance(tu_khoa, list) and len(tu_khoa) > 0:
                query_text = original_question + " " + " ".join(tu_khoa)

        fts_query = tokenize_query(query_text)

        if not fts_query.strip():
            submission[qid] = {"answer": []}
            continue

        try:
            cursor.execute('''
                SELECT chunk_id, bm25(bm25_docs) as score
                FROM bm25_docs
                WHERE bm25_docs MATCH ?
                ORDER BY score ASC
                LIMIT ?
            ''', (fts_query, config.RETRIEVAL_TOP_K))
            
            results = cursor.fetchall()
            chunk_ids = [row[0] for row in results]
            submission[qid] = {"answer": chunk_ids}
        except Exception as e:
            print(f"Lỗi query: {e}")
            submission[qid] = {"answer": []}

    conn.close()

    os.makedirs(os.path.dirname(config.SUBMISSION_BM25_TOP_K), exist_ok=True)
    with open(config.SUBMISSION_BM25_TOP_K, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 Hoàn tất BM25! Đã lưu tại: {config.SUBMISSION_BM25_TOP_K}")

if __name__ == "__main__":
    main()
```

---

### CELL 8: SEARCH VECTOR
```python
%%writefile search_public_low_ram_train.py
import json
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import config_train as config

def main():
    print("=========================================================")
    print("🔍 TÌM KIẾM VECTOR (SEMANTIC SEARCH) VỚI MILVUS 🔍")
    print("=========================================================\n")

    db_path = config.MILVUS_DB_PATH
    collection_name = config.MILVUS_COLLECTION

    if not os.path.exists(db_path):
        print("❌ Lỗi: Database Milvus không tồn tại!")
        return

    print("[1] Đang tải mô hình Embedding (BGE-M3) lên VRAM để nhúng câu hỏi...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

    print(f"\n[2] Đang tải danh sách câu hỏi từ {config.QUESTIONS_PUBLIC_FILE}...")
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        q_data = json.load(f)

    llm_data = {}
    if os.path.exists(config.LLM_OUTPUT_QUERIES):
        print("✅ Đã tìm thấy file LLM Rewritten Queries, sẽ sử dụng Câu Hỏi Pháp Lý.")
        with open(config.LLM_OUTPUT_QUERIES, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)

    query_ids = list(q_data.keys())
    if hasattr(config, 'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        query_ids = query_ids[:config.JUST_CHECK_QUERIES]

    print(f"\n[3] Kết nối tới Database Milvus: {db_path}...")
    client = MilvusClient(db_path)
    client.load_collection(collection_name)

    print("\n[4] Bắt đầu tìm kiếm từng câu hỏi...")
    submission = {}

    for qid in tqdm(query_ids, desc="Searching Semantic"):
        q_raw = q_data[qid]
        original_question = q_raw.get('question', '') if isinstance(q_raw, dict) else q_raw
        
        question_text = original_question
        if qid in llm_data and "cau_hoi_phap_ly" in llm_data[qid]:
            question_text = llm_data[qid]["cau_hoi_phap_ly"]

        query_vector = model.encode([question_text], normalize_embeddings=True)[0]

        search_res = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=config.RETRIEVAL_TOP_K,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["chunk_id"]
        )

        chunk_ids = []
        if search_res and len(search_res[0]) > 0:
            chunk_ids = [hit['entity']['chunk_id'] for hit in search_res[0]]

        submission[qid] = {"answer": chunk_ids}

    print(f"\n[5] Đang lưu kết quả Top {config.RETRIEVAL_TOP_K} Semantic ra file...")
    os.makedirs(os.path.dirname(config.SUBMISSION_SEMANTIC_TOP_K), exist_ok=True)
    with open(config.SUBMISSION_SEMANTIC_TOP_K, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    print(f"🎉 HOÀN TẤT SEMANTIC SEARCH! Đã lưu tại: {config.SUBMISSION_SEMANTIC_TOP_K}")

if __name__ == "__main__":
    main()
```

---

### CELL 9: RRF FUSION
```python
%%writefile rrf_fusion_train.py
import json
import os
from tqdm import tqdm
import config_train as config

SEMANTIC_FILE = config.SUBMISSION_SEMANTIC_TOP_K
BM25_FILE = config.SUBMISSION_BM25_TOP_K
OUTPUT_FILE = config.SUBMISSION_HYBRID_TOP_K

K_RRF = 60
TOP_K_FINAL = config.RRF_TOP_K

def fuse_rrf():
    print("=========================================================")
    print("🧬 BẮT ĐẦU LAI GHÉP (RRF FUSION) SEMANTIC + BM25 🧬")
    print("=========================================================\n")

    if not os.path.exists(SEMANTIC_FILE) or not os.path.exists(BM25_FILE):
        print("❌ Thiếu file đầu vào! Cần đảm bảo cả 2 file Semantic và BM25 đều đã được tạo.")
        return

    print("[1] Đang tải kết quả từ 2 phương pháp...")
    with open(SEMANTIC_FILE, 'r', encoding='utf-8') as f:
        semantic_data = json.load(f)

    with open(BM25_FILE, 'r', encoding='utf-8') as f:
        bm25_data = json.load(f)

    print(f"\n[2] Đang tính toán điểm RRF (k={K_RRF}) cho từng câu hỏi...")
    hybrid_submission = {}

    for qid in tqdm(semantic_data.keys(), desc="Fusion RRF"):
        semantic_chunks = semantic_data.get(qid, {}).get("answer", [])
        bm25_chunks = bm25_data.get(qid, {}).get("answer", [])

        rrf_scores = {}

        for rank, chunk_id in enumerate(semantic_chunks):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (K_RRF + rank + 1)

        for rank, chunk_id in enumerate(bm25_chunks):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (K_RRF + rank + 1)

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        top_k_chunks = []
        doc_chunk_count = {}
        for chunk, score in sorted_chunks:
            doc_id = chunk.split('_chunk_')[0]

            if doc_chunk_count.get(doc_id, 0) < getattr(config, 'MAX_CHUNKS_PER_DOC', 50):
                top_k_chunks.append(chunk)
                doc_chunk_count[doc_id] = doc_chunk_count.get(doc_id, 0) + 1

            if len(top_k_chunks) == TOP_K_FINAL:
                break

        hybrid_submission[qid] = {"answer": top_k_chunks}

    print(f"\n[3] Đang lưu kết quả Top {TOP_K_FINAL} ra file...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(hybrid_submission, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 HOÀN TẤT LAI GHÉP! Đã lưu kết quả tại {OUTPUT_FILE}")

if __name__ == "__main__":
    fuse_rrf()
```

---

### CELL 10: RERANKER
```python
%%writefile rerank_public_train.py
import json
import os
import sys
import glob
from tqdm import tqdm
from FlagEmbedding import FlagReranker
import config_train as config

TOP_K_FILE = config.SUBMISSION_HYBRID_TOP_K
CORPUS_DIR = config.PROCESSED_DATA_DIR
RERANKER_MODEL = config.RERANKER_MODEL
BATCH_SIZE = config.RERANKER_BATCH_SIZE

def load_data():
    print("[1] Đang tải danh sách câu hỏi...")
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        questions_raw = json.load(f)

    q_dict = {}
    for k, v in questions_raw.items():
        if isinstance(v, dict):
            q_dict[k] = v.get("question", "")
        else:
            q_dict[k] = v

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

    print(f"[3] Cần đọc nội dung của {len(needed_chunks)} đoạn văn (chunks) để chấm điểm.")
    doc_text_dict = {}
    
    chunk_files = []
    for d in glob.glob(os.path.join(CORPUS_DIR, "*")):
        if os.path.isdir(d):
            jsonl_path = os.path.join(d, "chunks.jsonl")
            if os.path.exists(jsonl_path):
                chunk_files.append(jsonl_path)
                
    for fpath in tqdm(chunk_files, desc="Đọc file chunks"):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                c_id = f"{data['doc_id']}_{data['chunk_id']}"
                if c_id in needed_chunks:
                    doc_text_dict[c_id] = data.get('text', '')

    return q_dict, top_k_data, doc_text_dict, rewritten_data

def main():
    print("=========================================================")
    print("🏆 BẮT ĐẦU CHẤM ĐIỂM CHUNG KẾT (RERANKING) 🏆")
    print("=========================================================\n")

    q_dict, top_k_data, doc_text_dict, rewritten_data = load_data()

    print(f"\n[4] Đang nạp mô hình Reranker ({RERANKER_MODEL}) lên VRAM...")
    reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
    
    # [ANA FIX]: BGE-Reranker-v2-m3 LÀ Cross Encoder, phải đặt True!
    use_cross_encoder = True

    print("\n[5] Bắt đầu quá trình chấm điểm chéo (Reranking)...")
    final_submission = {}

    for qid, data in tqdm(top_k_data.items(), desc="Reranking"):
        question_text = q_dict.get(qid, "")

        if qid in rewritten_data and "cau_hoi_phap_ly" in rewritten_data[qid]:
            combined_query = f"{question_text}\nÝ định tìm kiếm: {rewritten_data[qid]['cau_hoi_phap_ly']}"
        else:
            combined_query = question_text

        candidate_chunks = data['answer']
        pairs = []
        valid_chunks = []
        for chunk_id in candidate_chunks:
            if not chunk_id:
                continue
            text = doc_text_dict.get(chunk_id, "")
            pairs.append([combined_query, text])
            valid_chunks.append(chunk_id)

        if not pairs:
            final_submission[qid] = {"answer": []}
            continue

        try:
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    if use_cross_encoder:
                        # [ANA FIX] Đã thêm show_progress_bar=False để chống rác màn hình
                        scores = reranker.compute_score(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
                    else:
                        scores = reranker.compute_score(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
                finally:
                    sys.stderr = old_stderr
        except Exception as e:
            print(f"\n  Lỗi rerank câu {qid}: {e}")
            scores = [0.0] * len(pairs)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = [float(scores)]
        else:
            scores = [float(s) for s in scores]

        chunk_scores = list(zip(valid_chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        top5_doc_ids = []
        seen_docs = set()
        for chunk, score in chunk_scores:
            doc_id = chunk.split('_chunk_')[0]
            if doc_id not in seen_docs:
                top5_doc_ids.append(doc_id)
                seen_docs.add(doc_id)
            if len(top5_doc_ids) >= 5:
                break

        final_submission[qid] = {"answer": top5_doc_ids}

    print(f"\n[6] Đang lưu kết quả nộp bài ra file...")
    os.makedirs(os.path.dirname(config.SUBMISSION_FINAL), exist_ok=True)
    with open(config.SUBMISSION_FINAL, 'w', encoding='utf-8') as f:
        json.dump(final_submission, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 HOÀN TẤT PIPELINE! Kết quả được lưu tại: {config.SUBMISSION_FINAL}")

if __name__ == "__main__":
    main()
```

---

### CELL 11: BẢNG ĐIỀU KHIỂN (RUN PIPELINE)
> Khối này dùng để kích hoạt các file Python ở trên. Hãy chạy khối này cuối cùng.
```bash
# Chạy lần lượt toàn bộ Pipeline (flag -u = unbuffered, log hiện ngay lập tức)
!python -u preprocess_train.py
!python -u build_bm25_train.py
!python -u embed_train.py
!python -u build_local_milvus_train.py

# Bước 5 có thể tắt đi bằng cách cho dấu # vào đầu dòng nếu không muốn xài LLM
!python -u preprocess_queries_train.py

!python -u search_bm25_low_ram_train.py
!python -u search_public_low_ram_train.py
!python -u rrf_fusion_train.py
!python -u rerank_public_train.py

print("✅ ĐÃ CHẠY XONG TOÀN BỘ PIPELINE. HÃY TẢI FILE SUBMISSION.JSON TRONG THƯ MỤC WORKING VỀ NỘP KAGGLE!")
```
