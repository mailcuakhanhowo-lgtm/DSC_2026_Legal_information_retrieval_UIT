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
!pip install -q -U accelerate>=1.14.0 bitsandbytes>=0.50.1 flagembedding>=1.3.0 pyvi>=0.1.1 "sentence-transformers>=6.0.0" "transformers==5.15.0"
print("✅ Cài đặt xong!")
```

---

### CELL 1: CẤU HÌNH TRUNG TÂM
```python
%%writefile config_train.py
# ============================================================
# PHASE 2 CONFIG - Bỏ qua Embedding & Milvus, tìm kiếm numpy trực tiếp
# ⚠️ Cập nhật EMBEDED_DATA_INPUT và QUESTIONS_PUBLIC_FILE
#    dựa theo kết quả cell debug ở trên!
# ============================================================
import os

# ================= 1. CẤU HÌNH THƯ MỤC =================
DEFAULT_INPUT_PATH = r"/kaggle/input/datasets/nguynquckhanh/selected-context/selected-contexts"
PROCESSED_DATA_DIR = r"/kaggle/working/train/processed_data"
EMBEDED_DATA_DIR   = r"/kaggle/working/train/embeded_data"

# Đã cập nhật chính xác theo ảnh (chú ý có thư mục lồng nhau)
EMBEDED_DATA_INPUT = r"/kaggle/input/datasets/nguynquckhanh/embeded-data/embeded_data"

# Đã cập nhật chính xác theo ảnh
QUESTIONS_PUBLIC_FILE = r"/kaggle/input/datasets/nguynquckhanh/public-test/public-official.json"

JUST_CHECK_QUERIES = None
JUST_CHECK_CONTEXT = None

# ================= 2. CHUNKING =================
MAX_WORDS_PER_CHUNK = 6000
OVERLAP_WORDS = 50

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
EMBEDDING_MODEL           = "BAAI/bge-m3"
EMBEDDING_DIM             = 1024
MAX_POSITION_EMBEDDINGS   = 2048
EMBEDDING_BATCH_SIZE      = 4
# Không cần MILVUS — tìm kiếm trực tiếp bằng numpy từ EMBEDED_DATA_INPUT
SUBMISSION_SEMANTIC_TOP_K = r"/kaggle/working/train/submission_semantic_topK.json"

BM25_DB_PATH          = r"/kaggle/working/train/embeded_data/bm25_index.db"
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
```

---

### CELL 2: PREPROCESS (Chạy lại nhanh ~5 phút để lấy lại processed_data cho BM25 và Reranker)
```python
%%writefile preprocess_train.py
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
        is_clause  = re.match(r'^\d+\.\s+', line)
        is_point   = re.match(r'^([a-zà-ỹ]\)|-|\+)\s+', line)

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
            if current_article:     chunk_prefix += current_article + " "
            if current_clause_lead: chunk_prefix += current_clause_lead + " "
            current_chunk = chunk_prefix + line
        else:
            if line.endswith(':'):
                current_clause_lead = line
            if not current_chunk:
                chunk_prefix = ""
                if current_article:     chunk_prefix += current_article + " "
                if current_clause_lead: chunk_prefix += current_clause_lead + " "
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
            if step <= 0: step = max_words
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
                record = {"doc_id": doc_id, "chunk_id": chunk_id, "title": title, "text": text_with_header}
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_chunks += 1
        total_docs += 1

    print(f"\n🎉 HOÀN THÀNH! Xử lý {total_docs} văn bản, tạo {total_chunks} chunks.")

if __name__ == "__main__":
    process_corpus_to_md()
```

---

### CELL 3: BUILD BM25 (~2 phút)
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
    return "".join(s0[s1.index(c)] if c in s1 else c for c in text)

def tokenize_vietnamese(text):
    text = text.lower()
    text = remove_vietnamese_accents(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

def build_fts5_db():
    print("--- BẮT ĐẦU XÂY DỰNG TỪ ĐIỂN BM25 (SQLITE FTS5) ---")
    db_path  = config.BM25_DB_PATH
    data_dir = config.PROCESSED_DATA_DIR
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE VIRTUAL TABLE bm25_docs USING fts5(
            doc_id UNINDEXED, chunk_id UNINDEXED, title UNINDEXED, content,
            tokenize="unicode61 remove_diacritics 1"
        )
    ''')

    chunk_files = [
        os.path.join(d, "chunks.jsonl")
        for d in glob.glob(os.path.join(data_dir, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "chunks.jsonl"))
    ]
    print(f"Tìm thấy {len(chunk_files)} file chunks.")

    insert_count = 0
    for chunk_file in tqdm(chunk_files, desc="Đang nạp FTS5"):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                tokenized = tokenize_vietnamese(data.get('title','') + " " + data.get('text',''))
                cursor.execute(
                    "INSERT INTO bm25_docs (doc_id, chunk_id, title, content) VALUES (?, ?, ?, ?)",
                    (data.get('doc_id'), data.get('chunk_id'), data.get('title',''), tokenized)
                )
                insert_count += 1

    conn.commit()
    conn.close()
    print(f"🎉 Đã nạp thành công {insert_count} chunks vào SQLite BM25 FTS5!")

if __name__ == "__main__":
    build_fts5_db()
```

---

### CELL 4: EMBED VECTORS (~60–90 phút, chỉ cần chạy 1 lần rồi upload .npy lên Kaggle Dataset)
```python
%%writefile embed_train.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # CHỐNG PHÂN MẢNH VRAM

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
    out_dir  = config.EMBEDED_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("\n--- BƯỚC 1: ĐỌC DỮ LIỆU TỪ JSONL ---")
    doc_dirs    = glob.glob(os.path.join(data_dir, "*"))
    chunk_files = [
        os.path.join(d, "chunks.jsonl")
        for d in doc_dirs
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "chunks.jsonl"))
    ]

    all_texts    = []
    all_metadata = []

    print(f"Đang nạp toàn bộ nội dung từ {len(chunk_files)} file văn bản vào RAM...")
    for file_path in tqdm(chunk_files, desc="Đọc files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data     = json.loads(line)
                doc_id   = data.get("doc_id")
                chunk_id = data.get("chunk_id")
                text     = data.get("text", "")
                all_texts.append(text)
                all_metadata.append({
                    "chunk_id": f"{doc_id}_{chunk_id}",
                    "title":    data.get("title", ""),
                    "link":     ""
                })

    total_chunks = len(all_texts)
    print(f"🎉 Đã nạp xong {total_chunks:,} đoạn văn bản từ {len(chunk_files)} file!")

    print("\n--- BƯỚC 2: KHỞI TẠO MÔ HÌNH AI ---")
    print("💻 Đang kích hoạt thiết bị: Multi-GPU (Tự động phát hiện)")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("🚀 Bật chế độ Đa GPU (Multi-Process Pool)...")
    pool = model.start_multi_process_pool()

    print("\n--- BƯỚC 3: VECTOR HÓA & LƯU TỪNG PHẦN ---")
    chunk_size = 100000
    num_parts  = (total_chunks + chunk_size - 1) // chunk_size

    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx   = min((i + 1) * chunk_size, total_chunks)
        print(f"\n🚀 Đang xử lý Phần {i+1}: Từ {start_idx} đến {end_idx}...")

        batch_texts = all_texts[start_idx:end_idx]
        batch_meta  = all_metadata[start_idx:end_idx]

        embeddings = model.encode(
            batch_texts,
            pool=pool,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        emb_path  = os.path.join(out_dir, f"embeddings_part_{i+1}.npy")
        meta_path = os.path.join(out_dir, f"metadata_part_{i+1}.json")

        np.save(emb_path, np.array(embeddings).astype(np.float32))
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(batch_meta, f, ensure_ascii=False)

        print(f"✅ Đã lưu Phần {i+1} xong! (File: {emb_path})")

    model.stop_multi_process_pool(pool)
    print(f"\n🎉 XONG TOÀN BỘ! Vector đã lưu tại: {out_dir}")
    print("👉 Tiếp theo: Upload toàn bộ file .npy và .json trong thư mục đó lên Kaggle Dataset!")

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

### CELL 6: SEARCH BM25
```python
%%writefile search_bm25_low_ram_train.py
import json, sqlite3, re, os
from tqdm import tqdm
import config_train as config

def remove_vietnamese_accents(text):
    s1 = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ"
    s0 = "AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy"
    return "".join(s0[s1.index(c)] if c in s1 else c for c in text)

def tokenize_query(query):
    query = remove_vietnamese_accents(str(query).lower())
    query = re.sub(r'[^\w\s]',' ',query)
    return ' OR '.join([t for t in query.split() if t.strip()])

def main():
    print("--- TÌM KIẾM TỪ KHÓA BẰNG BM25 ---")
    if not os.path.exists(config.BM25_DB_PATH):
        print("❌ Chưa có Database BM25!"); return

    with open(config.QUESTIONS_PUBLIC_FILE,'r',encoding='utf-8') as f: q_data = json.load(f)
    llm_data = {}
    if os.path.exists(config.LLM_OUTPUT_QUERIES):
        with open(config.LLM_OUTPUT_QUERIES,'r',encoding='utf-8') as f: llm_data = json.load(f)

    query_ids = list(q_data.keys())
    if hasattr(config,'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        query_ids = query_ids[:config.JUST_CHECK_QUERIES]

    conn = sqlite3.connect(config.BM25_DB_PATH)
    cursor = conn.cursor()
    submission = {}

    for qid in tqdm(query_ids, desc="Searching BM25"):
        q_raw = q_data[qid]
        original_question = q_raw.get('question','') if isinstance(q_raw,dict) else q_raw
        query_text = original_question
        if qid in llm_data:
            tu_khoa = llm_data[qid].get("tu_khoa",[])
            if isinstance(tu_khoa,list) and tu_khoa:
                query_text = original_question + " " + " ".join(tu_khoa)
        fts_query = tokenize_query(query_text)
        if not fts_query.strip():
            submission[qid] = {"answer": []}; continue
        try:
            cursor.execute('SELECT chunk_id FROM bm25_docs WHERE bm25_docs MATCH ? ORDER BY bm25(bm25_docs) ASC LIMIT ?',
                           (fts_query, config.RETRIEVAL_TOP_K))
            submission[qid] = {"answer": [row[0] for row in cursor.fetchall()]}
        except Exception as e:
            print(f"Lỗi query: {e}")
            submission[qid] = {"answer": []}

    conn.close()
    os.makedirs(os.path.dirname(config.SUBMISSION_BM25_TOP_K), exist_ok=True)
    with open(config.SUBMISSION_BM25_TOP_K,'w',encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
    print(f"🎉 Hoàn tất BM25! Lưu tại: {config.SUBMISSION_BM25_TOP_K}")

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
            text = llm_data[qid]["cau_hoi_phap_ly"]
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
            doc_id = chunk.split('_chunk_')[0]
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
import glob
from tqdm import tqdm
import multiprocessing as mp
import config_train as config

TOP_K_FILE = config.SUBMISSION_HYBRID_TOP_K
CORPUS_DIR = config.PROCESSED_DATA_DIR
RERANKER_MODEL = config.RERANKER_MODEL
BATCH_SIZE = config.RERANKER_BATCH_SIZE

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

    print(f"[3] Cần đọc nội dung của {len(needed_chunks)} đoạn văn (chunks) để chấm điểm.")
    doc_text_dict = {}
    
    chunk_files = [
        os.path.join(d, "chunks.jsonl")
        for d in glob.glob(os.path.join(CORPUS_DIR, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "chunks.jsonl"))
    ]
                
    for fpath in tqdm(chunk_files, desc="Đọc file chunks"):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                c_id = f"{data['doc_id']}_{data['chunk_id']}"
                if c_id in needed_chunks:
                    doc_text_dict[c_id] = data.get('text', '')

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
            combined_query = f"{question_text}\nÝ định tìm kiếm: {rewritten_data[qid]['cau_hoi_phap_ly']}"
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
            doc_id = chunk.split('_chunk_')[0]
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
# PIPELINE HOÀN CHỈNH — Chạy theo 2 giai đoạn:
#
# ── GIAI ĐOẠN A: Chạy LẦN ĐẦU (chỉ cần 1 lần, rồi upload .npy lên Dataset)
#    Bước 1: Preprocess → Bước 2: BM25 → Bước 3: Embed → upload .npy
#
# ── GIAI ĐOẠN B: PHASE 2 (đã có .npy trên Dataset, bỏ qua Embed & Milvus)
#    Bước 1: Preprocess → Bước 2: BM25 → Bước 3: LLM (tuỳ chọn)
#    → Bước 4: BM25 Search → Bước 5: Semantic Search (.npy)
#    → Bước 6: RRF → Bước 7: Rerank → Submission
# ============================================================

# ──────────────────────────────────────────────────────────
# GIAI ĐOẠN A: Tạo vector lần đầu (bỏ comment nếu chạy lần đầu)
# ──────────────────────────────────────────────────────────
# !python -u preprocess_train.py
# !python -u build_bm25_train.py
# !python -u embed_train.py   # Tạo file .npy → upload lên Kaggle Dataset

# ──────────────────────────────────────────────────────────
# GIAI ĐOẠN B: PHASE 2 — Đã có .npy trên Dataset, không cần Embed lại
# ──────────────────────────────────────────────────────────

# Bước 1 & 2: Tái tạo processed_data và BM25 (~7 phút)
!python -u preprocess_train.py
!python -u build_bm25_train.py

# Bước 3: LLM Query Rewrite (comment # nếu muốn bỏ qua)
!python -u preprocess_queries_train.py

# Bước 4–7: Search (numpy trực tiếp, không cần Milvus!) → RRF → Rerank → Submission
!python -u search_bm25_low_ram_train.py
!python -u search_public_low_ram_train.py
!python -u rrf_fusion_train.py
# Chạy Reranker bằng Multiprocessing Pool trên cả 2 GPU
!python -u rerank_public_train.py

print("✅ PHASE 2 HOÀN TẤT! Tải file /kaggle/working/train/submission.json về nộp!")
```
