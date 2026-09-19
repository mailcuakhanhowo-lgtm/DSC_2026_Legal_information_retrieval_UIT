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
