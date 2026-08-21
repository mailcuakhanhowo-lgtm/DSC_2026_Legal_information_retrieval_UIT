import os
import re
import json
import glob
import unicodedata
import logging
# CẤU HÌNH THÔNG SỐ (Thay thế cho config.py)
INPUT_PATH = "/kaggle/input/datasets/nguynquckhanh/dsc-2026-srcndata2/_kaggle/selected-contexts/selected-contexts"
OUTPUT_DIR = "/kaggle/working/processed_data"
MAX_WORDS_PER_CHUNK = 6000
OVERLAP_WORDS = 50
JUST_CHECK = None  # Đặt thành số (vd: 10) nếu chỉ muốn test thử 10 file đầu

def filter_and_clean_text(text: str) -> str:
    """
    Sàng lọc rác pháp lý và làm sạch văn bản (Noise Filtering & Text Cleaning).
    """
    if not text:
        return ""
        
    # 1. Chuẩn hóa Unicode tiếng Việt về NFC
    text = unicodedata.normalize('NFC', text)
    
    # 1.5. Lọc phần Mở đầu (Preamble) để xóa "Cộng hòa xã hội...", giữ lại từ Căn cứ/Quyết định
    match_start = re.search(r'(QUYẾT ĐỊNH:|NGHỊ QUYẾT:|CĂN CỨ:|Điều 1\.)', text)
    if match_start:
        text = text[match_start.start():]
        
    # 2. Xóa phần thủ tục nơi nhận, chữ ký
    text = re.sub(
        r'(Nơi nhận:|KT\. BỘ TRƯỞNG).*?(?=(?:\n\s*(?:QUY CHẾ|PHỤ LỤC|Chương \d+|Điều \d+\.))|\Z)', 
        '', 
        text, 
        flags=re.DOTALL
    )
        
    # [ANA_FIX] 2.5 Nối các dòng bị đứt (dòng trước không kết thúc bằng dấu câu . : ; ! ?)
    # Xử lý các lỗi xuống dòng vô cớ do PDF Extraction sinh ra
    text = re.sub(r'(?<![\.\:\;\!\?])\s*\n\s*(?=[A-Za-zà-ỹ0-9])', ' ', text)
        
    # [ANA_FEATURE] 2.6 Phục hồi cấu trúc (Structure Restoration)
    # Bẻ các Điều, Khoản, Điểm bị dính liền ra thành các dòng riêng biệt
    text = re.sub(r'(?<=[a-zà-ỹ0-9\.\:\;\!\?])\s+(Điều\s+\d+\.\s+[A-ZÀ-Ỹ])', r'\n\n\1', text)
    text = re.sub(r'(?<=[a-zà-ỹ0-9\.\:\;\!\?])\s+(Chương\s+[IVXLCDM]+\.?\s+[A-ZÀ-Ỹ])', r'\n\n\1', text)
    
    # [ANA_FIX] Cấm bẻ Khoản (1., 2.) nếu đứng trước nó là Điều, Khoản, Chương, Mục, Phần (Tránh chém đôi "Điều 2.")
    text = re.sub(r'(?<!Điều)(?<!Chương)(?<!Mục)(?<!Phần)(?<!Khoản)(?<!Điểm)(?<=[a-zà-ỹ0-9\.\:\;\!\?])\s+(\d+\.\s+[A-ZÀ-Ỹ])', r'\n\n\1', text)
    
    # Cấm bẻ Điểm (a), b)) nếu đứng trước nó là "điểm" hoặc "khoản" (Tránh chém đôi "tại điểm a)")
    text = re.sub(r'(?<![Đđ]iểm)(?<![Kk]hoản)(?<=[a-zà-ỹ0-9\.\:\;\!\?])\s+([a-zà-ỹ]\)\s+[A-ZÀ-Ỹa-zà-ỹ])', r'\n\n\1', text)
        
    # 3. Thu gọn các khoảng trắng thừa trên cùng một dòng (giữ lại dấu xuống dòng \n)
    text = re.sub(r'[ \t]+', ' ', text)
        
    # 4. Thu gọn khoảng trắng thừa nhiều dòng
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def stateful_legal_chunker(text: str, max_words: int = 300, overlap_words: int = 50) -> list:
    """
    Hàm phân đoạn văn bản pháp luật dựa trên State Machine (Propositional Chunking).
    Đảm bảo 100% Chunk nhỏ đều được nhồi ngữ cảnh của Điều và Khoản cha.
    """
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
            
        # Nhận diện cấp bậc
        is_article = re.match(r'^(Điều\s+\d+|Chương\s+[IVXLCDM]+|PHỤ LỤC|MỤC)', line)
        is_clause = re.match(r'^\d+\.\s+', line)
        is_point = re.match(r'^([a-zà-ỹ]\)|-|\+)\s+', line)
        
        if is_article:
            flush_chunk()
            current_article = line
            current_clause_lead = ""  # Reset câu dẫn khi sang Điều mới
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
            # Dòng thường: Cộng dồn vào chunk hiện tại
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
            
    # LỚP 2: Sliding Window giới hạn độ dài an toàn tuyệt đối
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


def process_corpus_to_md(input_dir: str, output_dir: str, base_max_words=600, overlap=50):
    """
    Pipeline xử lý dữ liệu thô sang các file .md trong thư mục riêng cho từng document.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    json_files = glob.iglob(os.path.join(input_dir, "*.json"))
    
    total_docs = 0
    total_chunks = 0
    
    from tqdm import tqdm
    json_files_list = list(json_files)
    
    # [ANA_FEATURE]: Áp dụng giới hạn số lượng file chạy thử nghiệm
    if JUST_CHECK is not None:
        json_files_list = json_files_list[:JUST_CHECK]
        print(f"\n[TEST MODE] Đã bật chế độ test, chỉ xử lý {len(json_files_list)} văn bản đầu tiên...")
        
    for file_path in tqdm(json_files_list, desc="Tiền xử lý", unit="file"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
        except Exception as e:
            logging.warning(f"Lỗi đọc file {file_path}: {e}")
            continue
            
        doc_id = str(doc.get("id", "")).strip()
        title = doc.get("name", doc.get("title", "")).strip()
        passage = doc.get("passage", "").strip()
        
        if not doc_id or not passage:
            logging.warning(f"File {file_path} bị bỏ qua do thiếu 'id' hoặc 'passage'.")
            continue
            
        # Tạo thư mục riêng cho document này
        doc_dir = os.path.join(output_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
            
        # 1. Làm sạch văn bản thô
        cleaned_text = filter_and_clean_text(passage)
        
        # 2. Tính toán offset không gian cho Header Injection
        title_words = title.split()
        if len(title_words) > 60:
            title = " ".join(title_words[:60]) + "..."
            
        title_word_count = len(title.split())
        
        # Đảm bảo (Title + Chunk) luôn <= base_max_words
        dynamic_max_words = max(100, base_max_words - title_word_count)
        
        # 3. Tách đoạn ngữ nghĩa bằng thuật toán Stateful
        raw_chunks = stateful_legal_chunker(cleaned_text, max_words=dynamic_max_words, overlap_words=overlap)
        
        # 4. Ghi dữ liệu dạng file .jsonl (Gom tất cả chunk của 1 doc vào 1 file)
        jsonl_path = os.path.join(doc_dir, "chunks.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            for i, chunk_raw in enumerate(raw_chunks):
                chunk_id = f"chunk_{i}"
                
                # Header Injection
                text_with_header = f"# {title}\n\n{chunk_raw}" if title else chunk_raw
                
                # Tạo bản ghi JSON cho mỗi chunk
                record = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": text_with_header
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_chunks += 1
            
        total_docs += 1
            
    print(f"\nHOÀN THÀNH TIỀN XỬ LÝ CORPUS SANG FILE JSONL!")
    print(f"Tổng số văn bản xử lý thành công: {total_docs}")
    print(f"Tổng số chunks đã tạo: {total_chunks}")
    print(f"Thư mục lưu trữ: {output_dir}")
    

if __name__ == "__main__":
    if os.path.exists(INPUT_PATH):
        print(f"Bắt đầu tiền xử lý (Max words: {MAX_WORDS_PER_CHUNK}, Overlap: {OVERLAP_WORDS})")
        process_corpus_to_md(INPUT_PATH, OUTPUT_DIR, base_max_words=MAX_WORDS_PER_CHUNK, overlap=OVERLAP_WORDS)
    else:
        print(f"❌ LỖI: Không tìm thấy thư mục {INPUT_PATH}.")
        print("Sếp nhớ chỉnh lại biến INPUT_PATH ở đầu code cho khớp với đường dẫn dataset trên Kaggle nhé!")
