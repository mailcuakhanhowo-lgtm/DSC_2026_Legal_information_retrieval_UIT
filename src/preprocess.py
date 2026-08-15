import os
import re
import json
import glob
import unicodedata
import logging
import argparse
import config

# Thiết lập Logging hệ thống
logging.basicConfig(filename='pipeline_errors.log', level=logging.WARNING, 
                    format='%(asctime)s - %(message)s')

def filter_and_clean_text(text: str) -> str:
    """
    Sàng lọc rác pháp lý và làm sạch văn bản (Noise Filtering & Text Cleaning).
    """
    if not text:
        return ""
        
    # 1. Chuẩn hóa Unicode tiếng Việt về NFC
    text = unicodedata.normalize('NFC', text)
    
    # 2. Xóa phần thủ tục nơi nhận, chữ ký
    text = re.sub(
        r'(Nơi nhận:|KT\. BỘ TRƯỞNG).*?(?=(?:\n\s*(?:QUY CHẾ|PHỤ LỤC|Chương \d+|Điều \d+\.))|\Z)', 
        '', 
        text, 
        flags=re.DOTALL
    )
        
    # 3. Thu gọn các khoảng trắng thừa trên cùng một dòng (giữ lại dấu xuống dòng \n)
    text = re.sub(r'[ \t]+', ' ', text)
        
    # 4. Thu gọn khoảng trắng thừa nhiều dòng
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def smart_legal_chunker(text: str, max_words: int = 600, overlap_words: int = 50) -> list:
    """
    Hàm phân đoạn văn bản pháp luật 2 lớp (Two-layer Hybrid Chunking).
    """
    if not text or not text.strip():
        return []

    # LỚP 1: SEMANTIC CHUNKING 
    pattern = r'(?=(?:\n|^)\s*(?:' \
              r'Điều\s+\d+[a-zA-Z]*[\.\:]|' \
              r'(?:Chương|MỤC|Mục|PHỤ LỤC|Phụ lục)\s+(?:[IVXLCDM\d]+|[A-Z])|' \
              r'\d+(?:\.\d+)*[\.\)]\s*|' \
              r'[IVXLCDM]+\.\s+|' \
              r'[a-zà-ỹ]\)' \
              r'))'

    # Băm văn bản giữ lại tiêu đề ở đầu mỗi đoạn
    sections = re.split(pattern, text)
    final_chunks = []

    # LỚP 2: FALLBACK SLIDING WINDOW 
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        # Tách theo dấu cách đơn ' ' để đếm từ
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


def process_corpus_to_jsonl(input_dir: str, output_jsonl_path: str, base_max_words=600, overlap=50):
    """
    Pipeline xử lý dữ liệu thô sang chuẩn JSONL cho Vector Database & BM25.
    """
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        

    json_files = glob.iglob(os.path.join(input_dir, "*.json"))
    
    total_docs = 0
    total_chunks = 0
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
            except Exception as e:
                logging.warning(f"Lỗi đọc file {file_path}: {e}")

                continue
                
            doc_id = str(doc.get("id", "")).strip()
            title = doc.get("name", doc.get("title", "")).strip()
            passage = doc.get("passage", "").strip()
            link = doc.get("link", "").strip()
            
            if not doc_id or not passage:
                logging.warning(f"File {file_path} bị bỏ qua do thiếu 'id' hoặc 'passage'.")
                continue
                
            # 1. Làm sạch văn bản thô
            cleaned_text = filter_and_clean_text(passage)
            
            # 2. Tính toán offset không gian cho Header Injection
            title_words = title.split()
            if len(title_words) > 60:
                title = " ".join(title_words[:60]) + "..."
                
            title_word_count = len(title.split())
            
            # Đảm bảo (Title + Chunk) luôn <= base_max_words
            dynamic_max_words = max(100, base_max_words - title_word_count)
            
            # 3. Tách đoạn ngữ nghĩa
            raw_chunks = smart_legal_chunker(cleaned_text, max_words=dynamic_max_words, overlap_words=overlap)
            
            # 4. Ghi dữ liệu dạng JSONL
            for i, chunk_raw in enumerate(raw_chunks):
                chunk_id = f"{doc_id}_{i}"
                
                # Header Injection
                text_with_header = f"# {title}\n\n{chunk_raw}" if title else chunk_raw
                
                record = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": text_with_header,
                    "raw_text": chunk_raw,
                    "word_count": len(text_with_header.split(' ')), 
                    "link": link
                }
                
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_chunks += 1
                
            total_docs += 1
            
    print(f"\nHOÀN THÀNH TIỀN XỬ LÝ CORPUS!")
    print(f"Tổng số văn bản xử lý thành công: {total_docs}")
    print(f"Tổng số chunks đã tạo: {total_chunks}")
    print(f"File lưu tại: {output_jsonl_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiền xử lý văn bản pháp luật (Sàng lọc & Chunking).")
    parser.add_argument("--input", type=str, default=config.DEFAULT_INPUT_PATH, help=f"Đường dẫn file JSON (Mặc định: {config.DEFAULT_INPUT_PATH})")
    parser.add_argument("--output", type=str, default=config.DEFAULT_OUTPUT_DIR, help=f"Thư mục đầu ra (Mặc định: {config.DEFAULT_OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    max_w = config.MAX_WORDS_PER_CHUNK
    overlap_w = config.OVERLAP_WORDS
    
    # Nếu file tồn tại (do user truyền hoặc file default có thật)
    if os.path.exists(input_path):
        print(f"Bắt đầu tiền xử lý (Max words: {max_w}, Overlap: {overlap_w})")
        process_corpus_to_jsonl(input_path, output_path, max_words=max_w, overlap=overlap_w)
    else:
        # KHỐI KIỂM TRA NGHIỆM THU (Manual Verification) - Nếu không truyền tham số
        print("=== CHẠY THỬ NGHIỆM (DRY RUN) ===")
        print("Gợi ý chạy thật: python src/preprocess.py --input warmup.json --output processed_data\n")
        
        sample_doc = {
            "id": "740",
            "name": "Quyết định số 5868/QĐ-UBND",
            "passage": "Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015;\nTheo đề nghị của Giám đốc Sở Tư pháp.\nQUYẾT ĐỊNH:\nĐiều 1. Ban hành Quy chế làm việc.\nĐiều 2. Quy chế này có hiệu lực từ ngày ký.\nNơi nhận:\n- Như Điều 2;\n- Lưu: VT."
        }
        
        print(f"[1] Văn bản gốc:\n{sample_doc['passage']}\n")
        
        cleaned = filter_and_clean_text(sample_doc['passage'])
        print(f"[2] Sau khi lọc rác:\n{cleaned}\n")
        
        chunks = smart_legal_chunker(cleaned, max_words=max_w, overlap_words=overlap_w)
        print(f"[3] Cắt đoạn (Số lượng chunk: {len(chunks)}):")
        for idx, c in enumerate(chunks):
            print(f"--- Chunk {idx} ---\n# {sample_doc['name']}\n{c}\n")
