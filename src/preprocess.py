import os
import re
import json
import unicodedata

def filter_and_clean_text(text: str) -> str:
    """
    Sàng lọc rác pháp lý và làm sạch văn bản (Noise Filtering & Text Cleaning).
    """
    if not text:
        return ""
        
    # 1. Chuẩn hóa Unicode tiếng Việt về NFC
    text = unicodedata.normalize('NFC', text)
    
    # 2. Xóa bỏ ký tự rác sinh ra do lỗi copy, nối các từ bị đứt gãy
    # Ví dụ: BAN\r\n\nHÀNH -> BAN HÀNH
    text = re.sub(r'([A-ZÀ-Ỹ])\r?\n\s*\r?\n([A-ZÀ-Ỹ])', r'\1 \2', text)
    
    # 3. Lọc phần Mở đầu (Preamble)
    # Giữ lại từ "QUYẾT ĐỊNH:", "NGHỊ QUYẾT:", "CĂN CỨ:" hoặc "Điều 1."
    match_start = re.search(r'(QUYẾT ĐỊNH:|NGHỊ QUYẾT:|Điều 1\.)', text)
    if match_start:
        text = text[match_start.start():]
        
    # 4. Lọc phần Chữ ký & Nơi nhận (Signatures & Recipients)
    # Thay vì cắt cụt toàn bộ văn bản từ "Nơi nhận:", ta chỉ xóa phần khối Nơi nhận/Chữ ký
    # cho đến khi gặp phần văn bản đính kèm (QUY CHẾ, PHỤ LỤC, Chương, hoặc Điều mới).
    text = re.sub(r'(Nơi nhận:|KT\. BỘ TRƯỞNG).*?(?=(QUY CHẾ|PHỤ LỤC|Chương \d+|Điều \d+\.|\Z))', '', text, flags=re.DOTALL)
        
    # 5. Thu gọn khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def smart_legal_chunker(text: str, max_words=6000, overlap_words=50) -> list:
    """
    Chiến thuật Cắt đoạn 2 lớp (Two-layer Hybrid Chunking).
    Sử dụng Word Count làm Proxy cho Token Count (8192 tokens ~ 6000 words).
    """
    chunks = []
    
    # LỚP 1: Semantic Chunking (Tách đoạn theo Điều)
    # Sử dụng Lookahead regex để băm ngay trước chữ "Điều [số]."
    articles = re.split(r'(?=Điều \d+\.)', text)
    
    for article in articles:
        article = article.strip()
        if not article:
            continue
            
        words = article.split()
        
        # LỚP 2: Fallback Sliding Window (Nếu Điều quá dài)
        if len(words) <= max_words:
            chunks.append(article)
        else:
            step = max_words - overlap_words
            for i in range(0, len(words), step):
                chunk_words = words[i:i + max_words]
                chunks.append(" ".join(chunk_words))
                
    return chunks

def process_corpus(input_path: str, output_dir: str, max_words=6000, overlap=50):
    """
    Pipeline chính: Đọc JSON (file hoặc thư mục) -> Clean -> Chunk -> Tạo file .md (Small2Big Markdown).
    """
    import glob
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files_to_process = []
    if os.path.isdir(input_path):
        files_to_process = glob.glob(os.path.join(input_path, "*.json"))
        print(f"Đang xử lý thư mục chứa {len(files_to_process)} file JSON...")
    else:
        files_to_process = [input_path]
        print(f"Đang đọc dữ liệu từ: {input_path}...")
        
    total_count = 0
    
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        except Exception as e:
            print(f"Lỗi đọc file {file_path}: {e}")
            continue

        # Nếu file là list các dict, duyệt qua từng phần tử
        if not isinstance(documents, list):
            if isinstance(documents, dict):
                # Trường hợp 1: File JSON chỉ chứa đúng 1 văn bản
                if "passage" in documents and "id" in documents:
                    documents = [documents]
                # Trường hợp 2: File JSON chứa nhiều văn bản dạng Dictionary (ID làm key)
                else:
                    documents = [{"id": k, **v} for k, v in documents.items() if isinstance(v, dict)]
            else:
                continue
                
        for doc in documents:
            doc_id = doc.get("id")
            title = doc.get("name", "")
            passage = doc.get("passage", "")
            
            if not doc_id or not passage:
                continue
                
            # 1. Làm sạch & Lọc rác
            cleaned_text = filter_and_clean_text(passage)
            
            # 2. Cắt đoạn thông minh
            chunks = smart_legal_chunker(cleaned_text, max_words=max_words, overlap_words=overlap)
            
            # 3. Tạo thư mục và lưu Markdown
            doc_dir = os.path.join(output_dir, str(doc_id))
            os.makedirs(doc_dir, exist_ok=True)
            
            # Xóa các file .md cũ trong thư mục (nếu có) để tránh tồn đọng rác
            for f_name in os.listdir(doc_dir):
                if f_name.endswith('.md'):
                    os.remove(os.path.join(doc_dir, f_name))
            
            # Ghi các chunk ra file Markdown
            for i, chunk_text in enumerate(chunks):
                chunk_file_path = os.path.join(doc_dir, f"chunk_{i}.md")
                
                # Nhúng Title vào đầu mỗi Chunk
                markdown_content = f"# {title}\n\n{chunk_text}"
                
                with open(chunk_file_path, 'w', encoding='utf-8') as cf:
                    cf.write(markdown_content)
            
            total_count += 1
            
    print(f"✅ Hoàn tất tiền xử lý cho tổng cộng {total_count} văn bản. Đã lưu tại: {output_dir}")

import sys
import argparse
import config

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
        process_corpus(input_path, output_path, max_words=max_w, overlap=overlap_w)
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
