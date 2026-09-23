import os
import json
import glob
import sqlite3
from tqdm import tqdm
import config

# Thư mục lưu Index BM25 (dùng SQLite FTS5 để lưu trực tiếp lên ổ cứng)
BM25_DB_PATH = config.BM25_DB_PATH

def build_bm25_index():
    print("=========================================================")
    print("🚀 BẮT ĐẦU XÂY DỰNG INDEX BM25 (SIÊU NHẸ RAM) 🚀")
    print("=========================================================\n")

    # Tạo thư mục chứa nếu chưa có
    os.makedirs(os.path.dirname(BM25_DB_PATH), exist_ok=True)
    
    # Xóa DB cũ nếu có để làm lại từ đầu
    if os.path.exists(BM25_DB_PATH):
        os.remove(BM25_DB_PATH)

    print("[1] Đang khởi tạo kết nối Database (FTS5)...")
    conn = sqlite3.connect(BM25_DB_PATH)
    cursor = conn.cursor()
    
    # Tạo bảng Full Text Search 5 (FTS5) - Tự động hỗ trợ thuật toán BM25!
    cursor.execute('''
        CREATE VIRTUAL TABLE bm25_chunks USING fts5(
            chunk_id UNINDEXED,
            doc_id UNINDEXED,
            text
        )
    ''')
    
    print("\n[2] Đang đọc file tiền xử lý (Preprocessed Data) và đút vào Index...")
    all_jsonl_paths = glob.glob(os.path.join(config.DEFAULT_OUTPUT_DIR, "**", "*.jsonl"), recursive=True)
    # Lọc bỏ các thư mục lỡ có tên chứa đuôi .jsonl (ví dụ: corpus.jsonl)
    chunk_files = [f for f in all_jsonl_paths if os.path.isfile(f)]
    total_files = len(chunk_files)
    
    if total_files == 0:
        print(f"❌ Không tìm thấy file .jsonl nào trong {config.DEFAULT_OUTPUT_DIR}")
        print("Sếp cần chạy xong src/preprocess.py trước nha!")
        return

    total_inserted = 0
    
    for file_path in tqdm(chunk_files, desc="Quét files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                doc_id = record["doc_id"]
                chunk_id = record["chunk_id"]
                full_chunk_id = f"{doc_id}_{chunk_id}"
                text = record["text"]
                
                batch.append((full_chunk_id, doc_id, text))
                
                # Cứ 10,000 dòng thì lưu xuống ổ cứng 1 lần (Giải phóng RAM)
                if len(batch) >= 10000:
                    cursor.executemany('INSERT INTO bm25_chunks (chunk_id, doc_id, text) VALUES (?, ?, ?)', batch)
                    conn.commit()
                    total_inserted += len(batch)
                    batch = []
                    
            # Lưu nốt phần còn sót lại của file
            if batch:
                cursor.executemany('INSERT INTO bm25_chunks (chunk_id, doc_id, text) VALUES (?, ?, ?)', batch)
                conn.commit()
                total_inserted += len(batch)

    print(f"\n[3] Đang tối ưu hóa Index (Optimize)...")
    cursor.execute('INSERT INTO bm25_chunks(bm25_chunks) VALUES(\'optimize\')')
    conn.commit()
    conn.close()
    
    print(f"\n🎉 HOÀN TẤT! Đã Index thành công {total_inserted} đoạn văn vào ổ cứng!")
    print(f"📁 File Index BM25 nằm tại: {BM25_DB_PATH}")

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    build_bm25_index()
