import os
import json
import sqlite3
from tqdm import tqdm

import config

# Cấu hình đường dẫn lấy từ config
QUESTIONS_FILE = config.QUESTIONS_PUBLIC_FILE
BM25_DB_PATH = config.BM25_DB_PATH
OUTPUT_FILE = config.SUBMISSION_BM25_TOP_K
TOP_K = config.RETRIEVAL_TOP_K

def search_bm25():
    print("=========================================================")
    print("🔎 BẮT ĐẦU TÌM KIẾM KEYWORD BẰNG BM25 (SQLITE FTS5) 🔎")
    print("=========================================================\n")

    if not os.path.exists(BM25_DB_PATH):
        print(f"❌ Không tìm thấy file Index tại {BM25_DB_PATH}!")
        print("Sếp cần chạy file src/build_bm25.py trước nhé!")
        return

    print("[1] Đang tải danh sách câu hỏi Public...")
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions_raw = json.load(f)
        
    q_dict = {qid: item.get('question', '') if isinstance(item, dict) else item for qid, item in questions_raw.items()}
    
    print("[2] Đang kết nối với BM25 Index Database...")
    conn = sqlite3.connect(BM25_DB_PATH)
    cursor = conn.cursor()
    
    # Ép SQLite trả về điểm số BM25
    submission_dict = {}
    
    print("\n[3] Đang tiến hành tìm kiếm BM25 cho từng câu hỏi...")
    for qid, query in tqdm(q_dict.items(), desc="Tìm kiếm BM25"):
        # Làm sạch query để tránh lỗi cú pháp SQLite FTS5 (xóa dấu ngoặc, ký tự đặc biệt)
        safe_query = "".join(c if c.isalnum() else " " for c in query)
        
        # Tạo chuỗi truy vấn (Match bất kỳ từ nào, ưu tiên điểm BM25)
        # Sử dụng fts5 bm25() function để xếp hạng
        try:
            cursor.execute('''
                SELECT chunk_id
                FROM bm25_chunks 
                WHERE bm25_chunks MATCH ? 
                ORDER BY bm25(bm25_chunks) 
                LIMIT ?
            ''', (safe_query, TOP_K))
            
            results = cursor.fetchall()
            chunk_ids = [row[0] for row in results]
        except Exception as e:
            # Nếu câu hỏi quá dị, không match được gì
            chunk_ids = []
            
        submission_dict[qid] = {"answer": chunk_ids}
        
    conn.close()

    print(f"\n[4] Đang lưu kết quả Top {TOP_K} ra file...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(submission_dict, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 HOÀN TẤT! Đã lưu {len(submission_dict)} kết quả vào {OUTPUT_FILE}")

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    search_bm25()
