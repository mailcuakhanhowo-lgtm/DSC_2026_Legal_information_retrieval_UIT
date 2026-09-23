import os
import json
from tqdm import tqdm

import config

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
        # Lấy danh sách chunk từ 2 nguồn (nếu không có thì trả về mảng rỗng)
        semantic_chunks = semantic_data.get(qid, {}).get("answer", [])
        bm25_chunks = bm25_data.get(qid, {}).get("answer", [])
        
        rrf_scores = {}
        
        # 1. Chấm điểm rrf cho Semantic
        for rank, chunk_id in enumerate(semantic_chunks):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (K_RRF + rank + 1)  # rank + 1 vì rank bắt đầu từ 0
            
        # 2. Chấm điểm rrf cho BM25
        for rank, chunk_id in enumerate(bm25_chunks):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (K_RRF + rank + 1)
            
        # 3. Sắp xếp lại danh sách chunk theo điểm tổng RRF giảm dần
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 4. Chỉ lấy TOP_K_FINAL (30) chunk xịn nhất
        top_k_chunks = [chunk for chunk, score in sorted_chunks[:TOP_K_FINAL]]
        
        hybrid_submission[qid] = {"answer": top_k_chunks}
        
    print(f"\n[3] Đang lưu kết quả Top {TOP_K_FINAL} siêu xịn ra file...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(hybrid_submission, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 HOÀN TẤT LAI GHÉP! Đã lưu kết quả tại {OUTPUT_FILE}")
    print("👉 BƯỚC TIẾP THEO: Sửa TOP30_FILE trong rerank_public.py trỏ về file này rồi chạy Rerank là xong!")

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    fuse_rrf()
