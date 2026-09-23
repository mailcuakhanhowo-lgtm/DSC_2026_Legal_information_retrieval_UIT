import os
import json
import numpy as np
import torch
from pymilvus import MilvusClient
from tqdm import tqdm

import codecs
import sys
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

EMBEDED_DATA_DIR = r"D:\Project Vibe Coding\DSC_2026\embeded_data"
QUERIES_DB_PATH = r"D:\Project Vibe Coding\DSC_2026\embed_queries\milvus_queries_public.db"
import config
SUBMISSION_OUTPUT = config.SUBMISSION_SEMANTIC_TOP_K
TOP_K = config.RETRIEVAL_TOP_K

def generate_submission_low_ram():
    print(f"\n[1] Đang nạp 1000 Vectors câu hỏi Public từ Milvus...")
    queries_client = MilvusClient(uri=QUERIES_DB_PATH)
    queries_client.load_collection("queries")
    query_data = queries_client.query(
        collection_name="queries",
        filter="query_id != ''",
        output_fields=["query_id", "embedding"]
    )
    
    query_ids = [item["query_id"] for item in query_data]
    query_vectors = np.array([item["embedding"] for item in query_data], dtype=np.float32)
    
    # Chuyển query_vectors sang PyTorch GPU (nếu có) để tính toán ma trận cho lẹ
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_tensor = torch.tensor(query_vectors).to(device)
    # Chuẩn hóa (Normalize) để tính Cosine Similarity bằng Dot Product
    q_tensor = torch.nn.functional.normalize(q_tensor, p=2, dim=1)
    
    print(f"    Đã nạp {len(query_ids)} câu hỏi lên {device.upper()}.")

    print(f"\n[2] Bắt đầu rà soát từng phần (Chunked Search) để tiết kiệm RAM...")
    import glob
    npy_files = sorted(glob.glob(os.path.join(EMBEDED_DATA_DIR, "**", "*.npy"), recursive=True))
    
    # Khởi tạo mảng lưu trữ Top-K điểm số và Top-K DocIDs cho từng câu hỏi
    # Ghi nhận -1.0 cho điểm số ban đầu
    best_scores = torch.full((len(query_ids), TOP_K), -1.0, dtype=torch.float32).to(device)
    best_doc_ids = [[""] * TOP_K for _ in range(len(query_ids))]
    
    for npy_path in npy_files:
        base_name = os.path.basename(npy_path)
        part_name = base_name.replace("embeddings_", "").replace(".npy", "")
        json_path = os.path.join(os.path.dirname(npy_path), f"metadata_{part_name}.json")
        
        if not os.path.exists(json_path):
            continue
            
        print(f"    => Đang xử lý đối chiếu {part_name}...")
        # Load 100k vectors
        db_vectors = np.load(npy_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # Chuyển DB lên Tensor
        db_tensor = torch.tensor(db_vectors).to(device)
        db_tensor = torch.nn.functional.normalize(db_tensor, p=2, dim=1)
        
        # Tính toán ma trận độ tương đồng Cosine Similarity (Q x DB)
        # Kích thước: (1000, 100000)
        similarity_scores = torch.matmul(q_tensor, db_tensor.T)
        
        # Lấy Top K cao nhất trong cái chunk 100k này
        top_scores, top_indices = torch.topk(similarity_scores, k=TOP_K, dim=1)
        
        # Cập nhật vào danh sách Kỷ lục tổng
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        best_scores_cpu = best_scores.cpu().numpy()
        
        for q_idx in range(len(query_ids)):
            # Kết hợp Kỷ lục cũ và Kỷ lục mới của Chunk này
            combined_scores = np.concatenate((best_scores_cpu[q_idx], top_scores[q_idx]))
            # Sửa lại: Lấy chunk_id thay vì doc_id để Reranker đọc chính xác đoạn ngắn!
            combined_chunks = best_doc_ids[q_idx] + [metadata[idx]["chunk_id"] for idx in top_indices[q_idx]]
            
            # Sắp xếp lại để lấy Top K của cả 2
            sorted_idxs = np.argsort(combined_scores)[::-1]
            
            # Cập nhật lại kỷ lục (lọc trùng lặp chunk_id)
            new_best_chunks = []
            new_best_scores = []
            for i in sorted_idxs:
                chunk = combined_chunks[i]
                if chunk not in new_best_chunks:
                    new_best_chunks.append(chunk)
                    new_best_scores.append(combined_scores[i])
                if len(new_best_chunks) == TOP_K:
                    break
                    
            while len(new_best_chunks) < TOP_K:
                new_best_chunks.append("")
                new_best_scores.append(-1.0)
                
            best_scores_cpu[q_idx] = new_best_scores
            best_doc_ids[q_idx] = new_best_chunks
            
        best_scores = torch.tensor(best_scores_cpu).to(device)
        
        # Giải phóng RAM lập tức
        del db_vectors
        del db_tensor
        del similarity_scores
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n[3] Đang xuất file nộp bài...")
    submission_dict = {}
    for i, qid in enumerate(query_ids):
        submission_dict[qid] = {
            "answer": best_doc_ids[i]
        }
        
    os.makedirs(os.path.dirname(SUBMISSION_OUTPUT), exist_ok=True)
    with open(SUBMISSION_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(submission_dict, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 XONG! File {SUBMISSION_OUTPUT} đã hoàn tất bất chấp RAM yếu!")

if __name__ == "__main__":
    generate_submission_low_ram()
