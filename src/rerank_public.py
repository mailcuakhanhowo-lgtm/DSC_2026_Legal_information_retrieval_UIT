import os
# Ép HuggingFace tải model về ổ D (vì ổ C của Sếp chỉ còn 2.4GB trống)
os.environ["HF_HOME"] = r"D:\Project Vibe Coding\DSC_2026\local_llm_data\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\Project Vibe Coding\DSC_2026\local_llm_data\.hf_cache"

import json
import torch
from tqdm import tqdm
from pymilvus import MilvusClient
# Import cả 2 class để dùng tùy theo model
from FlagEmbedding import FlagReranker, FlagLLMReranker

import sys
# Chỉnh lại chuẩn UTF-8 để không bị lỗi tiếng Việt trên Windows (Cách mới không bị nuốt log)
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RERANKER_MODEL, RERANKER_BATCH_SIZE, LLM_CACHE_DIR
import config

# ================= CẤU HÌNH =================
# Lấy trực tiếp danh sách Top K đã qua lai ghép RRF
TOP_K_FILE = config.SUBMISSION_HYBRID_TOP_K
QUESTIONS_FILE = config.QUESTIONS_PUBLIC_FILE
CONTEXT_DB = config.MILVUS_DB_PATH
FINAL_SUBMISSION = config.FINAL_SUBMISSION

TOP_K = 5
BATCH_SIZE = RERANKER_BATCH_SIZE

def load_data():
    print("[1] Đang tải danh sách câu hỏi Public...")
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions_raw = json.load(f)
    # Lấy text câu hỏi (Đảm bảo bóc đúng chuỗi string từ biến 'question')
    q_dict = {qid: item.get('question', '') if isinstance(item, dict) else item for qid, item in questions_raw.items()}
    
    print(f"[2] Đang tải kết quả Top {config.RRF_TOP_K} thô (Retrieval)...")
    with open(TOP_K_FILE, 'r', encoding='utf-8') as f:
        top_k_data = json.load(f)
        
    # Tập hợp toàn bộ chunk_id cần thiết để truy vấn text từ Milvus 1 lần cho nhanh
    needed_chunk_ids = set()
    for qid, data in top_k_data.items():
        needed_chunk_ids.update(data['answer'])
    needed_chunk_ids = list(needed_chunk_ids)
    
    # ================= MỚI: ĐỌC TRỰC TIẾP TỪ Ổ ĐĨA ĐỂ TRÁNH TRÀN RAM =================
    print(f"[3] Đang quét các file JSON để trích xuất văn bản gốc cho {len(needed_chunk_ids)} chunks (0% RAM ngốn)...")
    import glob
    
    doc_text_dict = {}
    metadata_files = glob.glob(r"D:\Project Vibe Coding\DSC_2026\embeded_data\**\metadata_part_*.json", recursive=True)
    
    needed_set = set(needed_chunk_ids)
    
    for meta_file in tqdm(metadata_files, desc="Đọc JSON từ ổ cứng"):
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
            
        for item in metadata_list:
            if item["chunk_id"] in needed_set:
                doc_text_dict[item["chunk_id"]] = item["text"]
                needed_set.remove(item["chunk_id"])
                
            # Đỡ tốn thời gian quét tiếp nếu đã gom đủ hàng
            if not needed_set:
                break
                
        if not needed_set:
            break
            
    print(f"    -> Đã trích xuất thành công {len(doc_text_dict)} đoạn văn bản!")
            
    return q_dict, top_k_data, doc_text_dict

def main():
    q_dict, top_k_data, doc_text_dict = load_data()
    
    print(f"\n[4] Đang nạp {RERANKER_MODEL} lên VRAM (Sẽ tốn chút thời gian)...")
    # Tự động chọn Class đúng với cấu trúc Model
    if "gemma" in RERANKER_MODEL.lower():
        reranker = FlagLLMReranker(RERANKER_MODEL, use_fp16=True)
    else:
        # BAAI/bge-reranker-v2-m3 là Encoder (XLM-RoBERTa), KHÔNG PHẢI LLM!
        reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
    
    print("\n[5] Bắt đầu quá trình chấm điểm chéo (Reranking)...")
    final_submission = {}
    
    # Chuẩn bị context để tắt cái progress bar gây nhiễu của thư viện
    import sys, os
    
    # Duyệt qua từng câu hỏi
    for qid, data in tqdm(top_k_data.items(), desc="Reranking"):
        question_text = q_dict.get(qid, "")
        candidate_chunks = data['answer']
        
        # Cấu trúc đầu vào cho Reranker: [ [câu_hỏi, đoạn_văn_1], [câu_hỏi, đoạn_văn_2], ... ]
        pairs = []
        valid_chunks = []
        for chunk_id in candidate_chunks:
            if not chunk_id:
                continue
            text = doc_text_dict.get(chunk_id, "")
            pairs.append([question_text, text])
            valid_chunks.append(chunk_id)
            
        if not pairs:
            final_submission[qid] = {"answer": []}
            continue
            
        # Tắt progress bar gây nhiễu của thư viện bằng cách chuyển hướng stderr tạm thời
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                scores = reranker.compute_score(pairs, batch_size=BATCH_SIZE)
            finally:
                sys.stderr = old_stderr
        
        if isinstance(scores, float):
            scores = [scores]
            
        # Ghép cặp ChunkID và Điểm số, sau đó sắp xếp giảm dần
        chunk_scores = list(zip(valid_chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Sửa lại: Nộp đáp án phải là Doc ID gốc (loại bỏ đuôi _chunk_x) và lọc trùng lặp
        top5_doc_ids = []
        for chunk, score in chunk_scores:
            doc_id = chunk.split("_chunk_")[0]
            if doc_id not in top5_doc_ids:
                top5_doc_ids.append(doc_id)
            if len(top5_doc_ids) == TOP_K:
                break
                
        final_submission[qid] = {
            "answer": top5_doc_ids
        }
        
    print(f"\n[6] Xuất file nộp bài Vô Địch: {FINAL_SUBMISSION}")
    with open(FINAL_SUBMISSION, 'w', encoding='utf-8') as f:
        json.dump(final_submission, f, ensure_ascii=False, indent=4)
        
    print("🎉 HOÀN TẤT RERANKING! Sếp có thể tự tin đem file này đi nộp lấy cúp!")

if __name__ == "__main__":
    main()
