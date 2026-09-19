import json
import os
import sys
import sqlite3
from tqdm import tqdm
import multiprocessing as mp
import config_train as config

TOP_K_FILE     = config.SUBMISSION_HYBRID_TOP_K
RERANKER_MODEL = config.RERANKER_MODEL
BATCH_SIZE     = config.RERANKER_BATCH_SIZE

def load_data():
    print("[1] Đang tải danh sách câu hỏi...")
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        questions_raw = json.load(f)

    q_dict = {k: (v.get("question", "") if isinstance(v, dict) else v) for k, v in questions_raw.items()}

    print("[2] Đang tải kết quả Top 500 thô (Retrieval)...")
    with open(TOP_K_FILE, 'r', encoding='utf-8') as f:
        top_k_data = json.load(f)

    if hasattr(config, 'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        top_k_data = {k: top_k_data[k] for k in list(top_k_data.keys())[:config.JUST_CHECK_QUERIES]}

    needed_chunks = set()
    for qid, data in top_k_data.items():
        needed_chunks.update(data['answer'])

    print(f"[3] Cần đọc nội dung của {len(needed_chunks):,} chunks từ SQLite...")
    doc_text_dict = {}

    db_path = config.SQLITE_DB_PATH
    if not os.path.exists(db_path):
        print(f"❌ Không tìm thấy SQLite DB: {db_path}. Hãy chạy CELL 2 trước!")
        return q_dict, top_k_data, {}

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    needed_list = list(needed_chunks)
    BATCH = 500
    for i in range(0, len(needed_list), BATCH):
        batch = needed_list[i:i + BATCH]
        placeholders = ",".join("?" * len(batch))
        cursor.execute(
            f"SELECT chunk_id, markdown_content FROM chunks WHERE chunk_id IN ({placeholders})",
            batch
        )
        for chunk_id, md_content in cursor.fetchall():
            doc_text_dict[chunk_id] = md_content

    conn.close()
    print(f"    Đã đọc {len(doc_text_dict):,}/{len(needed_chunks):,} chunks từ SQLite.")
    return q_dict, top_k_data, doc_text_dict

def worker_rerank(gpu_id, query_ids, q_dict, top_k_data, doc_text_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from FlagEmbedding import FlagReranker
    print(f"\n[GPU {gpu_id}] Đang nạp mô hình lên VRAM (fp32)...")
    reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
    
    submission_part = {}
    
    for qid in tqdm(query_ids, desc=f"GPU {gpu_id} Reranking", position=gpu_id, leave=True):
        data = top_k_data[qid]
        question_text = q_dict.get(qid, "")

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
            doc_id = chunk.rsplit('_', 1)[0]   # format: {doc_id}_{i}
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

    q_dict, top_k_data, doc_text_dict = load_data()
    
    all_qids = list(top_k_data.keys())
    mid = len(all_qids) // 2
    qids_0 = all_qids[:mid]
    qids_1 = all_qids[mid:]
    
    print(f"\n[4] Khởi tạo Pool: GPU 0 làm {len(qids_0)} câu | GPU 1 làm {len(qids_1)} câu")
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(2) as pool:
        args = [
            (0, qids_0, q_dict, top_k_data, doc_text_dict),
            (1, qids_1, q_dict, top_k_data, doc_text_dict)
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
