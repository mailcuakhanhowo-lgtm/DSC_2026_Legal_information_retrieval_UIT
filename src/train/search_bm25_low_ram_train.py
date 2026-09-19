# ============================================================
# CELL 6: BM25 SEARCH — Load pkl, tokenize query bằng pyvi
# → BM25Okapi.get_scores() → Top-K chunk_ids
# ============================================================
import json, re, os, gc, pickle
from tqdm import tqdm
from pyvi import ViTokenizer
import config_train as config

def tokenize_query(text: str) -> list:
    """Tokenize query bằng pyvi — khớp với cách build BM25 ở CELL 3."""
    try:
        seg = ViTokenizer.tokenize(text)
    except:
        seg = text
    clean_str = re.sub(r'[^\w\s_]', ' ', seg.lower())
    return [w for w in clean_str.split() if len(w) > 1]

def main():
    print("=" * 70)
    print("CELL 6: BM25 SEARCH (BM25Okapi + pyvi)")
    print("=" * 70)

    bm25_path = config.BM25_PKL_PATH
    if not os.path.exists(bm25_path):
        print(f"❌ Không tìm thấy BM25 pkl: {bm25_path}")
        return

    # [1] Load BM25 index từ pkl
    print(f"\n[1] Load BM25 index từ: {bm25_path} ...")
    with open(bm25_path, "rb") as f:
        bm25_data  = pickle.load(f)
    bm25_model = bm25_data["bm25"]
    chunk_ids  = bm25_data["chunk_ids"]
    print(f"    Tổng chunks trong index: {bm25_data['total_chunks']:,}")

    # [2] Load câu hỏi
    print(f"\n[2] Đọc câu hỏi từ: {config.QUESTIONS_PUBLIC_FILE} ...")
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        q_data = json.load(f)

    query_ids = list(q_data.keys())
    if hasattr(config, 'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        query_ids = query_ids[:config.JUST_CHECK_QUERIES]
    print(f"    Số câu hỏi: {len(query_ids):,}")

    # [3] BM25 Search từng câu hỏi
    print(f"\n[3] BM25 Search Top-{config.RETRIEVAL_TOP_K} ...")
    import numpy as np
    submission = {}
    top_k = config.RETRIEVAL_TOP_K

    for qid in tqdm(query_ids, desc="BM25 Search"):
        q_raw    = q_data[qid]
        original = q_raw.get('question', '') if isinstance(q_raw, dict) else q_raw

        tokens = tokenize_query(original)
        if not tokens:
            submission[qid] = {"answer": []}
            continue

        scores     = bm25_model.get_scores(tokens)
        top_k_idx  = np.argpartition(scores, -top_k)[-top_k:]
        top_k_idx  = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        submission[qid] = {"answer": [chunk_ids[i] for i in top_k_idx]}

    del bm25_model, chunk_ids, bm25_data
    gc.collect()

    os.makedirs(os.path.dirname(config.SUBMISSION_BM25_TOP_K), exist_ok=True)
    with open(config.SUBMISSION_BM25_TOP_K, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
    print(f"\n🎉 Hoàn tất BM25 Search! Lưu tại: {config.SUBMISSION_BM25_TOP_K}")

if __name__ == "__main__":
    main()
