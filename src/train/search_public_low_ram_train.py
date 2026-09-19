# Tìm kiếm vector trực tiếp từ .npy + .json đã upload
import json, os, glob, gc
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import config_train as config

def main():
    print("=========================================================")
    print("🔍 SEMANTIC SEARCH TRỰC TIẾP TỪ .NPY (KHÔNG CẦN MILVUS) 🔍")
    print("=========================================================\n")

    embed_dir = config.EMBEDED_DATA_INPUT

    npy_files  = sorted(glob.glob(os.path.join(embed_dir, "embeddings_part_*.npy")))
    json_files = sorted(
        glob.glob(os.path.join(embed_dir, "metadata_part_*.json")) +
        glob.glob(os.path.join(embed_dir, "metadata_part_*.txt"))
    )

    if not npy_files or len(npy_files) != len(json_files):
        print(f"❌ Lỗi: {len(npy_files)} .npy vs {len(json_files)} metadata — không khớp!")
        return

    print(f"📦 Tìm thấy {len(npy_files)} phần vector. Đang đọc chunk_id...")

    all_chunk_ids = []
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        all_chunk_ids.extend([m["chunk_id"] for m in meta])
    total_chunks = len(all_chunk_ids)
    print(f"✅ Tổng số chunk: {total_chunks:,}")

    print(f"\n[2] Đang đọc câu hỏi từ {config.QUESTIONS_PUBLIC_FILE}...")
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        q_data = json.load(f)

    query_ids = list(q_data.keys())
    if hasattr(config, 'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        query_ids = query_ids[:config.JUST_CHECK_QUERIES]
    print(f"Số câu hỏi cần tìm: {len(query_ids)}")

    print("\n[3] Đang encode tất cả câu hỏi bằng BGE-M3...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

    query_texts = []
    for qid in query_ids:
        q_raw = q_data[qid]
        original = q_raw.get('question', '') if isinstance(q_raw, dict) else q_raw
        query_texts.append(original)

    query_matrix = model.encode(query_texts, normalize_embeddings=True,
                                batch_size=32, show_progress_bar=True)
    query_matrix = query_matrix.astype(np.float32)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ Encode xong! Shape: {query_matrix.shape}")

    print(f"\n[4] Đang tính điểm similarity cho {total_chunks:,} chunks...")
    n_queries = len(query_ids)
    top_k = config.RETRIEVAL_TOP_K

    accumulated_scores = [[] for _ in range(n_queries)]

    for npy_f in tqdm(npy_files, desc="Quét .npy parts"):
        embs = np.load(npy_f).astype(np.float32)
        part_scores = query_matrix @ embs.T
        for qi in range(n_queries):
            accumulated_scores[qi].append(part_scores[qi].copy())
        del embs, part_scores
        gc.collect()

    print("\n[5] Tổng hợp kết quả Top-K...")
    submission = {}
    for qi, qid in enumerate(tqdm(query_ids, desc="Tổng hợp Top-K")):
        full_scores = np.concatenate(accumulated_scores[qi])
        top_k_indices = np.argpartition(full_scores, -top_k)[-top_k:]
        top_k_indices = top_k_indices[np.argsort(full_scores[top_k_indices])[::-1]]
        submission[qid] = {"answer": [all_chunk_ids[i] for i in top_k_indices]}

    os.makedirs(os.path.dirname(config.SUBMISSION_SEMANTIC_TOP_K), exist_ok=True)
    with open(config.SUBMISSION_SEMANTIC_TOP_K, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
    print(f"\n🎉 HOÀN TẤT SEMANTIC SEARCH! Lưu tại: {config.SUBMISSION_SEMANTIC_TOP_K}")

if __name__ == "__main__":
    main()
