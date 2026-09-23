import json, os
from tqdm import tqdm
import config_train as config

K_RRF = 60
TOP_K_FINAL = config.RRF_TOP_K

def fuse_rrf():
    print("🧬 RRF FUSION SEMANTIC + BM25\n")
    if not os.path.exists(config.SUBMISSION_SEMANTIC_TOP_K) or not os.path.exists(config.SUBMISSION_BM25_TOP_K):
        print("❌ Thiếu file đầu vào!"); return

    with open(config.SUBMISSION_SEMANTIC_TOP_K,'r',encoding='utf-8') as f: semantic_data = json.load(f)
    with open(config.SUBMISSION_BM25_TOP_K,'r',encoding='utf-8') as f: bm25_data = json.load(f)

    hybrid_submission = {}
    for qid in tqdm(semantic_data.keys(), desc="Fusion RRF"):
        rrf_scores = {}
        for rank, chunk_id in enumerate(semantic_data.get(qid,{}).get("answer",[])):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id,0.0) + 1.0/(K_RRF+rank+1)
        for rank, chunk_id in enumerate(bm25_data.get(qid,{}).get("answer",[])):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id,0.0) + 1.0/(K_RRF+rank+1)
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_chunks = []
        doc_chunk_count = {}
        for chunk, score in sorted_chunks:
            doc_id = chunk.rsplit('_', 1)[0]   # format: {doc_id}_{i}
            if doc_chunk_count.get(doc_id,0) < getattr(config,'MAX_CHUNKS_PER_DOC',50):
                top_k_chunks.append(chunk)
                doc_chunk_count[doc_id] = doc_chunk_count.get(doc_id,0) + 1
            if len(top_k_chunks) == TOP_K_FINAL: break
        hybrid_submission[qid] = {"answer": top_k_chunks}

    os.makedirs(os.path.dirname(config.SUBMISSION_HYBRID_TOP_K), exist_ok=True)
    with open(config.SUBMISSION_HYBRID_TOP_K,'w',encoding='utf-8') as f:
        json.dump(hybrid_submission, f, ensure_ascii=False, indent=4)
    print(f"🎉 HOÀN TẤT! Lưu tại: {config.SUBMISSION_HYBRID_TOP_K}")

if __name__ == "__main__":
    fuse_rrf()
