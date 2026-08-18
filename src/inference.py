import os
import sys
import re
import json
import time
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path

# Them src vao sys.path de import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

def segment_text(text: str) -> str:
    if HAS_PYVI:
        try:
            return ViTokenizer.tokenize(text)
        except Exception:
            return text
    return text

def tokenize_for_bm25(text: str) -> list:
    """Tach tu tieng Viet dong bo voi Phase 1 Indexing"""
    segmented = segment_text(text)
    clean_str = re.sub(r'[^\w\s_]', ' ', segmented.lower())
    return [w for w in clean_str.split() if len(w) > 1]

def evaluate_retrieval(submission_dict: dict, ground_truth_dict: dict):
    """Danh gia diem Recall@5 va Precision@5 dua tren tap ground truth"""
    recall_list = []
    precision_list = []
    rule_violations = 0

    for q_id, q_val in ground_truth_dict.items():
        q_id_str = str(q_id)
        if isinstance(q_val, dict):
            true_answers = q_val.get("answer")
        else:
            true_answers = q_val

        if not true_answers:
            continue

        true_set = set(str(a).strip() for a in true_answers)
        pred_answers = submission_dict.get(q_id_str, {}).get("answer", [])
        
        if len(pred_answers) > config.TOP_K_SUBMISSION_DOCS:
            rule_violations += 1

        pred_set = set(str(a).strip() for a in pred_answers[:config.TOP_K_SUBMISSION_DOCS])

        intersection = len(true_set & pred_set)
        rec = intersection / len(true_set) if len(true_set) > 0 else 0.0
        prec = intersection / len(pred_set) if len(pred_set) > 0 else 0.0

        recall_list.append(rec)
        precision_list.append(prec)

    mean_recall = float(np.mean(recall_list)) if recall_list else 0.0
    mean_precision = float(np.mean(precision_list)) if precision_list else 0.0

    return {
        "recall@5": mean_recall,
        "precision@5": mean_precision,
        "evaluated_questions": len(recall_list),
        "rule_violations": rule_violations
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Runtime Inference cho LegalIR Task 1 (BM25 + Max Pooling Aggregation).")
    parser.add_argument("--input", type=str, default=str(config.PUBLIC_TEST_PATH), help="Duong dan file cau hoi (Mac dinh: public-official.json)")
    parser.add_argument("--output", type=str, default=str(config.SUBMISSION_PATH), help="Duong dan file dau ra submission.json")
    parser.add_argument("--bm25", type=str, default=str(config.BM25_INDEX_PATH), help="Duong dan file bm25_index.pkl")
    parser.add_argument("--evaluate", action="store_true", help="Tu dong danh gia diem Recall@5 / Precision@5 neu file dau vao co dap an")
    
    args = parser.parse_args()
    
    start_time = time.time()
    print("BAT DAU GIAI DOAN 2: RUNTIME INFERENCE")
    print(f"File cau hoi dau vao: {args.input}")
    print(f"BM25 Index Model:   {args.bm25}")
    print(f"File nop bai dau ra: {args.output}")

    # 1. Load BM25 Index
    if not os.path.exists(args.bm25):
        print(f"Khong tim thay file BM25 Index tai: {args.bm25}")
        print("Vui long chay lenh truoc: python src/preprocess_and_index.py")
        sys.exit(1)

    print("Dang tai BM25 Index vao bo nho...")
    with open(args.bm25, 'rb') as f:
        bm25_payload = pickle.load(f)

    bm25 = bm25_payload["bm25"]
    chunk_ids = np.array(bm25_payload["chunk_ids"])
    chunk_doc_map = bm25_payload["chunk_doc_map"]
    
    print(f"Da tai xong BM25 Index chua {len(chunk_ids)} chunks!")

    # Xay dung Inverted Index Vectorized de tang toc do truy van
    print("Dang khoi tao Vectorized Inverted Index...")
    t_idx_start = time.time()
    inverted_index = {}
    doc_len = np.array(bm25.doc_len, dtype=np.float32)
    k1 = float(bm25.k1)
    b = float(bm25.b)
    avgdl = float(bm25.avgdl)
    doc_len_penalty = k1 * (1.0 - b + b * (doc_len / avgdl))

    for doc_idx, freqs in enumerate(bm25.doc_freqs):
        for word, tf in freqs.items():
            if word not in inverted_index:
                inverted_index[word] = ([], [])
            inverted_index[word][0].append(doc_idx)
            inverted_index[word][1].append(tf)

    for word in inverted_index:
        d_ids, tfs = inverted_index[word]
        inverted_index[word] = (np.array(d_ids, dtype=np.int32), np.array(tfs, dtype=np.float32))
    
    print(f"Khoi tao Inverted Index hoan tat trong {time.time() - t_idx_start:.2f}s!")

    # 2. Load file cau hoi
    if not os.path.exists(args.input):
        print(f"Khong tim thay file cau hoi: {args.input}")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)

    total_queries_count = len(queries_data)
    print(f"Tong so cau hoi can suy luan: {total_queries_count}")

    submission_dict = {}
    top_k_chunks = config.TOP_K_CANDIDATE_CHUNKS  # 50 chunks
    top_k_docs = config.TOP_K_SUBMISSION_DOCS       # 5 doc_ids

    # 3. Chay suy luan cho tung cau hoi
    corpus_size = bm25.corpus_size
    for q_idx, (q_id, q_val) in enumerate(queries_data.items()):
        q_id_str = str(q_id)
        
        if isinstance(q_val, dict):
            query_text = q_val.get("question", "")
        else:
            query_text = str(q_val)

        if not query_text.strip():
            submission_dict[q_id_str] = {"answer": []}
            continue

        # A. Tach tu tieng Viet cho cau hoi
        query_tokens = tokenize_for_bm25(query_text)

        if not query_tokens:
            submission_dict[q_id_str] = {"answer": []}
            continue

        # B. Tinh diem BM25 Vectorized tren Inverted Index
        scores = np.zeros(corpus_size, dtype=np.float32)
        has_match = False
        for q in query_tokens:
            if q in inverted_index and q in bm25.idf:
                q_doc_ids, q_tfs = inverted_index[q]
                idf = bm25.idf[q]
                scores[q_doc_ids] += idf * (q_tfs * (k1 + 1.0)) / (q_tfs + doc_len_penalty[q_doc_ids])
                has_match = True

        if not has_match:
            submission_dict[q_id_str] = {"answer": []}
            continue

        # C. Lay Top 50 chunk_id co diem BM25 cao nhat
        if len(scores) > top_k_chunks:
            top_chunk_indices = np.argpartition(scores, -top_k_chunks)[-top_k_chunks:]
            top_chunk_indices = top_chunk_indices[np.argsort(scores[top_chunk_indices])[::-1]]
        else:
            top_chunk_indices = np.argsort(scores)[::-1]

        # D. DOCUMENT AGGREGATION (MAX POOLING): Gom nhom chunk ve doc_id goc
        doc_max_scores = {}
        for idx in top_chunk_indices:
            score = scores[idx]
            if score <= 0:
                continue
            c_id = chunk_ids[idx]
            doc_id = chunk_doc_map[c_id]
            
            if doc_id not in doc_max_scores or score > doc_max_scores[doc_id]:
                doc_max_scores[doc_id] = score

        # E. Sap xep doc_id theo diem Max Pooling giam dan va cat Top 5
        sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)
        predicted_doc_ids = [str(doc_id) for doc_id, score in sorted_docs[:top_k_docs]]

        # F. Format ket qua tuan thu nghiem ngat chuan BTC
        submission_dict[q_id_str] = {
            "answer": predicted_doc_ids
        }

        # Log tien do moi 200 cau hoi
        if (q_idx + 1) % 200 == 0 or (q_idx + 1) == total_queries_count:
            print(f"Tien do: Da xu ly {q_idx + 1}/{total_queries_count} cau hoi ({(q_idx + 1)/total_queries_count*100:.1f}%)...")

    # 4. Ghi ket qua ra submission.json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(submission_dict, f, ensure_ascii=False, indent=2)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("SUY LUAN BAI THI HOAN TAT THANH CONG!")
    print(f"Tong so cau hoi da xu ly: {len(submission_dict)}")
    print(f"File nop bai luu tai:     {output_path.resolve()}")
    print(f"Tong thoi gian suy luan:  {elapsed_time:.2f} giay ({elapsed_time/len(queries_data):.4f}s/cau)")
    print("=" * 60)

    # 5. Danh gia tu dong neu truyen co --evaluate hoac neu file co dap an
    has_labels = False
    if queries_data:
        first_val = next(iter(queries_data.values()))
        if isinstance(first_val, dict) and first_val.get("answer") is not None:
            has_labels = True
        elif isinstance(first_val, list):
            has_labels = True

    if args.evaluate or has_labels:
        try:
            eval_metrics = evaluate_retrieval(submission_dict, queries_data)
            print("\nKET QUA TU DANH GIA CUC BO (LOCAL EVALUATION):")
            print(f"  Mean Recall@5:    {eval_metrics['recall@5']:.4f} ({eval_metrics['recall@5']*100:.2f}%)")
            print(f"  Mean Precision@5: {eval_metrics['precision@5']:.4f} ({eval_metrics['precision@5']*100:.2f}%)")
            print(f"  So cau vi pham luat (>5 doc_ids): {eval_metrics['rule_violations']}")
        except Exception as e:
            print(f"Khong the chay tu danh gia: {e}")

if __name__ == "__main__":
    main()
