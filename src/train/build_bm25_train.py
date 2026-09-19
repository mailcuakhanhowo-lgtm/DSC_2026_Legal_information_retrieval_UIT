# ============================================================
# CELL 3: BUILD BM25OKAPI INDEX (TỪ SQLITE)
# ============================================================
import sqlite3, json, time, os, pickle
import multiprocessing as mp
from tqdm import tqdm
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi
import config_train as config

# --- Hàm worker chạy song song ---
def _tokenize_worker(row):
    cid, did, text = row
    tokens = ViTokenizer.tokenize(text).split()
    return cid, did, tokens

def main():
    print("=========================================================")
    print("📖 BUILD BM25OKAPI INDEX (THAY THẾ FTS5) 📖")
    print("=========================================================\n")
    
    db_path = config.SQLITE_DB_PATH
    if not os.path.exists(db_path):
        print(f"❌ Không tìm thấy {db_path}. Hãy chạy CELL 2 trước!")
        return

    # [1] Kết nối SQLite và đếm tổng chunk
    print(f"[1] Kết nối CSDL: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]
    print(f"    Tổng số chunks: {total_chunks:,}")

    cursor.execute("SELECT chunk_id, doc_id, markdown_content FROM chunks ORDER BY rowid")
    rows = cursor.fetchall()
    conn.close()

    # [2] Tokenize bằng pyvi (Multiprocessing)
    print(f"\n[2] Tokenize {total_chunks:,} chunks bằng pyvi ViTokenizer (3 cores)...")
    t0 = time.time()
    
    corpus_tokens = []
    chunk_ids     = []
    chunk_doc_map = {}

    with mp.Pool(processes=3) as pool:
        results = list(tqdm(pool.imap(_tokenize_worker, rows, chunksize=5000), 
                            total=total_chunks, desc="Tokenizing"))
                            
    for cid, did, tokens in results:
        chunk_ids.append(cid)
        chunk_doc_map[cid] = did
        corpus_tokens.append(tokens)

    t1 = time.time()
    print(f"    ✅ Tokenize xong trong {t1 - t0:.2f} giây.")

    # [3] Build BM25
    print("\n[3] Khởi tạo mô hình BM25Okapi...")
    t2 = time.time()
    bm25 = BM25Okapi(corpus_tokens)
    t3 = time.time()
    print(f"    ✅ Build BM25 xong trong {t3 - t2:.2f} giây.")

    # [4] Lưu xuống file
    out_pkl = config.BM25_PKL_PATH
    print(f"\n[4] Lưu index BM25 ra file: {out_pkl}...")
    t4 = time.time()
    data_to_save = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "chunk_doc_map": chunk_doc_map,
        "total_chunks": total_chunks
    }
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(data_to_save, f)
    t5 = time.time()
    print(f"    ✅ Lưu xong trong {t5 - t4:.2f} giây. Kích thước file: {os.path.getsize(out_pkl)/(1024*1024):.2f} MB")

    print("\n🎉 HOÀN THÀNH BUILD BM25!")

if __name__ == "__main__":
    main()
