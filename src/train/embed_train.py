# ============================================================
# CELL 4: EMBED VECTORS — Đọc từ SQLite (legal_corpus.db)
# Xuất ra .npy + .json parts để upload lên Kaggle Dataset
# ============================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc, json, sqlite3, time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import config_train as config

def embed_corpus_local_numpy():
    print("=" * 70)
    print("CELL 4: EMBED VECTORS (đọc từ SQLite → xuất .npy)")
    print("=" * 70)

    db_path = config.SQLITE_DB_PATH
    out_dir = config.EMBEDED_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(db_path):
        print(f"❌ Không tìm thấy SQLite DB: {db_path}")
        return

    print(f"\n[1] Đọc chunks từ SQLite: {db_path} ...")
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]
    print(f"    Tổng số chunks: {total_chunks:,}")

    cursor.execute("SELECT chunk_id, title, markdown_content FROM chunks ORDER BY rowid")
    rows = cursor.fetchall()
    conn.close()

    all_texts    = [row[2] for row in rows]   # markdown_content
    all_metadata = [
        {"chunk_id": row[0], "title": row[1], "link": ""}
        for row in rows
    ]
    del rows
    gc.collect()
    print(f"    Đã nạp {total_chunks:,} chunks vào RAM!")

    print("\n[2] Khởi tạo Embedding Model (Multi-GPU Pool)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    pool  = model.start_multi_process_pool()

    print("\n[3] Vector hóa & lưu từng phần ...")
    chunk_size = 100000
    num_parts  = (total_chunks + chunk_size - 1) // chunk_size

    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx   = min((i + 1) * chunk_size, total_chunks)
        print(f"\n🚀 Phần {i+1}/{num_parts}: [{start_idx:,} → {end_idx:,}]")

        embeddings = model.encode(
            all_texts[start_idx:end_idx],
            pool=pool,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        emb_path  = os.path.join(out_dir, f"embeddings_part_{i+1}.npy")
        meta_path = os.path.join(out_dir, f"metadata_part_{i+1}.json")
        np.save(emb_path, np.array(embeddings).astype(np.float32))
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(all_metadata[start_idx:end_idx], f, ensure_ascii=False)
        print(f"    ✅ Đã lưu: {emb_path}")

    model.stop_multi_process_pool(pool)
    print(f"\n🎉 XONG! Vector đã lưu tại: {out_dir}")

if __name__ == "__main__":
    embed_corpus_local_numpy()
