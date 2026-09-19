import os
import json
import glob
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN KAGGLE
# ==========================================
INPUT_DIR = "/kaggle/working/processed_data"
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
MAX_POSITION_EMBEDDINGS = 2048
BATCH_SIZE = 64 # Giới hạn ở 64 mỗi GPU T4 để đảm bảo an toàn tuyệt đối cho VRAM

def embed_corpus_kaggle_numpy():
    print("--- BƯỚC 1: ĐỌC DỮ LIỆU TỪ JSONL ---")
    chunk_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.jsonl"), recursive=True)
    total_files = len(chunk_files)
    if total_files == 0:
        print(f"Không tìm thấy file .jsonl nào trong {INPUT_DIR}.")
        print("Sếp nhớ chạy đoạn code chia Chunk trước nhé!")
        return

    all_texts = []
    all_metadata = []

    print("Đang nạp toàn bộ nội dung văn bản vào RAM (Kaggle dư sức)...")
    for file_path in tqdm(chunk_files, desc="Đọc files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                doc_id = record["doc_id"]
                chunk_id = record["chunk_id"]
                
                all_texts.append(record["text"])
                all_metadata.append({
                    "chunk_id": f"{doc_id}_{chunk_id}",
                    "doc_id": doc_id,
                    "title": record.get("title", ""),
                    "text": record["text"], # Lưu lại text luôn để sau này Milvus có cái mà xài
                    "link": ""
                })
                
    total_chunks = len(all_texts)
    print(f"Đã nạp xong {total_chunks} đoạn văn bản!")

    print("\n--- BƯỚC 2 & 3: VECTOR HÓA & LƯU TỪNG PHẦN (CHỐNG MẤT DỮ LIỆU) ---")
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    model.max_seq_length = MAX_POSITION_EMBEDDINGS
    pool = model.start_multi_process_pool()
    
    CHUNK_SAVE_SIZE = 100000
    
    # Duyệt qua từng đoạn 100.000 văn bản
    for i in range(0, total_chunks, CHUNK_SAVE_SIZE):
        end_idx = min(i + CHUNK_SAVE_SIZE, total_chunks)
        part_num = (i // CHUNK_SAVE_SIZE) + 1
        
        print(f"\n🚀 Đang xử lý Phần {part_num}: Từ {i} đến {end_idx}...")
        
        # Cắt lấy 100k text và metadata
        texts_subset = all_texts[i:end_idx]
        meta_subset = all_metadata[i:end_idx]
        
        # Nặn Vector (Sử dụng encode bản mới nhất có tích hợp đa luồng và normalize)
        embeddings = model.encode(
            texts_subset, 
            pool=pool, 
            batch_size=BATCH_SIZE, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Lưu ngay lập tức xuống đĩa cứng
        np_path = f'/kaggle/working/embeddings_part_{part_num}.npy'
        meta_path = f'/kaggle/working/metadata_part_{part_num}.json'
        
        print(f"💾 Đang lưu {len(texts_subset)} kết quả xuống {np_path}...")
        np.save(np_path, embeddings)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_subset, f, ensure_ascii=False)
            
        # Dọn rác giải phóng RAM
        del embeddings
        del texts_subset
        del meta_subset
        
    model.stop_multi_process_pool(pool)
    print("\n🎉 HOÀN TẤT TẤT CẢ! Đã cắt và lưu an toàn toàn bộ Dataset thành từng cục!")

if __name__ == "__main__":
    embed_corpus_kaggle_numpy()
