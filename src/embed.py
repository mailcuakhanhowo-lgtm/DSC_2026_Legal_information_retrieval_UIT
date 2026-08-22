import os
import json
import glob
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config

# Thư mục đầu ra cho các file npy/json (Sếp đã tạo sẵn)
EMBEDED_DATA_DIR = r"D:\Project Vibe Coding\DSC_2026\embeded_data"
# Kích thước mỗi đợt nặn Vector trước khi lưu ra file
CHUNK_SAVE_SIZE = 100000

def embed_corpus_local_numpy():
    print("=========================================================")
    print("🚀 BẮT ĐẦU QUY TRÌNH NẶN VECTOR CHUẨN KAGGLE TẠI NHÀ 🚀")
    print("=========================================================\n")

    # Đảm bảo thư mục lưu trữ đã tồn tại
    os.makedirs(EMBEDED_DATA_DIR, exist_ok=True)

    print("--- BƯỚC 1: ĐỌC DỮ LIỆU TỪ JSONL ---")
    chunk_files = glob.glob(os.path.join(config.DEFAULT_OUTPUT_DIR, "**", "*.jsonl"), recursive=True)
    total_files = len(chunk_files)
    
    if total_files == 0:
        print(f"❌ Không tìm thấy file .jsonl nào trong {config.DEFAULT_OUTPUT_DIR}.")
        print("Sếp nhớ chạy preprocess.py trước nhé!")
        return

    all_texts = []
    all_metadata = []

    print(f"Đang nạp toàn bộ nội dung từ {total_files} file văn bản vào RAM...")
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
                    "text": record["text"], 
                    "link": ""
                })
                
    # Nếu Sếp có bật chế độ Test (JUST_CHECK) trong config.py
    if getattr(config, 'JUST_CHECK', None) is not None:
        print(f"\n⚠️ CHẾ ĐỘ TEST ĐANG BẬT! Chỉ xử lý {config.JUST_CHECK} đoạn văn bản.")
        all_texts = all_texts[:config.JUST_CHECK]
        all_metadata = all_metadata[:config.JUST_CHECK]

    total_chunks = len(all_texts)
    print(f"🎉 Đã nạp xong {total_chunks} đoạn văn bản!")

    print("\n--- BƯỚC 2: KHỞI TẠO MÔ HÌNH AI ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Đang kích hoạt thiết bị: {device.upper()}")
    
    model = SentenceTransformer(config.EMBEDDING_MODEL, trust_remote_code=True, device=device)
    model.max_seq_length = config.MAX_POSITION_EMBEDDINGS
    
    print(f"\n--- BƯỚC 3: VECTOR HÓA & LƯU TỪNG PHẦN ---")
    # Duyệt qua từng đoạn 100.000 văn bản
    for i in range(0, total_chunks, CHUNK_SAVE_SIZE):
        end_idx = min(i + CHUNK_SAVE_SIZE, total_chunks)
        part_num = (i // CHUNK_SAVE_SIZE) + 1
        
        print(f"\n🚀 Đang xử lý Phần {part_num}: Từ {i} đến {end_idx}...")
        
        # Cắt lấy 100k text và metadata
        texts_subset = all_texts[i:end_idx]
        meta_subset = all_metadata[i:end_idx]
        
        # Nặn Vector bằng các thông số máy nhà (batch_size được lấy từ config)
        embeddings = model.encode(
            texts_subset, 
            batch_size=config.EMBEDDING_BATCH_SIZE, 
            show_progress_bar=True,
            normalize_embeddings=True # Giữ chuẩn COSINE
        )
        
        # Lưu ngay lập tức xuống đĩa cứng
        np_path = os.path.join(EMBEDED_DATA_DIR, f"embeddings_part_{part_num}.npy")
        meta_path = os.path.join(EMBEDED_DATA_DIR, f"metadata_part_{part_num}.json")
        
        print(f"💾 Đang lưu {len(texts_subset)} kết quả xuống {EMBEDED_DATA_DIR}...")
        np.save(np_path, embeddings)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_subset, f, ensure_ascii=False)
            
        # Dọn rác giải phóng RAM
        del embeddings
        del texts_subset
        del meta_subset
        
    print(f"\n🎉 HOÀN TẤT TẤT CẢ! Đã cắt và lưu an toàn toàn bộ Dataset thành từng cục trong thư mục {EMBEDED_DATA_DIR}!")

if __name__ == "__main__":
    import codecs
    import sys
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    embed_corpus_local_numpy()
