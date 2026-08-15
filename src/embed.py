import os
import json
import torch
from sentence_transformers import SentenceTransformer
import config

def embed_corpus():
    print(f"Bắt đầu nạp mô hình: {config.EMBEDDING_MODEL}...")
    # Tự động chọn GPU nếu có, ngược lại dùng CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang chạy trên thiết bị: {device.upper()}")
    
    # Nạp mô hình SentenceTransformer
    try:
        model = SentenceTransformer(config.EMBEDDING_MODEL, trust_remote_code=True, device=device)
    except Exception as e:
        print(f"Lỗi nạp mô hình: {e}")
        return

    # Đọc file corpus.jsonl
    input_jsonl = os.path.join(config.DEFAULT_OUTPUT_DIR, 'corpus.jsonl')
    if not os.path.exists(input_jsonl):
        print(f"Lỗi: Không tìm thấy {input_jsonl}. Vui lòng chạy preprocess.py trước.")
        return
        
    print("Đang đếm số lượng đoạn văn bản...")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        total_chunks = sum(1 for _ in f)
        
    print(f"Đã tìm thấy {total_chunks} đoạn văn bản (chunks) cần Vector hóa.")
    
    if total_chunks == 0:
        print("Không có dữ liệu để nhúng. Vui lòng kiểm tra lại corpus.jsonl.")
        return

    import numpy as np
    
    # Xác định tên file đầu ra
    base_output = getattr(config, 'EMBEDDING_OUTPUT_FILE', 'embeddings.jsonl')
    if base_output.endswith('.jsonl'):
        npy_output_file = base_output.replace('.jsonl', '.npy')
        meta_output_file = base_output.replace('.jsonl', '_meta.jsonl')
    else:
        npy_output_file = base_output + '.npy'
        meta_output_file = base_output + '_meta.jsonl'
        
    batch_size = getattr(config, 'EMBEDDING_BATCH_SIZE', 32)
    
    print(f"Tiến hành Vector hóa... Batch size: {batch_size}.")
    print(f"Lưu Vector tại: {npy_output_file} | Lưu Metadata tại: {meta_output_file}")
    
    # Khởi tạo batch tạm
    batch_texts = []
    batch_metadata = []
    processed_count = 0

    from tqdm import tqdm
    pbar = tqdm(total=total_chunks, desc="Vector hóa", unit="chunk")

    # Khởi tạo kho lưu trữ tổng
    all_embeddings = []
    all_metadata_records = []

    with open(input_jsonl, 'r', encoding='utf-8') as in_f:
        for i, line in enumerate(in_f):
            record_in = json.loads(line)
            text = record_in.get("text", "")
            
            batch_texts.append(text)
            batch_metadata.append({
                "doc_id": record_in.get("doc_id", ""),
                "chunk_id": record_in.get("chunk_id", ""),
                "title": record_in.get("title", ""),
                "link": record_in.get("link", "")
            })
            
            # Xử lý khi Batch đầy hoặc là chunk cuối cùng
            if len(batch_texts) >= batch_size or i == total_chunks - 1:
                # Tạo Embedding
                embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
                all_embeddings.append(embeddings)
                
                # Đưa metadata vào kho tổng
                for j in range(len(batch_texts)):
                    record = {
                        "doc_id": batch_metadata[j]["doc_id"],
                        "chunk_id": batch_metadata[j]["chunk_id"],
                        "title": batch_metadata[j]["title"],
                        "link": batch_metadata[j]["link"]
                    }
                    all_metadata_records.append(record)
                
                processed_count += len(batch_texts)
                pbar.update(len(batch_texts))
                
                batch_texts.clear()
                batch_metadata.clear()
                
    pbar.close()
            
    # Hợp nhất ma trận
    if not all_embeddings:
        print("Lỗi: Không có vector nào được sinh ra.")
        return
        
    final_embeddings = np.vstack(all_embeddings)
    
    # Lưu file
    np.save(npy_output_file, final_embeddings)
    
    with open(meta_output_file, 'w', encoding='utf-8') as out_f:
        for record in all_metadata_records:
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"✅ Hoàn tất Vector hóa. Đã lưu {processed_count} vectors.")
    print(f" - Vector ma trận: {npy_output_file} (Shape: {final_embeddings.shape})")
    print(f" - Metadata ánh xạ: {meta_output_file}")

if __name__ == "__main__":
    embed_corpus()
