import os
import json
import glob
import torch
from sentence_transformers import SentenceTransformer
import config

def get_all_chunks(processed_dir: str):
    """Quét và lấy đường dẫn của tất cả các file .md trong thư mục processed_data"""
    chunk_files = glob.glob(os.path.join(processed_dir, "**", "*.md"), recursive=True)
    return chunk_files

def extract_metadata_from_path(file_path: str):
    """
    Trích xuất doc_id và chunk_id từ đường dẫn file
    Ví dụ: processed_data/21/chunk_0.md -> doc_id='21', chunk_id='chunk_0'
    """
    path_parts = os.path.normpath(file_path).split(os.sep)
    chunk_filename = path_parts[-1]
    doc_id = path_parts[-2]
    chunk_id = chunk_filename.replace('.md', '')
    return doc_id, chunk_id

def embed_corpus():
    print(f"Bắt đầu nạp mô hình: {config.EMBEDDING_MODEL}...")
    # Tự động chọn GPU nếu có, ngược lại dùng CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang chạy trên thiết bị: {device.upper()}")
    
    # Nạp mô hình SentenceTransformer
    # trust_remote_code=True thường cần thiết cho các mô hình custom trên HuggingFace
    try:
        model = SentenceTransformer(config.EMBEDDING_MODEL, trust_remote_code=True, device=device)
    except Exception as e:
        print(f"Lỗi nạp mô hình: {e}")
        return

    # Quét file
    chunk_files = get_all_chunks(config.DEFAULT_OUTPUT_DIR)
    total_chunks = len(chunk_files)
    print(f"Đã tìm thấy {total_chunks} đoạn văn bản (chunks) cần Vector hóa.")
    
    if total_chunks == 0:
        print("Không có dữ liệu để nhúng. Vui lòng chạy preprocess.py trước.")
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
    
    # Khởi tạo kho lưu trữ tổng (giữ trên RAM, an toàn với ~100k vectors)
    all_embeddings = []
    all_metadata_records = []
    processed_count = 0

    for i, file_path in enumerate(chunk_files):
        # Đọc nội dung
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        doc_id, chunk_id = extract_metadata_from_path(file_path)
        
        batch_texts.append(text)
        batch_metadata.append({"doc_id": doc_id, "chunk_id": chunk_id})
        
        # Xử lý khi Batch đầy hoặc là chunk cuối cùng
        if len(batch_texts) >= batch_size or i == total_chunks - 1:
            # Tạo Embedding bằng mô hình
            embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            
            # Đưa ma trận vector của batch này vào kho tổng
            all_embeddings.append(embeddings)
            
            # Đưa metadata vào kho tổng (thứ tự khớp 100% với vector)
            for j in range(len(batch_texts)):
                record = {
                    "doc_id": batch_metadata[j]["doc_id"],
                    "chunk_id": batch_metadata[j]["chunk_id"],
                    # BM25 sẽ được tính ở Pipeline riêng
                }
                all_metadata_records.append(record)
            
            processed_count += len(batch_texts)
            print(f"Đã xử lý: {processed_count}/{total_chunks} chunks...")
            
            # Xóa batch tạm để nạp batch mới
            batch_texts.clear()
            batch_metadata.clear()
            
    # 1. Hợp nhất tất cả các ma trận con thành 1 ma trận duy nhất (N x 768)
    if not all_embeddings:
        print("Lỗi: Không có vector nào được sinh ra. Quá trình nhúng thất bại.")
        return
        
    final_embeddings = np.vstack(all_embeddings)
    
    # 2. Lưu file .npy (Chứa thuần số thực, tốc độ đọc siêu tốc)
    np.save(npy_output_file, final_embeddings)
    
    # 3. Lưu file metadata .jsonl (Chứa ID để ánh xạ)
    with open(meta_output_file, 'w', encoding='utf-8') as out_f:
        for record in all_metadata_records:
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"✅ Hoàn tất Vector hóa. Đã lưu {processed_count} vectors.")
    print(f" - Vector ma trận: {npy_output_file} (Shape: {final_embeddings.shape})")
    print(f" - Metadata ánh xạ: {meta_output_file}")

if __name__ == "__main__":
    embed_corpus()
