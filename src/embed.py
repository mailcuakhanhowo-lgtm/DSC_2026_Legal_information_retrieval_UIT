import os
import json
import torch
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
import glob
import config

def get_all_chunks(processed_dir: str):
    """Quét và lấy đường dẫn của tất cả các file .md trong thư mục processed_data"""
    chunk_files = glob.glob(os.path.join(processed_dir, "**", "*.md"), recursive=True)
    return chunk_files

def extract_metadata_from_path(file_path: str):
    """Trích xuất doc_id và chunk_id từ đường dẫn file"""
    path_parts = os.path.normpath(file_path).split(os.sep)
    chunk_filename = path_parts[-1]
    doc_id = path_parts[-2]
    chunk_id = chunk_filename.replace('.md', '')
    return doc_id, chunk_id

def init_milvus_collection(client, collection_name, dim):
    """Khởi tạo Bảng (Collection) trong Milvus Lite với Schema chuẩn"""
    if client.has_collection(collection_name=collection_name):
        print(f"Bảng '{collection_name}' đã tồn tại. Đang xóa để tạo mới...")
        client.drop_collection(collection_name=collection_name)
        
    print(f"Đang tạo bảng '{collection_name}'...")
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    
    # Định nghĩa các cột (Fields)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=200, is_primary=True)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535) # Lưu nguyên văn bản
    schema.add_field(field_name="link", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
    
    # Tạo Index cho cột Vector để tìm kiếm siêu tốc
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="COSINE", index_type="AUTOINDEX")
    
    # Tạo Collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    print("✅ Đã tạo bảng thành công!")

def embed_corpus():
    # 1. KẾT NỐI MILVUS LITE
    print(f"Kết nối tới Milvus Lite Database: {config.MILVUS_DB_PATH}")
    milvus_client = MilvusClient(uri=config.MILVUS_DB_PATH)
    init_milvus_collection(milvus_client, config.MILVUS_COLLECTION, config.EMBEDDING_DIM)

    # 2. KHỞI TẠO MÔ HÌNH AI
    print(f"\nBắt đầu nạp mô hình: {config.EMBEDDING_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang chạy trên thiết bị: {device.upper()}")
    
    try:
        model = SentenceTransformer(config.EMBEDDING_MODEL, trust_remote_code=True, device=device)
        # THIẾT LẬP KHIÊN BẢO VỆ VRAM: Giới hạn chiều dài đọc hiểu ở mức 2048 token
        model.max_seq_length = config.MAX_POSITION_EMBEDDINGS
        print(f"Đã bật Khiên bảo vệ VRAM: max_seq_length = {model.max_seq_length}")
    except Exception as e:
        print(f"Lỗi nạp mô hình: {e}")
        return

    # 3. CHUẨN BỊ DỮ LIỆU ĐẦU VÀO
    # 3. CHUẨN BỊ DỮ LIỆU ĐẦU VÀO
    print("\nĐang quét các file Markdown (.md) trong thư mục processed_data...")
    chunk_files = get_all_chunks(config.DEFAULT_OUTPUT_DIR)
    total_chunks = len(chunk_files)
    
    if total_chunks == 0:
        print(f"Lỗi: Không tìm thấy file .md nào trong {config.DEFAULT_OUTPUT_DIR}. Vui lòng chạy preprocess.py trước.")
        return
        
    batch_size = getattr(config, 'EMBEDDING_BATCH_SIZE', 2)
    print(f"Đã tìm thấy {total_chunks} đoạn văn bản. Tiến hành Vector hóa (Batch size: {batch_size})...")

    # 4. CHẠY PIPELINE (ĐỌC -> VECTOR HÓA -> INSERT MILVUS)
    from tqdm import tqdm
    pbar = tqdm(total=total_chunks, desc="Nhúng & Lưu vào Milvus", unit="chunk")

    batch_texts = []
    batch_metadata = []
    processed_count = 0

    for i, file_path in enumerate(chunk_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            
        doc_id, chunk_id = extract_metadata_from_path(file_path)
        
        # Trích xuất title từ dòng đầu tiên của Markdown (vd: "# Quyết định số 5868/QĐ-UBND")
        title = ""
        lines = text.split('\n')
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()
            
        batch_texts.append(text)
        batch_metadata.append({
            "chunk_id": f"{doc_id}_{chunk_id}", # Đảm bảo Unique Key (VD: 21_chunk_0)
            "doc_id": doc_id,
            "title": title,
            "link": "" # File MD không lưu link, tạm để trống
        })
        
        # Khi gom đủ 1 mẻ (Batch) hoặc chạm dòng cuối cùng
        if len(batch_texts) >= batch_size or i == total_chunks - 1:
            # Chuyển chữ thành Vector (Bật normalize để dùng được Metric COSINE)
            embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            
            # Gói dữ liệu thành dạng Dictionary chuẩn của Milvus
            data_to_insert = []
            for j in range(len(batch_texts)):
                data_to_insert.append({
                    "chunk_id": batch_metadata[j]["chunk_id"],
                    "doc_id": batch_metadata[j]["doc_id"],
                    "title": batch_metadata[j]["title"],
                    "text": batch_texts[j],
                    "link": batch_metadata[j]["link"],
                    "embedding": embeddings[j].tolist()
                })
            
            # Đẩy toàn bộ cục dữ liệu (Text + ID + Vector) vào Milvus Lite
            milvus_client.insert(collection_name=config.MILVUS_COLLECTION, data=data_to_insert)
            
            processed_count += len(batch_texts)
            pbar.update(len(batch_texts))
            
            # Làm sạch Batch để đón mẻ mới
            batch_texts.clear()
            batch_metadata.clear()
            
    pbar.close()
    
    print(f"\n✅ HOÀN TẤT! Đã Vector hóa và lưu thành công {processed_count} chunks vào Milvus Lite.")
    print(f"Cơ sở dữ liệu lưu tại: {config.MILVUS_DB_PATH}")
    print(f"Tên bảng: {config.MILVUS_COLLECTION}")

if __name__ == "__main__":
    embed_corpus()
