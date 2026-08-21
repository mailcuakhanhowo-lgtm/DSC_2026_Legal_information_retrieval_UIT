import os
import json
import torch
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
import glob
from tqdm import tqdm

# Cấu hình đường dẫn trên Kaggle
# Khi upload dataset lên Kaggle, thường nó nằm ở /kaggle/input/tên-dataset/...
# Sếp cần đổi đường dẫn này trỏ tới thư mục chứa các file .md của mình
INPUT_DIR = "/kaggle/working/processed_data"
MILVUS_DB_PATH = "/kaggle/working/milvus_legal.db"
MILVUS_COLLECTION = "legal_chunks"
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
EMBEDDING_DIM = 1024
MAX_POSITION_EMBEDDINGS = 2048
BATCH_SIZE = 32 # Giảm xuống 32 để T4 16GB không bị quá tải (Sequence length 2048 rất tốn RAM)

def init_milvus_collection(client, collection_name, dim):
    if client.has_collection(collection_name=collection_name):
        print(f"Bảng {collection_name} đã tồn tại. Đang xóa để tạo mới...")
        client.drop_collection(collection_name=collection_name)
        
    print(f"Đang tạo bảng {collection_name}...")
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=200, is_primary=True)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="link", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
    
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="COSINE", index_type="AUTOINDEX")
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

def embed_corpus_kaggle():
    print("--- BƯỚC 1: CHUẨN BỊ DỮ LIỆU ---")
    chunk_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.jsonl"), recursive=True)
    total_files = len(chunk_files)
    if total_files == 0:
        print(f"Không tìm thấy file .jsonl nào trong {INPUT_DIR}.")
        print("Sếp nhớ kiểm tra lại biến INPUT_DIR xem đã trỏ đúng vào thư mục Dataset trên Kaggle chưa nhé!")
        return

    all_texts = []
    all_metadata = []

    print("Đang đọc nội dung các file JSONL (RAM 30GB của Kaggle dư sức chứa)...")
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
                    "link": ""
                })
                
    total_chunks = len(all_texts)

    print("\n--- BƯỚC 2: ĐÁNH THỨC 2 QUÁI VẬT T4 (MULTI-GPU) ---")
    # Tải mô hình
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    model.max_seq_length = MAX_POSITION_EMBEDDINGS
    
    # Kích hoạt tính năng Multi-Process của thư viện
    pool = model.start_multi_process_pool()
    
    print(f"Đang Vector hóa {total_chunks} đoạn (Batch Size: {BATCH_SIZE}/GPU) trên 2xT4...")
    
    # Dùng hàm encode mới nhất, hỗ trợ thanh tiến trình (show_progress_bar)
    embeddings = model.encode(all_texts, pool=pool, batch_size=BATCH_SIZE, show_progress_bar=True)
    
    # Giải phóng GPU sau khi xong
    model.stop_multi_process_pool(pool)
    print("✅ Hoàn tất Vector hóa!")

    print("\n--- BƯỚC 3: LƯU VÀO DATABASE MILVUS LITE ---")
    milvus_client = MilvusClient(uri=MILVUS_DB_PATH)
    init_milvus_collection(milvus_client, MILVUS_COLLECTION, EMBEDDING_DIM)
    
    # Chèn dữ liệu vào Milvus (Chia mẻ 5000 để an toàn cho ổ cứng Kaggle)
    insert_batch_size = 5000
    for i in tqdm(range(0, total_chunks, insert_batch_size), desc="Đang chèn vào Milvus"):
        end_idx = min(i + insert_batch_size, total_chunks)
        data_to_insert = []
        for j in range(i, end_idx):
            data_to_insert.append({
                "chunk_id": all_metadata[j]["chunk_id"],
                "doc_id": all_metadata[j]["doc_id"],
                "title": all_metadata[j]["title"],
                "text": all_texts[j],
                "link": all_metadata[j]["link"],
                "embedding": embeddings[j].tolist()
            })
        milvus_client.insert(collection_name=MILVUS_COLLECTION, data=data_to_insert)
        
    print(f"\n🎉 HOÀN TẤT DỰ ÁN!")
    print(f"Toàn bộ Database {total_chunks} vector đã được nén vào file: {MILVUS_DB_PATH}")
    print("Sếp hãy nhìn sang bảng bên phải của Kaggle (mục Output) và tải file milvus_legal.db về máy nhé!")

if __name__ == "__main__":
    embed_corpus_kaggle()

