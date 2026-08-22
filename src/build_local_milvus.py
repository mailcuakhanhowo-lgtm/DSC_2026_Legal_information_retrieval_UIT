import os
import glob
import json
import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient, DataType

# Import cấu hình từ file config
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MILVUS_DB_PATH, MILVUS_COLLECTION, EMBEDDING_DIM

# Thư mục Sếp lưu các file tải về từ Kaggle
EMBEDED_DATA_DIR = r"D:\Project Vibe Coding\DSC_2026\embeded_data"

def init_milvus_collection(client, collection_name, dim):
    if client.has_collection(collection_name=collection_name):
        print(f"⚠️ Bảng {collection_name} đã tồn tại. Đang xóa để tạo mới...")
        client.drop_collection(collection_name=collection_name)
        
    print(f"🔨 Đang tạo bảng {collection_name} trong Milvus...")
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

def build_local_database():
    print("==================================================")
    print("🚀 BẮT ĐẦU ÉP KHUÔN DỮ LIỆU VÀO DATABASE LOCAL 🚀")
    print("==================================================\n")

    # 1. Tìm tất cả các file cặp .npy và .json (Quét đệ quy mọi ngóc ngách)
    npy_files = glob.glob(os.path.join(EMBEDED_DATA_DIR, "**", "*.npy"), recursive=True)
    if not npy_files:
        print(f"❌ LỖI: Không tìm thấy file .npy nào trong thư mục {EMBEDED_DATA_DIR}")
        print("Sếp nhớ tải file từ Kaggle về và bỏ vào đúng thư mục nhé!")
        return
        
    print(f"✅ Đã tìm thấy {len(npy_files)} cục Vector (.npy).")
    
    # 2. Khởi tạo Database
    milvus_client = MilvusClient(uri=MILVUS_DB_PATH)
    init_milvus_collection(milvus_client, MILVUS_COLLECTION, EMBEDDING_DIM)
    
    total_inserted = 0
    
    # 3. Duyệt qua từng cặp file và chèn vào DB
    for npy_path in sorted(npy_files):
        # Tìm file json tương ứng nằm cùng thư mục với file npy
        base_name = os.path.basename(npy_path)
        part_name = base_name.replace("embeddings_", "").replace(".npy", "")
        json_path = os.path.join(os.path.dirname(npy_path), f"metadata_{part_name}.json")
        
        if not os.path.exists(json_path):
            print(f"⚠️ CẢNH BÁO: Bỏ qua {base_name} vì không tìm thấy file json đi kèm: {json_path}")
            continue
            
        print(f"\n⏳ Đang xử lý cục: {part_name}...")
        
        # Load dữ liệu lên RAM
        print("   - Đang load Vector và Metadata vào RAM...")
        embeddings = np.load(npy_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        if len(embeddings) != len(metadata):
            print(f"❌ LỖI NGHIÊM TRỌNG: Độ dài Vector ({len(embeddings)}) không khớp với Metadata ({len(metadata)}) tại {part_name}")
            continue
            
        # Chia nhỏ để chèn cho an toàn, tránh vỡ RAM của Milvus Lite
        insert_batch_size = 5000
        num_items = len(embeddings)
        
        print(f"   - Đang chèn {num_items} dòng vào Milvus...")
        for i in tqdm(range(0, num_items, insert_batch_size), desc="   Chèn vào DB"):
            end_idx = min(i + insert_batch_size, num_items)
            data_to_insert = []
            
            for j in range(i, end_idx):
                data_to_insert.append({
                    "chunk_id": metadata[j]["chunk_id"],
                    "doc_id": metadata[j]["doc_id"],
                    "title": metadata[j]["title"],
                    "text": metadata[j]["text"],
                    "link": metadata[j].get("link", ""),
                    "embedding": embeddings[j].tolist()
                })
                
            milvus_client.insert(collection_name=MILVUS_COLLECTION, data=data_to_insert)
            total_inserted += (end_idx - i)
            
        # Giải phóng RAM cho file tiếp theo
        del embeddings
        del metadata
        
    print(f"\n🎉 THÀNH CÔNG RỰC RỠ! Đã nhồi tổng cộng {total_inserted} Vector vào Database!")
    print(f"📁 Database hiện tại nặng chịch, nằm gọn tại: {MILVUS_DB_PATH}")

if __name__ == "__main__":
    import codecs
    import sys
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    build_local_database()
