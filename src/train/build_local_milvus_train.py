import sys
import os
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import glob
import json
import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient, DataType
from config_train import MILVUS_DB_PATH, MILVUS_COLLECTION, EMBEDDING_DIM

# Thư mục lưu các file npy/json đã Embed (cách ly trong src\train)
EMBEDED_DATA_DIR = r"D:\Project Vibe Coding\DSC_2026\src\train\embeded_data"


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

    npy_files = glob.glob(os.path.join(EMBEDED_DATA_DIR, "*.npy"))
    if not npy_files:
        print(f"❌ LỖI: Không tìm thấy file .npy nào trong thư mục {EMBEDED_DATA_DIR}")
        print("Sếp nhớ chạy embed_train.py trước nhé!")
        return

    print(f"✅ Đã tìm thấy {len(npy_files)} cục Vector (.npy).")

    milvus_client = MilvusClient(uri=MILVUS_DB_PATH)
    os.makedirs(os.path.dirname(MILVUS_DB_PATH), exist_ok=True)
    init_milvus_collection(milvus_client, MILVUS_COLLECTION, EMBEDDING_DIM)

    total_inserted = 0

    for npy_path in sorted(npy_files):
        base_name = os.path.basename(npy_path)
        part_name = base_name.replace("embeddings_", "").replace(".npy", "")
        json_path = os.path.join(os.path.dirname(npy_path), f"metadata_{part_name}.json")

        if not os.path.exists(json_path):
            print(f"⚠️ CẢNH BÁO: Bỏ qua {base_name} vì không tìm thấy file json đi kèm: {json_path}")
            continue

        print(f"\n⏳ Đang xử lý cục: {part_name}...")

        embeddings = np.load(npy_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        if len(embeddings) != len(metadata):
            print(f"❌ LỖI NGHIÊM TRỌNG: Độ dài Vector ({len(embeddings)}) không khớp với Metadata ({len(metadata)}) tại {part_name}")
            continue

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

        del embeddings
        del metadata

    print(f"\n🎉 THÀNH CÔNG RỰC RỠ! Đã nhồi tổng cộng {total_inserted} Vector vào Database!")
    print(f"📁 Database hiện tại nằm gọn tại: {MILVUS_DB_PATH}")


if __name__ == "__main__":
    build_local_database()
