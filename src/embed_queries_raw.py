import os
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

import json
import torch
import config
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

OUTPUT_DIR = r"D:\Project Vibe Coding\DSC_2026\embed_queries"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DB_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "milvus_queries_raw.db") # Lưu thành DB riêng để dễ so sánh
COLLECTION_NAME = "queries"
LIMIT_QUESTIONS = 100 # Chỉ test trên 100 câu đầu giống hệt HyDE

def init_query_collection(client, collection_name, dim):
    if client.has_collection(collection_name=collection_name):
        print(f"Bảng '{collection_name}' đã tồn tại. Đang xóa để tạo mới...")
        client.drop_collection(collection_name=collection_name)
        
    print(f"Đang tạo bảng '{collection_name}'...")
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    
    # Định nghĩa các trường (Fields)
    schema.add_field(field_name="query_id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
    schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="ground_truth", datatype=DataType.VARCHAR, max_length=1000) # Lưu list doc_id dạng chuỗi
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
    
    # Tạo Index cho cột Vector
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="COSINE", index_type="AUTOINDEX")
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    print("✅ Đã tạo bảng thành công!")

def embed_queries_raw():
    print("\n[HỆ THỐNG] Bắt đầu quá trình Vector hóa (Embed) câu hỏi GỐC (Không HyDE)...")

    # 1. KẾT NỐI MILVUS
    print(f"\n1. Kết nối tới Milvus Database mới: {DB_OUTPUT_PATH}")
    client = MilvusClient(uri=DB_OUTPUT_PATH)
    init_query_collection(client, COLLECTION_NAME, config.EMBEDDING_DIM)

    # 2. ĐỌC DỮ LIỆU GỐC (TRAIN.JSON)
    train_path = r"D:\Project Vibe Coding\DSC_2026\train.json"
    if not os.path.exists(train_path):
        print(f"❌ LỖI: Không tìm thấy file {train_path}")
        return
        
    print(f"\n2. Đang đọc các truy vấn gốc từ {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Lấy đúng 100 câu đầu tiên
    query_ids = list(train_data.keys())[:LIMIT_QUESTIONS]
    questions = [train_data[qid]["question"] for qid in query_ids]
    ground_truths = [json.dumps(train_data[qid]["answer"]) for qid in query_ids]
    
    print(f"   Đã tải {len(query_ids)} câu truy vấn gốc.")

    # 3. KHỞI TẠO MÔ HÌNH NHÚNG
    print(f"\n3. Đang nạp mô hình nhúng: {config.EMBEDDING_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBEDDING_MODEL, trust_remote_code=True, device=device)
    model.max_seq_length = config.MAX_POSITION_EMBEDDINGS
    print("✅ Tải mô hình thành công!")

    # 4. NHÚNG (EMBED) CÁC CÂU HỎI GỐC THÀNH VECTOR
    print("\n4. Đang chuyển đổi question thành Vectors...")
    query_embeddings = model.encode(
        questions, 
        batch_size=config.EMBEDDING_BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )

    # 5. LƯU VÀO MILVUS
    print("\n5. Đang lưu Vectors vào Milvus...")
    data_to_insert = []
    for i in range(len(query_ids)):
        data_to_insert.append({
            "query_id": query_ids[i],
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "embedding": query_embeddings[i].tolist()
        })
    
    client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    
    print(f"\n🎉 HOÀN TẤT! Đã nhúng và lưu {len(query_ids)} truy vấn gốc (Raw).")
    print(f"Cơ sở dữ liệu lưu tại: {DB_OUTPUT_PATH}")

if __name__ == "__main__":
    embed_queries_raw()
