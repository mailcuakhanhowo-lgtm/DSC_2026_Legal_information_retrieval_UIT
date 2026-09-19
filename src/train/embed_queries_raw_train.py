import os
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

import json
import torch
import config_train as config
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

OUTPUT_DIR = r"D:\Project Vibe Coding\DSC_2026\embed_queries"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DB_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "milvus_queries_raw.db") # LÆ°u thÃ nh DB riÃªng Ä‘á»ƒ dá»… so sÃ¡nh
COLLECTION_NAME = "queries"
LIMIT_QUESTIONS = 100 # Chá»‰ test trÃªn 100 cÃ¢u Ä‘áº§u giá»‘ng há»‡t HyDE

def init_query_collection(client, collection_name, dim):
    if client.has_collection(collection_name=collection_name):
        print(f"Báº£ng '{collection_name}' Ä‘Ã£ tá»“n táº¡i. Äang xÃ³a Ä‘á»ƒ táº¡o má»›i...")
        client.drop_collection(collection_name=collection_name)
        
    print(f"Äang táº¡o báº£ng '{collection_name}'...")
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    
    # Äá»‹nh nghÄ©a cÃ¡c trÆ°á»ng (Fields)
    schema.add_field(field_name="query_id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
    schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="ground_truth", datatype=DataType.VARCHAR, max_length=1000) # LÆ°u list doc_id dáº¡ng chuá»—i
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
    
    # Táº¡o Index cho cá»™t Vector
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="COSINE", index_type="AUTOINDEX")
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    print("âœ… ÄÃ£ táº¡o báº£ng thÃ nh cÃ´ng!")

def embed_queries_raw():
    print("\n[Há»† THá»NG] Báº¯t Ä‘áº§u quÃ¡ trÃ¬nh Vector hÃ³a (Embed) cÃ¢u há»i Gá»C (KhÃ´ng HyDE)...")

    # 1. Káº¾T Ná»I MILVUS
    print(f"\n1. Káº¿t ná»‘i tá»›i Milvus Database má»›i: {DB_OUTPUT_PATH}")
    client = MilvusClient(uri=DB_OUTPUT_PATH)
    init_query_collection(client, COLLECTION_NAME, config.EMBEDDING_DIM)

    # 2. Äá»ŒC Dá»® LIá»†U Gá»C (TRAIN.JSON)
    train_path = r"D:\Project Vibe Coding\DSC_2026\train.json"
    if not os.path.exists(train_path):
        print(f"âŒ Lá»–I: KhÃ´ng tÃ¬m tháº¥y file {train_path}")
        return
        
    print(f"\n2. Äang Ä‘á»c cÃ¡c truy váº¥n gá»‘c tá»« {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Láº¥y Ä‘Ãºng 100 cÃ¢u Ä‘áº§u tiÃªn
    query_ids = list(train_data.keys())[:LIMIT_QUESTIONS]
    questions = [train_data[qid]["question"] for qid in query_ids]
    ground_truths = [json.dumps(train_data[qid]["answer"]) for qid in query_ids]
    
    print(f"   ÄÃ£ táº£i {len(query_ids)} cÃ¢u truy váº¥n gá»‘c.")

    # 3. KHá»žI Táº O MÃ” HÃŒNH NHÃšNG
    print(f"\n3. Äang náº¡p mÃ´ hÃ¬nh nhÃºng: {config.EMBEDDING_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBEDDING_MODEL, trust_remote_code=True, device=device)
    model.max_seq_length = config.MAX_POSITION_EMBEDDINGS
    print("âœ… Táº£i mÃ´ hÃ¬nh thÃ nh cÃ´ng!")

    # 4. NHÃšNG (EMBED) CÃC CÃ‚U Há»ŽI Gá»C THÃ€NH VECTOR
    print("\n4. Äang chuyá»ƒn Ä‘á»•i question thÃ nh Vectors...")
    query_embeddings = model.encode(
        questions, 
        batch_size=config.EMBEDDING_BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )

    # 5. LÆ¯U VÃ€O MILVUS
    print("\n5. Äang lÆ°u Vectors vÃ o Milvus...")
    data_to_insert = []
    for i in range(len(query_ids)):
        data_to_insert.append({
            "query_id": query_ids[i],
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "embedding": query_embeddings[i].tolist()
        })
    
    client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    
    print(f"\nðŸŽ‰ HOÃ€N Táº¤T! ÄÃ£ nhÃºng vÃ  lÆ°u {len(query_ids)} truy váº¥n gá»‘c (Raw).")
    print(f"CÆ¡ sá»Ÿ dá»¯ liá»‡u lÆ°u táº¡i: {DB_OUTPUT_PATH}")

if __name__ == "__main__":
    embed_queries_raw()
