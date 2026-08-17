# CẤU HÌNH HỆ THỐNG - DỄ THEO DÕI VÀ ĐIỀU CHỈNH

# 1. Đường dẫn Dữ liệu (Paths)
DEFAULT_INPUT_PATH = r"D:\Project Vibe Coding\DSC_2026\selected-contexts\selected-contexts\context_21.json"
DEFAULT_OUTPUT_DIR = r"D:\Project Vibe Coding\DSC_2026\src\processed_data"

# 1.5. Cấu hình Chạy thử nghiệm (Testing)
JUST_CHECK = 1000  # Số lượng văn bản (context) tối đa muốn chạy. Đặt thành None nếu muốn chạy toàn bộ.

# 2. Tham số Tiền xử lý (Preprocessing)
# Áp dụng cho Sliding Window (Lớp 2)
MAX_WORDS_PER_CHUNK = 6000  # Sẽ bị giới hạn bởi max_seq_length của model khi Embedding
OVERLAP_WORDS = 50          # Số từ gối đầu nhau để không đứt nghĩa

# 3. Thông số Mô hình (Models)
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
EMBEDDING_DIM = 1024
MAX_POSITION_EMBEDDINGS = 2048

# 4. Cấu hình Milvus Lite (Vector Database)
MILVUS_DB_PATH = "milvus_legal.db"
MILVUS_COLLECTION = "legal_chunks"
EMBEDDING_BATCH_SIZE = 2  # Cố định ở 2 để an toàn tuyệt đối cho card RTX 5050 8GB VRAM
