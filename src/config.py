# CẤU HÌNH HỆ THỐNG - DỄ THEO DÕI VÀ ĐIỀU CHỈNH

# 1. Đường dẫn Dữ liệu (Paths)
DEFAULT_INPUT_PATH = r"D:\Project Vibe Coding\DSC_2026\selected-contexts\selected-contexts\context_21.json"
DEFAULT_OUTPUT_DIR = r"processed_data"

# 2. Tham số Tiền xử lý (Preprocessing)
# Áp dụng cho Sliding Window (Lớp 2)
MAX_WORDS_PER_CHUNK = 6000  # Phù hợp với ngữ cảnh 8192 tokens
OVERLAP_WORDS = 50          # Số từ gối đầu nhau để không đứt nghĩa

# 3. Thông số Mô hình (Models)
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
EMBEDDING_DIM = 1024
MAX_POSITION_EMBEDDINGS = 8192

# 4. Cấu hình Vector hóa (Embedding Phase)
EMBEDDING_OUTPUT_FILE = r"vector_database.jsonl"
EMBEDDING_BATCH_SIZE = 6  # Mức 6 là điểm cân bằng hoàn hảo nhất cho VRAM T4
