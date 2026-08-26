# CẤU HÌNH HỆ THỐNG - DỄ THEO DÕI VÀ ĐIỀU CHỈNH

# 1. Đường dẫn Dữ liệu (Paths)
DEFAULT_INPUT_PATH = r"D:\Project Vibe Coding\DSC_2026\selected-contexts\selected-contexts"
DEFAULT_OUTPUT_DIR = r"D:\Project Vibe Coding\DSC_2026\src\processed_data"

# 1.5. Cấu hình Chạy thử nghiệm (Testing)
JUST_CHECK = None  # Số lượng văn bản (context) tối đa muốn chạy. Đặt thành None nếu muốn chạy toàn bộ.

# 2. Tham số Tiền xử lý (Preprocessing)
# Áp dụng cho Sliding Window (Lớp 2)
MAX_WORDS_PER_CHUNK = 6000  # Sẽ bị giới hạn bởi max_seq_length của model khi Embedding
OVERLAP_WORDS = 50          # Số từ gối đầu nhau để không đứt nghĩa

# 3. Thông số Mô hình (Models)
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
EMBEDDING_DIM = 1024
MAX_POSITION_EMBEDDINGS = 2048

# 4. Cấu hình Milvus Lite (Vector Database)
MILVUS_DB_PATH = r"D:\Project Vibe Coding\DSC_2026\embeded_data\milvus_legal.db"
MILVUS_COLLECTION = "legal_chunks"
EMBEDDING_BATCH_SIZE = 32  # Tăng lên 32 để tận dụng sức mạnh GPU (VRAM đang dư dả)

# 5. Cấu hình Local LLM (HyDE Generation)
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LLM_CACHE_DIR = r"D:\Project Vibe Coding\DSC_2026\local_llm_data\.hf_cache"
LLM_INPUT_QUERIES = r"D:\Project Vibe Coding\DSC_2026\train.json"
LLM_OUTPUT_QUERIES = r"D:\Project Vibe Coding\DSC_2026\local_llm_data\hyde_queries.json"
LLM_TEST_LIMIT = 100
LLM_SYSTEM_PROMPT = """Bạn là một chuyên gia pháp lý và là một công cụ hỗ trợ tìm kiếm văn bản pháp luật.
Nhiệm vụ của bạn là đọc câu hỏi của người dùng và tạo ra một đoạn văn bản pháp luật giả định (HyDE) để làm mồi tìm kiếm.
YÊU CẦU KỸ THUẬT NGHIÊM NGẶT (TUYỆT ĐỐI TUÂN THỦ):
1. VĂN PHONG: Sử dụng từ vựng pháp lý chuyên ngành, hành văn mạch lạc, chặt chẽ. Phải diễn đạt lại và mở rộng từ khóa trong câu hỏi một cách tự nhiên, TUYỆT ĐỐI KHÔNG sao chép y nguyên câu hỏi như một cái máy.
2. CẤM XUỐNG DÒNG: Câu trả lời phải là ĐÚNG MỘT ĐOẠN VĂN DUY NHẤT. Tuyệt đối không được có bất kỳ dấu xuống dòng (Enter) nào.
3. CẤM BỊA TIÊU ĐỀ: Tuyệt đối KHÔNG được có chữ "Điều", "Khoản", "Điểm", "Chương" ở đầu câu trả lời. Hãy viết thẳng vào phần nội dung quy định.
4. CẤM RÁC: Tuyệt đối KHÔNG chào hỏi, KHÔNG giải thích. Chỉ in ra đúng nội dung đoạn văn giả định."""

# ================= CẤU HÌNH RERANKER (PHASE 2) =================
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
RERANKER_BATCH_SIZE = 4