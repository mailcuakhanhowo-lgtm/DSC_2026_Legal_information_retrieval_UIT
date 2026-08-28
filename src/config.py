# ==============================================================================
# BẢNG ĐIỀU KHIỂN TRUNG TÂM (CONFIG.PY)
# Sắp xếp theo đúng thứ tự luồng chạy của Hệ thống tìm kiếm pháp luật (Pipeline)
# ==============================================================================

# ================= 1. CẤU HÌNH DỮ LIỆU GỐC & THỬ NGHIỆM =================
DEFAULT_INPUT_PATH = r"D:\Project Vibe Coding\DSC_2026\selected-contexts\selected-contexts"
QUESTIONS_PUBLIC_FILE = r"D:\Project Vibe Coding\DSC_2026\public-official.json"
# Số lượng văn bản tối đa muốn chạy thử (Testing). Đặt thành None nếu muốn chạy toàn bộ.
JUST_CHECK = None  

# ================= 2. TIỀN XỬ LÝ (PREPROCESSING) =================
# Mục tiêu: Chặt nhỏ văn bản thành các Chunk để tránh tràn ngữ cảnh Model.
DEFAULT_OUTPUT_DIR = r"D:\Project Vibe Coding\DSC_2026\src\processed_data"
MAX_WORDS_PER_CHUNK = 6000  # Giới hạn bởi max_seq_length của model Embedding
OVERLAP_WORDS = 50          # Gối đầu để không đứt đoạn ý nghĩa pháp lý

# ================= 3. TRUY VẤN GIẢ ĐỊNH (HyDE GENERATION) =================
# Mục tiêu: Dùng Local LLM để dịch câu hỏi ngắn thành đoạn văn bản luật mẫu.
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

# ================= 4. LẬP CHỈ MỤC & VÒNG LOẠI TÌM KIẾM (HYBRID SEARCH) =================
# Mục tiêu: Quét 1.3 triệu văn bản để bắt ra một tập hợp các ứng viên tiềm năng nhất (Vòng loại).
# ---------------- Nhánh 4A: Tìm kiếm Ngữ nghĩa (Semantic / Vector) ----------------
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
EMBEDDING_DIM = 1024
MAX_POSITION_EMBEDDINGS = 2048
EMBEDDING_BATCH_SIZE = 32
MILVUS_DB_PATH = r"D:\Project Vibe Coding\DSC_2026\embeded_data\milvus_legal.db"
MILVUS_COLLECTION = "legal_chunks"
# File kết quả xuất ra sau khi quét Vector
SUBMISSION_SEMANTIC_TOP_K = r"D:\Project Vibe Coding\DSC_2026\submission\submission_semantic_topK.json"

# ---------------- Nhánh 4B: Tìm kiếm Từ khóa (BM25 / Lexical) ----------------
# Nơi lưu DB Mục lục ngược (Inverted Index) của BM25 (dùng SQLite FTS5)
BM25_DB_PATH = r"D:\Project Vibe Coding\DSC_2026\embeded_data\bm25_index.db"
# File kết quả xuất ra sau khi quét Từ khóa
SUBMISSION_BM25_TOP_K = r"D:\Project Vibe Coding\DSC_2026\submission\submission_bm25_topK.json"

# ---> Điểm chốt vòng loại: Bắt số lượng K dư dả để tránh lọt lưới
RETRIEVAL_TOP_K = 100

# ================= 5. LAI GHÉP & CHẮT LỌC (RRF FUSION) =================
# Mục tiêu: Trộn 2 danh sách Vòng loại lại với nhau để tìm ra những ứng viên "học đều cả 2 môn".
# Số lượng giữ lại sau khi lai ghép (Cắt bớt để đưa cho Reranker)
RRF_TOP_K = 60
# File lưu kết quả sau khi trộn
SUBMISSION_HYBRID_TOP_K = r"D:\Project Vibe Coding\DSC_2026\submission\submission_hybrid_topK.json"

# ================= 6. VÒNG CHUNG KẾT & NỘP BÀI (RERANKING) =================
# Mục tiêu: Đọc kỹ từng ứng viên trong danh sách lai ghép, chấm điểm chính xác và chốt Top 5.
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
RERANKER_BATCH_SIZE = 4
# File chốt hạ cuối cùng đem nộp Kaggle
FINAL_SUBMISSION = r"D:\Project Vibe Coding\DSC_2026\submission\submission.json"