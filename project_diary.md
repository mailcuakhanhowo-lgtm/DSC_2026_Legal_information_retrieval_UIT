# SỔ NHẬT KÝ DỰ ÁN (PROJECT DIARY)
Dự án: DSC 2026 - Task 1 (Legal Information Retrieval)

Tệp này được quản lý bởi Thư ký tổng quyền (Secretary) nhằm ghi lại tiến độ, báo cáo từ các đặc vụ, và các quyết định quan trọng đã được Người dùng (User) thông qua.

---

## 📅 Cập nhật ngày 14/08/2026
**Người báo cáo:** Ana (Data Analyst)

### 1. Cơ sở hạ tầng & Quản lý Mã nguồn
- Đã khởi tạo thành công Local Git Repository.
- Đã đẩy commit đầu tiên lên Remote GitHub: [DSC_2026_Legal_information_retrieval_UIT](https://github.com/mailcuakhanhowo-lgtm/DSC_2026_Legal_information_retrieval_UIT).
- Cấu trúc lưu trữ: Đã tạo thư mục `src/` để chuẩn bị cho giai đoạn Engineer code. 
- Đã cấu hình tệp `.gitignore` chặn đẩy các file dữ liệu khổng lồ (`.zip`, `.json`) lên GitHub, tránh làm phình to repo.

### 2. Kiến trúc Hệ thống & Luật lệ
- **Kiến trúc chốt:** Hệ thống sẽ áp dụng mô hình **Hybrid Search (Retrieval + Reranking)**.
- **Môi trường chạy:** Tận dụng công nghệ **Colab** để sử dụng sức mạnh tính toán của GPU.
- Mọi thay đổi đều đã được cập nhật vào tệp `file_index.md` và `rule.md` theo đúng quy định.

### 3. Kế hoạch tiếp theo (Next Steps)
- **Nhiệm vụ cho Engineer:** Cần chuẩn bị tạo file `.py` đầu tiên trong thư mục `src/` để thực hiện tải và trích xuất dữ liệu thô phục vụ cho quá trình EDA (Phân tích Khám phá Dữ liệu).

### 4. Quyết định Thiết kế (Từ User & Phân tích)
- **Tiền xử lý dữ liệu (Metadata):** Bắt buộc giữ nguyên `title` (thuộc tính `name` của văn bản) gắn liền với nội dung trong quá trình cắt đoạn/xử lý. Việc này giúp AI luôn nắm được đoạn nội dung đó thuộc về bộ luật/văn bản nào, tăng độ chính xác khi truy xuất.
- **Tiêu chuẩn Cắt đoạn (Chunking):** 
  - **Bắt buộc:** Áp dụng thuật toán **Sliding Window** (Cửa sổ trượt / Chunk Overlap) làm tiêu chuẩn tối thiểu để không làm đứt gãy ngữ cảnh pháp lý giữa các đoạn.
  - **Nâng cao (Định hướng kiến trúc):** Triển khai kiến trúc **Small2Big (Parent-Child Document Retrieval)**. Mô hình Vector sẽ chỉ so khớp trên các đoạn văn bản rất nhỏ và đặc thù (giúp tăng độ nhạy và chính xác). Khi truy xuất thành công, hệ thống sẽ trả về ID của văn bản lớn chứa đoạn nhỏ đó. Đây là phương án Kim Cương hoàn toàn phù hợp với luật chơi của Ban tổ chức.

## 📅 Cập nhật ngày 18/08/2026
**Người báo cáo:** Search Systems Engineer

### 6. Hoàn thiện Hệ thống Baseline Truy xuất Thông tin Pháp luật (BM25 + SQLite)
- **Kiến trúc Modular:**
  - Quy hoạch dự án theo cấu trúc chuẩn: `src/config.py`, `src/preprocess_and_index.py`, `src/inference.py`, `indices/`, `logs/`.
- **Bảo tồn Logic Chunking:** Tích hợp nguyên vẹn 100% hai hàm `filter_and_clean_text` và `smart_legal_chunker` để xử lý 8.532 văn bản luật, tạo 1.384.789 chunks kèm Title Injection `# {name}\n\n{chunk}`.
- **Lưu trữ & Chỉ mục (Giai đoạn 1):**
  - Nạp toàn bộ 1.384.789 chunks vào SQLite DB `indices/legal_corpus.db` (bảng `chunks`, đánh index trên `doc_id`).
  - Phân tách từ tiếng Việt qua PyVi và huấn luyện mô hình BM25Okapi, đóng gói thành `indices/bm25_index.pkl` kèm bảng ánh xạ tra cứu ngược `chunk_doc_map`.
- **Suy luận & Tổng hợp (Giai đoạn 2):**
  - Xây dựng `src/inference.py` truy vấn Top 50 chunks qua BM25, áp dụng cơ chế Document Aggregation (Max Pooling) để lấy điểm cao nhất của từng `doc_id` gốc, cắt lấy Top 5 doc_ids.
  - Định dạng xuất `submission.json` đáp ứng nghiêm ngặt định dạng của Ban tổ chức.
- **Ràng buộc:** Tuyệt đối không dùng GPU, FAISS, hay Reranker; không thêm emoji/icon vào mã nguồn.

---

