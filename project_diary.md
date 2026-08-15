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

## 📅 Báo cáo cập nhật (Module Preprocessing)
**Người báo cáo:** Engineer / User

### 5. Cập nhật Mã nguồn - Module Tiền xử lý (`src/preprocess.py`)
- **Tích hợp Command Line Interface (CLI):** Đã import `sys` và `argparse`. Hỗ trợ nhận tham số từ dòng lệnh (Ví dụ: `--input` cho file đầu vào và `--output` cho thư mục lưu kết quả) để chạy linh hoạt với nhiều bộ dữ liệu khác nhau.
- **Cải tiến Luồng Thực thi (Execution Flow):** 
  - Nếu truyền đủ tham số, hệ thống sẽ thực thi hàm xử lý dữ liệu thật `process_corpus()`.
  - Nếu chạy không tham số, hệ thống tự động rơi vào chế độ **DRY RUN** (Chạy thử nghiệm) để in ra console quy trình cắt đoạn của văn bản mẫu (ID: 740) và hiển thị Help text gợi ý.
- **Đánh giá từ Reviewer:** Bản cập nhật chuẩn mực. Việc dùng `argparse` giúp script tránh bị hard-code tên file, là tiền đề vững chắc để tự động hóa toàn bộ Stage 0 trên Google Colab.

---
