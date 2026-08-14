# Chỉ Mục Dự Án: DSC 2026 - Task 1 (LegalIR)

Đây là tài liệu chỉ mục các tệp dữ liệu và tài liệu quan trọng trong thư mục dự án, được thiết lập để team (Analyst, Engineer, Reviewer) dễ dàng theo dõi và truy xuất.

## 1. Dữ liệu huấn luyện và kiểm thử (Queries / Datasets)
- **`train.json`** (`D:\Project Vibe Coding\DSC_2026\train.json`): Tập dữ liệu huấn luyện chính để phát triển và tinh chỉnh (fine-tune) mô hình.
- **`warmup.json`** (`D:\Project Vibe Coding\DSC_2026\warmup.json`): Tập dữ liệu mẫu (kích thước nhỏ) phục vụ vòng Warm-up, giúp team làm quen với input/output và luồng xử lý.
- **`public-official.json`** (`D:\Project Vibe Coding\DSC_2026\public-official.json`): Tập câu hỏi dùng trong giai đoạn Public Test (dùng để chạy mô hình, dự đoán và nộp bài lên hệ thống).

## 2. Kho ngữ liệu pháp luật (Knowledge Base / Corpus)
- **`selected-contexts.zip`** (`D:\Project Vibe Coding\DSC_2026\selected-contexts.zip`): Kho văn bản được chọn lọc phục vụ cho quá trình truy vấn (Retrieval). Khi giải nén sẽ gồm nhiều tệp `context_*.json`. 
  - Mỗi văn bản chứa các thông tin: `id`, `name` (tiêu đề), `link` (đường dẫn gốc), và `passage` (nội dung chi tiết dùng để tìm kiếm).

## 3. Tài liệu hướng dẫn & Quy định
- **`DSC2026_Task1_LegalIR_Data_Overview.pdf`** (`D:\Project Vibe Coding\DSC_2026\DSC2026_Task1_LegalIR_Data_Overview.pdf`) và bản gốc `.docx` (`D:\Project Vibe Coding\DSC_2026\DSC2026_Task1_LegalIR_Data_Overview.docx`): Tài liệu cốt lõi mô tả chi tiết bài toán:
  - Mục tiêu bài toán.
  - Cấu trúc dữ liệu.
  - Công thức đánh giá (độ đo chính: Recall, độ đo phụ: Precision).
  - Quy định nộp bài (file `submission.zip` chứa `submission.json`).
  - **Ràng buộc cực kỳ quan trọng**: Tối đa 5 `document_id` cho mỗi câu hỏi, nếu vượt quá sẽ bị 0 điểm.
- **`[DSC@UIT 2026] Danh sách mô hình.xlsx`** (`D:\Project Vibe Coding\DSC_2026\[DSC@UIT 2026] Danh sách mô hình.xlsx`): Danh sách mô hình gốc (dạng bảng tính).
- **`allowed_models.md`** (`D:\Project Vibe Coding\DSC_2026\allowed_models.md`): File trích xuất (Markdown) danh sách các mô hình được phép sử dụng từ file Excel trên.

## 4. Công cụ đánh giá (Evaluation Tool)
- **`Scoring-Program-Task-LegalIR.zip`** (`D:\Project Vibe Coding\DSC_2026\Scoring-Program-Task-LegalIR.zip`): Chương trình chấm điểm (Scoring Program) chuẩn từ ban tổ chức. 
  - *Ghi chú cho Engineer:* Cần giải nén và sử dụng code trong này để xây dựng bộ đánh giá Offline (Local Validation) nhằm đo lường điểm Recall/Precision chuẩn xác nhất trong lúc huấn luyện.

## 5. Các tệp quản lý nội bộ & tệp khác
- **`file_index.md`** (`D:\Project Vibe Coding\DSC_2026\file_index.md`): Tệp chỉ mục lưu trữ danh sách và đường dẫn toàn bộ các file trong dự án.
- **`rule.md`** (`D:\Project Vibe Coding\DSC_2026\rule.md`): Tệp lưu trữ các "Điều luật thép" (quy định cốt lõi) bắt buộc tuân thủ cho mọi thành viên/đặc vụ.
- **`read_docx.ps1`** (`D:\Project Vibe Coding\DSC_2026\read_docx.ps1`): Script PowerShell nháp tạo trong quá trình hệ thống cố gắng đọc file docx (có thể xóa).
