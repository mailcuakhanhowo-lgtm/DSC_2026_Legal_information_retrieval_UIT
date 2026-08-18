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
- **`evaluate.py`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\evaluate.py`): Script Python đánh giá cục bộ (Offline Evaluation) tính toán Precision@5, Recall@5 và F1-Score giữa file kết quả dự đoán (`submission.json`) và file nhãn chuẩn Ground Truth (`train.json`/`warmup.json`).
- **`scoring.py`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\scoring.py`): Script chấm điểm chuẩn mô phỏng môi trường Docker của BTC.
- **`Scoring-Program-Task-LegalIR.zip`** (`D:\Project Vibe Coding\DSC_2026\Scoring-Program-Task-LegalIR.zip`): Gói chương trình chấm điểm gốc từ Ban tổ chức.

## 5. Các tệp quản lý nội bộ & tệp khác
- **`file_index.md`** (`D:\Project Vibe Coding\DSC_2026\file_index.md`): Tệp chỉ mục lưu trữ danh sách và đường dẫn toàn bộ các file trong dự án.
- **`rule.md`** (`D:\Project Vibe Coding\DSC_2026\rule.md`): Tệp lưu trữ các "Điều luật thép" (quy định cốt lõi) bắt buộc tuân thủ cho mọi thành viên/đặc vụ.
- **`project_diary.md`** (`D:\Project Vibe Coding\DSC_2026\project_diary.md`): Sổ nhật ký ghi lại tiến trình và các quyết định quan trọng của dự án.

- **`read_docx.ps1`** (`D:\Project Vibe Coding\DSC_2026\read_docx.ps1`): Script PowerShell nháp tạo trong quá trình hệ thống cố gắng đọc file docx (có thể xóa).

## 6. Mã nguồn & Mô hình Chỉ mục (Source Code & Indices)
- **`run_public_test.py`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\run_public_test.py`): Script chạy thực nghiệm suy luận nhanh trên tập public test và xuất file `submission.json`.
- **`src/`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\src\`): Thư mục chính chứa mã nguồn Python của dự án.
- **`src/config.py`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\src\config.py`): Quản lý tập trung các đường dẫn môi trường (Local & Docker BTC) và tham số cấu hình hệ thống.
- **`src/preprocess_and_index.py`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\src\preprocess_and_index.py`): Module Giai đoạn 1 (Offline Indexing) - Làm sạch, cắt đoạn pháp luật (bảo tồn logic), nạp SQLite Database và build mô hình BM25Okapi với PyVi.
- **`src/inference.py`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\src\inference.py`): Module Giai đoạn 2 (Runtime Inference) - Truy vấn BM25 Top 50 Vectorized Inverted Index, Document Aggregation (Max Pooling), cắt Top 5 doc_ids và xuất file `submission.json` chuẩn BTC.
- **`src/preprocess.py`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\src\preprocess.py`): Script tiền xử lý dữ liệu thử nghiệm ban đầu.

## 7. Cơ sở dữ liệu và Chỉ mục (Indices & Databases)
- **`indices/legal_corpus.db`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\indices\legal_corpus.db`): Cơ sở dữ liệu SQLite lưu trữ toàn bộ các chunks pháp luật kèm metadata.
- **`indices/bm25_index.pkl`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\indices\bm25_index.pkl`): File Pickle lưu trữ mô hình BM25Okapi và bảng ánh xạ tra cứu ngược (chunk_doc_map).

## 8. Nhật ký Thực thi (Logs)
- **`logs/pipeline.log`** (`c:\Users\Le Trong Hieu\Desktop\UIT-SPD UIT DSC\DSC_2026_Legal_information_retrieval_UIT\logs\pipeline.log`): Ghi lại chi tiết tiến trình thực thi tiền xử lý, index và thống kê số lượng.
