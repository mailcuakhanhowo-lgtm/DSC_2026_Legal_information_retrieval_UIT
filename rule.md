# Quy tắc Code & Chỉnh sửa (Code Editing Guidelines)

## 1. Nguyên tắc Chỉnh sửa File
- Tuyệt đối CHỈ SỬA những dòng code (hoặc block code) cần thiết cho yêu cầu hiện tại.
- KHÔNG ĐƯỢC viết đè (rewrite) lại toàn bộ file nếu không phải là yêu cầu đập đi xây lại cấu trúc, nhằm tránh làm mất các đoạn code tùy chỉnh hoặc những chỉnh sửa chưa được commit của người dùng.
- Ưu tiên sử dụng công cụ thay thế từng phần (multi_replace_file_content) để khoanh vùng chính xác điểm cần sửa.
