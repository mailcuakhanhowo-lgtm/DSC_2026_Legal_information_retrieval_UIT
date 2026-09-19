import json
import numpy as np

# ĐƯỜNG DẪN CẤU HÌNH (Sẽ gắn link thật sau khi có script search)
GROUND_TRUTH_FILE = r"D:\Project Vibe Coding\DSC_2026\train.json"
SUBMISSION_FILE = r"D:\Project Vibe Coding\DSC_2026\src\train\submission.json"

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_at_k(y_pred, y_true, k_list=[1, 3, 5, 10]):
    """
    Chấm điểm Precision và Recall tại các mức K (Top K kết quả).
    Lưu ý: BTC chỉ chấm k=5 và phạt nếu quá 5. Nhưng nội bộ ta chấm cả k=10 để xem model có tìm thấy ở rìa top 5 không.
    """
    evaluated_qids = set(y_pred.keys()).intersection(set(y_true.keys()))
    print(f"📊 Đang chấm điểm dựa trên {len(evaluated_qids)} câu hỏi có đáp án (Trùng khớp giữa Predict và Truth)\n")
    print(f"{'K':<5} | {'Recall (%)':<12} | {'Precision (%)':<15}")
    print("-" * 38)
    
    results = {}
    
    for k in k_list:
        recalls = []
        precisions = []
        
        for qid in evaluated_qids:
            true_docs = set(y_true[qid])
            # Lấy đúng k kết quả đầu tiên từ dự đoán
            pred_docs = y_pred.get(qid, [])[:k]
            pred_docs_set = set(pred_docs)
            
            # Tính giao (intersection)
            hits = len(true_docs & pred_docs_set)
            
            # Tính Recall: Tìm được bao nhiêu % trong số đáp án đúng
            if len(true_docs) > 0:
                recall = hits / len(true_docs)
                recalls.append(recall)
                
            # Tính Precision: Trong số các doc lấy ra, có bao nhiêu % là đúng
            if len(pred_docs) > 0:
                precision = hits / len(pred_docs)
            else:
                precision = 0.0
            precisions.append(precision)
            
        # Trung bình toàn tập dữ liệu
        mean_recall = np.mean(recalls) * 100
        mean_precision = np.mean(precisions) * 100
        
        results[k] = {"recall": mean_recall, "precision": mean_precision}
        print(f"{k:<5} | {mean_recall:<12.2f} | {mean_precision:<15.2f}")
        
    return results

def main():
    print(f"[HỆ THỐNG] Bắt đầu chấm điểm Dữ liệu nội bộ...\n")
    
    try:
        truth_data = read_json(GROUND_TRUTH_FILE)
        # Chỉ lấy phần "answer" của Ground Truth
        y_true = {k: v['answer'] for k, v in truth_data.items() if v.get('answer') is not None}
        print(f"✅ Đã nạp {len(y_true)} câu Ground Truth từ {GROUND_TRUTH_FILE}")
    except Exception as e:
        print(f"❌ LỖI đọc Ground Truth: {e}")
        return

    try:
        pred_data = read_json(SUBMISSION_FILE)
        # Format nộp bài bắt buộc phải có key "answer" theo luật BTC
        y_pred = {k: v.get('answer', []) for k, v in pred_data.items()}
        print(f"✅ Đã nạp {len(y_pred)} câu Dự đoán (Submission) từ {SUBMISSION_FILE}\n")
    except Exception as e:
        print(f"⚠️ Chưa tìm thấy file Submission (Sẽ chạy sau khi có thuật toán search): {e}")
        print("Tạo script chấm điểm thành công. Đợi có file submission để test!")
        return
    
    # Kiểm tra số lượng
    if len(y_pred) != len(y_true):
        print(f"⚠️ CẢNH BÁO: Số lượng câu dự đoán ({len(y_pred)}) khác với câu gốc ({len(y_true)}).")

    # Chấm điểm tại các mốc K=1, 3, 5, 10
    evaluate_at_k(y_pred, y_true, k_list=[1, 3, 5, 10])

if __name__ == "__main__":
    import codecs
    import sys
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    main()
