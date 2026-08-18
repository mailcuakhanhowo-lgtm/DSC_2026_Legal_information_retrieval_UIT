import json
import argparse

def evaluate_submission(truth_file: str, pred_file: str, top_k: int = 5):
    """
    Hàm đánh giá kết quả truy xuất so với đáp án chuẩn từ BTC.
    """
    try:
        with open(truth_file, 'r', encoding='utf-8') as f:
            truth_data = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file đáp án gốc: {truth_file}")
        return
        
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file dự đoán: {pred_file}")
        return

    total_precision = 0.0
    total_recall = 0.0
    valid_queries = 0

    for qid, truth_info in truth_data.items():
        # Bỏ qua file public-official vì ko có đáp án
        actual = truth_info.get("answer", [])
        if actual is None or len(actual) == 0:
            continue
            
        valid_queries += 1
        
        # Lấy kết quả dự đoán của hệ thống
        predicted_info = pred_data.get(qid, {})
        predicted = predicted_info.get("answer", [])
        
        # Nếu trả về nhiều hơn top_k (5) document_id, điểm = 0
        if len(predicted) > top_k:
            continue
            
        if not predicted:
            continue
            
        # Tính True Positives (Số lượng văn bản dự đoán đúng)
        tp = len(set(actual) & set(predicted))
        
        # Công thức Precision & Recall
        precision = tp / len(predicted)
        recall = tp / len(actual)
        
        total_precision += precision
        total_recall += recall

    if valid_queries == 0:
        print("Không tìm thấy câu hỏi nào có đáp án hợp lệ để đánh giá. Hãy kiểm tra lại file truth.")
        return

    # Tính điểm trung bình trên toàn bộ tập dữ liệu
    avg_precision = total_precision / valid_queries
    avg_recall = total_recall / valid_queries
    
    # Tính F1-Score (Trung bình điều hòa của Precision và Recall)
    f1_score = 0.0
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ CỤC BỘ (OFFLINE EVALUATION)")
    print("="*50)
    print(f"Tổng số câu hỏi được chấm  : {valid_queries}")
    print(f"Precision@{top_k}              : {avg_precision:.4f}")
    print(f"Recall@{top_k}                 : {avg_recall:.4f}")
    print(f"F1-Score                  : {f1_score:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chấm điểm hệ thống Retrieval")
    parser.add_argument("--truth", type=str, default="data/raw/train.json", help="Đường dẫn file JSON chứa đáp án (Ground truth)")
    parser.add_argument("--pred", type=str, default="submission.json", help="Đường dẫn file JSON do hệ thống dự đoán")
    
    args = parser.parse_args()
    
    evaluate_submission(args.truth, args.pred)