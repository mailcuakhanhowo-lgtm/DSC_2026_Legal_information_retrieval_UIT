import os
import sys
import json
import time
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# Chỉnh lại chuẩn UTF-8 để không bị lỗi tiếng Việt trên Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config_train as config

# ================= 1. BỘ NGẮT KÉP (DUAL STOPPING CRITERIA) =================
class DualStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, timeout_seconds=5.0):
        self.tokenizer = tokenizer
        self.timeout = timeout_seconds
        self.start_time = time.time()
        
    def reset(self):
        self.start_time = time.time()

    def __call__(self, input_ids, scores, **kwargs):
        # Ngắt nếu quá thời gian (chống kẹt GPU)
        if time.time() - self.start_time > self.timeout:
            return True
            
        # Ngắt nếu phát hiện ký tự đóng JSON
        last_token_str = self.tokenizer.decode(input_ids[0][-1])
        if '}' in last_token_str:
            return True
            
        return False

# ================= 2. BỘ LỌC CƠ HỌC (REGEX JSON PARSER) =================
def parse_llm_json(raw_text, original_query):
    fallback_result = {
        "original_query": original_query,
        "cau_hoi_phap_ly": original_query,
        "tu_khoa": []
    }
    
    try:
        # Trích xuất khối nằm trong ngoặc nhọn
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            return fallback_result
            
        json_str = match.group(0)
        
        # Sửa lỗi dấu phẩy thừa trước ngoặc đóng (Lỗi phổ biến của LLM)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        data = json.loads(json_str)
        
        # Đảm bảo có đủ 2 key cần thiết
        cau_hoi_phap_ly = data.get("cau_hoi_phap_ly", "").strip()
        tu_khoa = data.get("tu_khoa", [])
        
        if not cau_hoi_phap_ly:
            cau_hoi_phap_ly = original_query
        if not isinstance(tu_khoa, list):
            tu_khoa = []
            
        return {
            "original_query": original_query,
            "cau_hoi_phap_ly": cau_hoi_phap_ly,
            "tu_khoa": tu_khoa
        }
    except Exception as e:
        # Nếu lỗi (thiếu ngoặc, lồng nháy kép sai,...), lập tức dùng Fallback
        return fallback_result

# ================= 3. TRỤC CHÍNH (MAIN PIPELINE) =================
def main():
    print(f"[1] Đang tải {config.LLM_MODEL_NAME} lên VRAM...")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME, cache_dir=config.LLM_CACHE_DIR)
    
    # BƠM TEMPLATE MẶC ĐỊNH CHO VINALLAMA NẾU BỊ TÁC GIẢ QUÊN CONFIG
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=config.LLM_CACHE_DIR
    )
    
    print(f"[2] Đang đọc danh sách câu hỏi từ {config.QUESTIONS_PUBLIC_FILE}...")
    with open(config.QUESTIONS_PUBLIC_FILE, 'r', encoding='utf-8') as f:
        questions_raw = json.load(f)
        
    query_ids = list(questions_raw.keys())
    if hasattr(config, 'JUST_CHECK_QUERIES') and config.JUST_CHECK_QUERIES is not None:
        query_ids = query_ids[:config.JUST_CHECK_QUERIES]
        
    print(f"Tổng số câu hỏi cần dịch: {len(query_ids)}")
    
    # Nạp dữ liệu cũ để chạy tiếp (Resume) nếu bị ngắt ngang
    done_data = {}
    if os.path.exists(config.LLM_OUTPUT_QUERIES):
        with open(config.LLM_OUTPUT_QUERIES, 'r', encoding='utf-8') as f:
            done_data = json.load(f)
            
    stopping_criteria = DualStoppingCriteria(tokenizer, timeout_seconds=6.0)
    criteria_list = StoppingCriteriaList([stopping_criteria])
    
    print("\n[3] BẮT ĐẦU DỊCH TRUY VẤN (QUERY REWRITING)...")
    
    for i, qid in enumerate(tqdm(query_ids, desc="Dịch truy vấn")):
        if qid in done_data:
            continue
            
        original_query = questions_raw[qid].get('question', '') if isinstance(questions_raw[qid], dict) else questions_raw[qid]
        
        try:
            # 1. Ghép Prompt
            messages = [
                {"role": "system", "content": config.LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f"Phân tích câu hỏi sau:\n{original_query}"}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # MỚM MỒI TOKEN (Prefill Forcing)
            prefill_text = '{\n  "cau_hoi_phap_ly": "'
            prompt += prefill_text
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Reset đồng hồ
            stopping_criteria.reset()
            
            # 2. Sinh Text với VRAM cực nhẹ
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    stopping_criteria=criteria_list,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 3. Trích xuất văn bản
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Bơm lại đoạn mớm mồi vào kết quả để JSON parse không bị thiếu ngoặc mở
            full_response = prefill_text + generated_text
            
            # Khóa cuối nếu mô hình không thèm đóng ngoặc (bị ngắt bởi time out)
            if '}' not in full_response:
                full_response += '"\n}'
                
            # 4. Gọt dũa bằng Regex Parser
            final_data = parse_llm_json(full_response, original_query)
            done_data[qid] = final_data
            
        except Exception as e:
            # 5. Cứu hộ khẩn cấp: Fallback về nguyên bản
            print(f" Lỗi câu {qid}: {e} -> Fallback")
            done_data[qid] = {
                "original_query": original_query,
                "cau_hoi_phap_ly": original_query,
                "tu_khoa": []
            }
            
        finally:
            # Xả van VRAM chống rò rỉ sau mỗi câu
            torch.cuda.empty_cache()
            
        # 6. Auto-save
        if (i + 1) % 20 == 0:
            with open(config.LLM_OUTPUT_QUERIES, 'w', encoding='utf-8') as f:
                json.dump(done_data, f, ensure_ascii=False, indent=4)
                
    # Lưu file cuối
    with open(config.LLM_OUTPUT_QUERIES, 'w', encoding='utf-8') as f:
        json.dump(done_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ HOÀN TẤT DỊCH TRUY VẤN! File lưu tại: {config.LLM_OUTPUT_QUERIES}")

if __name__ == "__main__":
    main()
