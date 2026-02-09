# train.py

# ==============================================================================
# 1. THIẾT LẬP MÔI TRƯỜNG & THƯ VIỆN
# ==============================================================================
import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

print("Đã tải xong các thư viện cần thiết.")

# ==============================================================================
# 2. ĐỊNH NGHĨA CÁC BIẾN CỐ ĐỊNH
# ==============================================================================
TRAIN_FILE = "train.jsonl"
VALID_FILE = "validation.jsonl"
MODEL_NAME = "google/flan-t5-small"  # Nâng cấp lên 'base' để có hiệu suất tốt hơn một chút
OUTPUT_DIR = "fine_tuned_wisdom_model_v1"
CHECKPOINT_DIR = "checkpoints"

# ==============================================================================
# 3. HÀM TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
def load_and_prepare_dataset(file_path, tokenizer, max_input_length=512, max_target_length=256):
    """Đọc tệp .jsonl, định dạng prompt và tokenize."""
    
    # Đọc dữ liệu từ file .jsonl
    records = [json.loads(line) for line in Path(file_path).read_text(encoding='utf-8').splitlines()]
    dataset = Dataset.from_list(records)
    
    def preprocess_function(examples):
        # Thiết kế prompt có cấu trúc để hướng dẫn AI
        prompts = [
            f"Dựa trên chuỗi suy luận sau:\n{tp}\n\nHãy trả lời câu hỏi:\n{q}\n\nTrả lời:"
            for q, tp in zip(examples["question"], examples["thought_process"])
        ]
        
        # Tokenize input và output
        model_inputs = tokenizer(prompts, max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=examples["answer"], max_length=max_target_length, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_dataset = dataset.map(preprocess_function, batched=True)
    print(f"Đã xử lý xong {len(processed_dataset)} mẫu từ tệp {file_path}")
    return processed_dataset

# ==============================================================================
# 4. KHỞI TẠO MODEL VÀ TOKENIZER
# ==============================================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"Đã khởi tạo xong model '{MODEL_NAME}' và tokenizer.")

# ==============================================================================
# 5. CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN
# ==============================================================================
train_dataset = load_and_prepare_dataset(TRAIN_FILE, tokenizer)
validation_dataset = load_and_prepare_dataset(VALID_FILE, tokenizer)

# Data Collator để xử lý padding động, giúp tiết kiệm bộ nhớ
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ==============================================================================
# 6. CẤU HÌNH QUÁ TRÌNH HUẤN LUYỆN
# ==============================================================================
training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=10,              # Tăng epoch vì dữ liệu ít (35 mẫu)
    learning_rate=3e-5,
    per_device_train_batch_size=1,    # BẮT BUỘC để mức 1 để tránh OOM
    per_device_eval_batch_size=1,     # BẮT BUỘC để mức 1
    gradient_accumulation_steps=8,    # Tích lũy 8 bước để có batch size thực tế = 8
    weight_decay=0.01,
    save_total_limit=1,               # Chỉ giữ 1 file checkpoint để tiết kiệm ổ cứng
    predict_with_generate=True,
    fp16=True,                        # BẮT BUỘC (GTX 1050 Ti hỗ trợ tốt)
    optim="adamw_torch",              # Tiêu chuẩn
    gradient_checkpointing=True,      # THÊM: Tiết kiệm cực nhiều VRAM khi train
    load_best_model_at_end=True,
    report_to="none"
)

# ==============================================================================
# 7. KHỞI TẠO VÀ CHẠY TRAINER
# ==============================================================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator
)

print("\nBẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN (Ngày T+1 -> T+3)...")
trainer.train()
print("...QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT.")

# ==============================================================================
# 8. LƯU LẠI KẾT QUẢ TỐT NHẤT
# =Lưu mô hình và tokenizer tốt nhất vào thư mục cuối cùng
# ==============================================================================
print(f"\nLưu lại mô hình tốt nhất vào thư mục: '{OUTPUT_DIR}'")
trainer.save_model(OUTPUT_DIR)
print("...HOÀN TẤT LƯU TRỮ.")

print("\nNGÔI ĐỀN ĐÃ ĐƯỢC DỰNG XÂY. MÔ HÌNH TRI THỨC ĐÃ SẴN SÀNG.")