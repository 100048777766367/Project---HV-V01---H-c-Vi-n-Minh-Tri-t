# Học Viện Minh Triết (HV-V01) - Wisdom Model Fine-tuning

Dự án này thực hiện việc tinh chỉnh (fine-tuning) mô hình **Flan-T5** để trả lời các câu hỏi về bản thể luận (Ontology) liên quan đến các khái niệm triết học như Vô Thường, Nghiệp, và Giải Thoát. Đặc điểm nổi bật là mô hình được huấn luyện để suy luận thông qua một chuỗi tư duy (Thought Process) có cấu trúc.

## 1. Cấu trúc thư mục
- `train.py`: Script chính để xử lý dữ liệu và huấn luyện mô hình.
- `train.jsonl`: Tập dữ liệu huấn luyện (33 mẫu) chứa các nhãn thực thể.
- `validation.jsonl`: Tập dữ liệu kiểm định để đánh giá độ chính xác.
- `checkpoints/`: Lưu trữ các bước huấn luyện trung gian.
- `fine_tuned_wisdom_model_v1/`: Thư mục chứa mô hình hoàn thiện sau khi train.

## 2. Định dạng dữ liệu
Mỗi bản ghi trong tệp `.jsonl` bao gồm:
- **question**: Câu hỏi từ người dùng.
- **thought_process**: Chuỗi suy luận logic dựa trên các mã thực thể (ví dụ: [A1] -> [B5]).
- **node_trace**: Danh sách các nút tri thức được truy xuất.
- **answer**: Câu trả lời hoàn chỉnh cuối cùng.

## 3. Cài đặt & Sử dụng

### Yêu cầu hệ thống
- Python 3.8+
- GPU hỗ trợ FP16 (Khuyến nghị: GTX 1050 Ti 4GB VRAM trở lên).

### Cài đặt thư viện
```bash
pip install torch transformers datasets accelerate sentencepiece
