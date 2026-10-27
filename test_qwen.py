import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 모델 경로
MODEL_PATH = "/qwen/test1/models/Qwen3-VL-2B-Instruct"

print("모델 로딩 시작...")

# 모델 로딩
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    local_files_only=True
).eval()

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("모델 로딩 완료!")

# 테스트 질문
question = "대한민국의 수도는 어디인가요?"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": question}
        ]
    }
]

# 입력 처리
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# 추론 실행
print("질문:", question)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False
    )

# 결과 출력
generated = outputs[:, inputs["input_ids"].shape[1]:]

answer = processor.batch_decode(
    generated,
    skip_special_tokens=True
)[0]

print("답변:", answer)
print("테스트 완료!")