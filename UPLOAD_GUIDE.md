# Hugging Face 모델 업로드 가이드

## 1. 사전 준비

### Hugging Face Token 생성
1. https://huggingface.co/settings/tokens 접속
2. "New token" 클릭  
3. **"Write" 권한** 선택 (업로드를 위해 필수!)
4. 토큰 이름 입력 후 생성

### .env 파일 설정
```bash
# .env 파일에 다음 내용 추가/수정
HUGGINGFACE_HUB_TOKEN=your_actual_token_here
HF_USERNAME=your_huggingface_username  
HF_MODEL_NAME=pii-detection-model
UPLOAD_TO_HF=true
```

## 2. 실행 방법

### 자동 훈련 + 업로드
```bash
# 훈련과 업로드를 한번에
bash scripts/train_and_upload.sh
```

### 기존 모델만 업로드
```bash
# 이미 훈련된 모델만 업로드
python src/upload_to_hf.py --model_path ./results/your_model_directory
```

### 커스텀 모델명으로 업로드
```bash
python src/upload_to_hf.py \
    --model_path ./results/your_model_directory \
    --model_name 
```

## 3. 업로드되는 파일들

- `pytorch_model.bin` - 모델 가중치
- `config.json` - 모델 설정
- `tokenizer.json` - 토크나이저
- `tokenizer_config.json` - 토크나이저 설정
- `README.md` - 모델 카드 (자동 생성)

## 4. 업로드 후 사용법

```python
from transformers import pipeline

# 업로드된 모델 사용
pii_detector = pipeline(
    "token-classification",
    model="your_username/your_model_name",
    aggregation_strategy="simple"
)

# PII 탐지 실행
text = "안녕하세요, 제 이름은 홍길동입니다."
results = pii_detector(text)
print(results)
```

## 5. 문제 해결

### Token 권한 부족
- **Write** 권한이 있는 토큰인지 확인
- 토큰이 유효한지 확인

### 업로드 실패
- 네트워크 연결 확인
- 모델 파일이 손상되지 않았는지 확인
- Hugging Face 서비스 상태 확인

### 모델이 너무 큰 경우
- Git LFS가 자동으로 처리됨
- 대용량 파일도 업로드 가능

## 6. 보안 주의사항

- ⚠️ **토큰을 공개 저장소에 커밋하지 마세요**
- .env 파일이 .gitignore에 포함되어 있는지 확인
- 토큰이 유출되면 즉시 재발급받으세요