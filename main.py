from transformers import AutoTokenizer, AutoModelForTokenClassification
from huggingface_hub import HfApi, create_repo

# ===== 0. 사전 설정 =====
# 먼저 터미널에서:
# huggingface-cli login

# 모델 경로 (학습이 끝난 폴더)
model_path = "results/roberta-large-250812-0613/checkpoint-750"

# Hugging Face 계정 & 리포 이름
username = "psh3333"           # HF 계정 아이디
repo_name = "roberta-large-korean-pii2"  # 만들 리포 이름
repo_id = f"{username}/{repo_name}"

# ===== 1. HF API 객체 생성 & 리포 만들기 =====
api = HfApi()
create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

# ===== 2. 모델 & 토크나이저 로드 =====
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# ===== 3. 허브로 업로드 =====
commit_msg = "Upload Korean PII detection model"
model.push_to_hub(repo_id, commit_message=commit_msg)
tokenizer.push_to_hub(repo_id, commit_message=commit_msg)

print(f"[✓] 업로드 완료: https://huggingface.co/{repo_id}")
