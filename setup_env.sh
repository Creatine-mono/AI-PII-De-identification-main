#!/bin/bash

# PII Detection 프로젝트 환경 변수 설정
# 현재 디렉터리를 프로젝트 루트로 가정

echo "PII Detection 프로젝트 환경 변수를 설정합니다..."

# 프로젝트 루트 디렉터리 (현재 디렉터리)
export BASE_DIR="$(pwd)"

# 데이터 디렉터리들
export DATA_DIR="$(pwd)/data"
export GEN_DIR="$(pwd)/generated-data"
export MODEL_DIR="$(pwd)/models"
export LLM_MODELS="$(pwd)/llm-models"
export SAVE_DIR="$(pwd)/results"

# WandB 관련 (선택사항)
export WANDB_DIR="$(pwd)/wandb"
export wandb_api_key="e5e1e1bc225ee85982c51523f92208d86953a7bd" 

# 디렉터리 생성
echo "필요한 디렉터리들을 생성합니다..."
mkdir -p "$DATA_DIR"
mkdir -p "$GEN_DIR"
mkdir -p "$GEN_DIR/placeholder" 
mkdir -p "$MODEL_DIR"
mkdir -p "$LLM_MODELS"
mkdir -p "$SAVE_DIR"
mkdir -p "$WANDB_DIR"


echo ""
echo "환경 변수 설정 완료:"
echo "BASE_DIR: $BASE_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "GEN_DIR: $GEN_DIR"
echo "MODEL_DIR: $MODEL_DIR"
echo "LLM_MODELS: $LLM_MODELS"
echo "SAVE_DIR: $SAVE_DIR"
echo "WANDB_DIR: $WANDB_DIR"
echo ""
echo "사용법:"
echo "1. 현재 세션에서만 사용: source ./setup_env.sh"
echo "2. 영구 설정: echo 'source $(pwd)/setup_env.sh' >> ~/.bashrc"
echo ""

# 1. 가상환경 생성 및 활성화 (존재하면 재활성화)
if [ ! -d "./pii-env" ]; then
  python3 -m venv pii-env
fi
source pii-env/bin/activate

# 2. pip 최신화
pip install --upgrade pip

# 2. CUDA 12.1 지원 PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# 3. 기존 프로젝트 기본 패키지
pip install transformers pandas tqdm numpy faker
pip install text_unidecode
pip install datasets
pip install spacy
pip install seqeval
# 4. 추가로 필요한 패키지
pip install sentencepiece           # DeBERTa 토크나이저에 필수
pip install accelerate              # Trainer/분산학습 최적화
pip install huggingface_hub          # HF 모델 업로드/다운로드
pip install evaluate                 # metrics 모듈
pip install tokenizers==0.19.1       # 최신 토크나이저 버전 고정
pip install bitsandbytes             # 메모리 절감(8bit/4bit)
pip install python-dotenv            # 환경변수 로딩
pip install pytorch_lightning
pip install --upgrade "transformers[torch]"

# 5. 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0

echo "Setup complete. 가상환경 활성화 및 CUDA 지원 PyTorch 설치 완료."
echo "현재 CUDA 사용 가능 여부: $(python -c 'import torch; print(torch.cuda.is_available())')"

