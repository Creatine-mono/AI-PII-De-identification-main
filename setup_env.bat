@echo off
setlocal enabledelayedexpansion
echo PII Detection 프로젝트 환경을 설정합니다...

REM === 0. 프로젝트 루트 ===
set "BASE_DIR=%~dp0"
set "DATA_DIR=%BASE_DIR%data"
set "GEN_DIR=%BASE_DIR%generated-data"
set "MODEL_DIR=%BASE_DIR%models"
set "LLM_MODELS=%BASE_DIR%llm-models"
set "SAVE_DIR=%BASE_DIR%results"
set "WANDB_DIR=%BASE_DIR%wandb"

REM === 1. 환경변수(세션 한정) ===
set CUDA_VISIBLE_DEVICES=0
set PYTHONHASHSEED=42
set TRANSFORMERS_NO_ADVISORY_WARNINGS=True
set TRANSFORMERS_NO_TORCHVISION=1

REM 선택: Hugging Face & W&B 토큰을 환경변수로 미리 넣었다면 자동 로그인
REM set HF_TOKEN=hf_xxx
REM set WANDB_API_KEY=xxx

echo 필요한 디렉터리들을 생성합니다...

REM === 2. 디렉터리 생성 ===
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%GEN_DIR%" mkdir "%GEN_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%LLM_MODELS%" mkdir "%LLM_MODELS%"
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"
if not exist "%WANDB_DIR%" mkdir "%WANDB_DIR%"
if not exist "%GEN_DIR%\placeholder" mkdir "%GEN_DIR%\placeholder"
if not exist "%GEN_DIR%\direct" mkdir "%GEN_DIR%\direct"

echo.
echo 환경 변수 설정 완료:
echo BASE_DIR: "%BASE_DIR%"
echo DATA_DIR: "%DATA_DIR%"
echo GEN_DIR: "%GEN_DIR%"
echo MODEL_DIR: "%MODEL_DIR%"
echo LLM_MODELS: "%LLM_MODELS%"
echo SAVE_DIR: "%SAVE_DIR%"
echo WANDB_DIR: "%WANDB_DIR%"
echo.
echo 디렉터리 생성 완료!
echo.

REM === 3. 가상환경 생성/활성화 ===
if not exist "%BASE_DIR%pii-env" (
    python -m venv "%BASE_DIR%pii-env"
)
call "%BASE_DIR%pii-env\Scripts\activate.bat"

REM === 4. pip 최신화 ===
python -m pip install --upgrade pip

REM === 5. CUDA 12.1용 PyTorch 설치 (필요 시 버전 조정) ===
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

REM === 6. 프로젝트 의존성 설치 ===
REM (requirements.txt에서 torch 항목은 제거해두는 게 안전)
python -m pip install -r requirements.txt

REM 추가 패키지(혹시 누락 시)
python -m pip install --upgrade transformers datasets accelerate sentencepiece
python -m pip install wandb huggingface_hub

REM === 7. 로그인(옵션: 토큰이 있으면 자동) ===
if defined HF_TOKEN (
    echo Hugging Face 토큰으로 로그인 중...
    echo %HF_TOKEN% | huggingface-cli login --token
)
if defined WANDB_API_KEY (
    echo Weights & Biases 토큰으로 로그인 중...
    wandb login %WANDB_API_KEY%
)

echo.
echo Setup complete. 가상환경 활성화 및 CUDA 지원 PyTorch 설치 완료.
echo 현재 CUDA 사용 가능 여부:
python -c "import torch; print(torch.cuda.is_available())"
echo.

echo 사용 예:
echo   set path_file=%BASE_DIR%training\train_single_large.py
echo   set path_cfg_dir=%BASE_DIR%cfgs\single-gpu
echo   python "%path_file%" --dir "%path_cfg_dir%" --name cfg0.yaml

endlocal

endlocal
