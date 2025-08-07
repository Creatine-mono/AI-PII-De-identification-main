@echo off
echo PII Detection 프로젝트 환경을 설정합니다...

REM 현재 디렉터리를 프로젝트 루트로 설정
set BASE_DIR=%~dp0
set DATA_DIR=%BASE_DIR%data
set GEN_DIR=%BASE_DIR%generated-data
set MODEL_DIR=%BASE_DIR%models
set LLM_MODELS=%BASE_DIR%llm-models
set SAVE_DIR=%BASE_DIR%results
set WANDB_DIR=%BASE_DIR%wandb

REM 환경변수 설정
set CUDA_VISIBLE_DEVICES=0
set PYTHONHASHSEED=42
set TRANSFORMERS_NO_ADVISORY_WARNINGS=True

echo 필요한 디렉터리들을 생성합니다...

REM 디렉터리 생성
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%GEN_DIR%" mkdir "%GEN_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%LLM_MODELS%" mkdir "%LLM_MODELS%"
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"
if not exist "%WANDB_DIR%" mkdir "%WANDB_DIR%"

REM 생성된 데이터용 하위 디렉터리들
if not exist "%GEN_DIR%\placeholder" mkdir "%GEN_DIR%\placeholder"
if not exist "%GEN_DIR%\direct" mkdir "%GEN_DIR%\direct"

echo.
echo 환경 변수 설정 완료:
echo BASE_DIR: %BASE_DIR%
echo DATA_DIR: %DATA_DIR%
echo GEN_DIR: %GEN_DIR%
echo MODEL_DIR: %MODEL_DIR%
echo LLM_MODELS: %LLM_MODELS%
echo SAVE_DIR: %SAVE_DIR%
echo WANDB_DIR: %WANDB_DIR%
echo.
echo 디렉터리 생성 완료!
echo 이제 Python 스크립트들을 실행할 수 있습니다.
echo.
pause