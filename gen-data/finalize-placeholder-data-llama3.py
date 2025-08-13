import os
import sys
from pathlib import Path
import pandas as pd
import random
import re
from typing import List
import string
import unicodedata
from kiwipiepy import Kiwi

# Add project root to Python path for package imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 모듈만 임포트하고 함수는 개별로 시도
from src.gendata_placeholder_mistral import (
    split_model_response,
    pii_total_uniques,
    token_labels,
    inject_pii,
    verify_df,
)
try:
    # 모듈에 함수가 있으면 그걸 먼저 씀
    from src.gendata_placeholder_mistral import pii_placeholders_cleaned as _pii_clean
except Exception:
    _pii_clean = None  # 없으면 로컬 fallback 사용

random.seed(42)

# 한국어 형태소 분
kiwi = Kiwi()

def tokenize_with_spacy(text: str):
    # None/NaN 방어
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    tokens = []
    trailing_ws = []
    n = len(text)
    for tok in kiwi.tokenize(text):
        start = tok.start
        end = tok.start + tok.len
        tokens.append(tok.form)  # 또는 text[start:end]
        # end가 마지막 인덱스일 수도 있으니 슬라이스로 안전 체크
        trailing_ws.append(end < n and text[end:end+1].isspace())
    return tokens, trailing_ws

# ⬇️ 모듈에 동일 함수가 없을 때를 위한 안전한 fallback 구현
def pii_placeholders_cleaned(pii_phs, text, *args, **kwargs):
    """
    중괄호 자리표시자 { ... }를 정리(clean-up)한다.
    - NFKC 정규화, BOM/선행공백 제거
    - {{ ... }} 같은 중복 중괄호 정리
    - { Phone-Num } -> {PHONE_NUM} 처럼 대소문자/문자 정규화
    - pii_phs 목록에 있는 placeholder만 확정적으로 정규화
    """
    # 모듈에 원본 함수가 있으면 그걸 우선 사용
    if _pii_clean is not None:
        return _pii_clean(pii_phs, text, *args, **kwargs)

    # 이하 로컬 fallback
    s = "" if text is None else str(text)
    s = unicodedata.normalize("NFKC", s).lstrip("\ufeff \t\r\n")
    if not s:
        return ""

    # 전각 중괄호 -> 반각
    s = s.replace("｛", "{").replace("｝", "}")

    # {{ ... }} → { ... } 정리
    s = re.sub(r"\{\{+\s*", "{", s)
    s = re.sub(r"\s*\}\}+", "}", s)

    # 허용 placeholder 집합 (대문자 기준)
    ph_set = {p.upper() for p in (pii_phs or [])}

    # {   something messy   } → {CLEANED_NAME}
    def _repl(m):
        inner = m.group(1)
        # 특수문자/공백 -> '_'로, 양끝 '_' 제거, 대문자
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", inner).strip("_").upper()
        # 목록에 있는 placeholder면 확정 정규화
        if cleaned in ph_set:
            return "{" + cleaned + "}"
        # 목록에 없으면 그래도 정규화된 형태로 교체(필요시 여기서 원문 유지로 바꿔도 됨)
        return "{" + cleaned + "}" if cleaned else m.group(0)

    # 중괄호 안의 내용을 최대 64자 정도로 제한해 과도치 매칭 방지
    s = re.sub(r"\{\s*([^{}]{1,64})\s*\}", _repl, s)
    return s


if __name__ == '__main__':
    # Inputs
    save_path = Path(os.getenv('DATA_DIR')) / 'mdd-gen/llama3_placeholder_10K_v0.jsonl'
    pii_data_path = Path(os.getenv('GEN_DIR')) / 'pii_syn_data.csv'
    SPLIT_PERCENT = 1.0
    THRESHOLD = 0.70
    DOC_PREFIX = 'llama3-syn-v0'
    DEBUG = False

    # Base dir
    path_data = Path(os.getenv('GEN_DIR'))

    # Load data
    df = pd.concat([
        pd.read_csv(path_data / 'placeholder/output.csv', encoding='UTF-8'),
    ], axis=0)

    df = df.dropna(subset=['generated_text']).reset_index(drop=True)
    if DEBUG:
        df = df.copy().iloc[0:5, :]

    # Parse LLM response from entire generated text (prompt + response)
    df['gen_response'] = df.apply(lambda x: split_model_response(x=x), axis=1)
    df['gen_response'] = df['gen_response'].fillna('').astype(str).str.strip()

    # Unique pii_placeholders
    df = df.rename(columns={'fields_used': 'fields_used_str'})
    def _split_fields(v):
        if pd.isna(v):
            return []
        return [s.strip() for s in str(v).split(',') if s.strip()]
    df['fields_used'] = df['fields_used_str'].apply(_split_fields)
    pii_placeholders = list(df['fields_used'].explode().dropna().unique())

    # Clean up messy placeholder names between curly braces
    df['full_text'] = df.apply(
        lambda x: pii_placeholders_cleaned(pii_phs=x.fields_used, text=x.gen_response),
        axis=1
    )

    # Count number of pii-placeholders inserted by LLM
    df['num_pii_fields_requested'] = df.fields_used.apply(lambda x: len(x))
    df['num_pii_fields_identified'] = df.apply(
        lambda x: pii_total_uniques(pii_phs=x.fields_used, text=x.full_text), axis=1
    )
    df['pii_ratio'] = df['num_pii_fields_identified'] / df['num_pii_fields_requested']

    # 빈 데이터 가드
    df = df[df.pii_ratio >= THRESHOLD].reset_index(drop=True)
    print(f'Num. Samples: {len(df):,}')
    if len(df) == 0:
        raise ValueError(f"No samples remain after pii_ratio >= {THRESHOLD}.")

    # file_name 생성
    if 'file_name' not in df.columns:
        df['file_name'] = [f"{DOC_PREFIX}_src_{i}" for i in range(len(df))]

    df['tokens'], df['trailing_whitespace'] = zip(
        *df['full_text'].map(tokenize_with_spacy)
    )

    # Load PII Data
    df_pii = pd.read_csv(pii_data_path)
    df_pii.rename(columns={'NAME': 'YOUR_NAME', 'ID_NUM': 'IDENTIFICATION_NUM'}, inplace=True)

    available = [c for c in pii_placeholders if c in df_pii.columns]
    missing   = [c for c in pii_placeholders if c not in df_pii.columns]
    if missing:
        print(f"[WARN] Missing PII columns: {missing}")
    if not available:
        raise ValueError("No valid PII columns found in df_pii matching placeholders.")

    pii_placeholders = available
    df_pii = df_pii[pii_placeholders].reset_index(drop=True)
    df_pii = df_pii.fillna("").astype(str)

    def get_pii_row(ii: int):
        if len(df_pii) == 0:
            raise ValueError("df_pii is empty after filtering placeholders.")
        return df_pii.iloc[ii % len(df_pii)]

    # Insert PII into Full Text
    df_final = None
    for ii in range(len(df)):
        gen, pii = df.iloc[[ii]], get_pii_row(ii)
        gen = gen.reset_index(drop=True)
        gen_explode = gen.copy().explode(['tokens', 'trailing_whitespace']).reset_index(drop=True)

        # Incorporate PII into placeholders
        gen_pii = inject_pii(row=gen_explode, pii=pii, pii_placeholders=pii_placeholders)

        # Apply competition label names
        gen_pii['label'] = gen_pii['label'].str.replace('-YOUR_NAME', '-NAME', regex=False)
        gen_pii['label'] = gen_pii['label'].str.replace('-IDENTIFICATION_NUM', '-ID_NUM', regex=False)

        # Create full text with pii filled-in
        text = []
        for t, ws in zip(gen_pii["tokens"], gen_pii["trailing_whitespace"]):
            text.append(t)
            if ws:
                text.append(" ")
        text = ''.join(text)

        # Aggregate results
        tmp = (gen_pii.groupby('file_name')
               .agg({"tokens": lambda x: x.tolist(),
                     "trailing_whitespace": lambda x: x.tolist(),
                     "label": lambda x: x.tolist()})
               .reset_index())

        # Assign new full text
        tmp['full_text'] = text

        # Combine results
        cols = list(set(gen.columns) - set(tmp.columns))
        new_gen = pd.concat([gen[cols], tmp], axis=1)

        # Concatenate into final dataframe
        if df_final is None:
            df_final = new_gen
        else:
            df_final = pd.concat([df_final, new_gen], axis=0).reset_index(drop=True)
        if ii % 50 == 0:
            print(f'Completed {ii} of {len(df):,}')

    # Document ID
    df_final['document'] = [DOC_PREFIX + f'_{i}' for i in range(len(df_final))]

    # Reduce to only required columns
    df_final.rename(columns={'label': 'labels'}, inplace=True)

    # View results
    if DEBUG:
        verify_df(df=df_final.copy())
    print(f'df_final.shape: {df_final.shape}')
    df_final = df_final[['document', 'full_text', 'tokens', 'trailing_whitespace', 'labels']]
    print(f'df_final.shape: {df_final.shape}')

    # Save to disk (레코드 지향 + 줄단위 + 한글 그대로)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_final[['document','full_text','tokens','trailing_whitespace','labels']].to_json(
        save_path, orient='records', lines=True, force_ascii=False
    )
    import json
    with open(save_path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            json.loads(line)
    print("JSONL OK:", save_path)

    print('End of Script - Completed')

