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


PH_TOKEN = re.compile(r"^\{?([A-Za-z0-9_]{1,64})\}?$")

def _norm_ph(name: str) -> str:
    try:
        return PH_MAP_TO_CSV.get(name, name)
    except NameError:
        return name

# 기존 inject_pii_inline 함수를 아래 코드로 전체 교체하세요.
def inject_pii_inline(gen_explode, pii_row):
    """
    pii_row에 있는 실제 값으로 플레이스홀더를 교체하고 정확한 라벨을 생성합니다.
    """
    final_tokens = []
    final_ws = []
    final_labels = []

    # gen_explode는 이미 토큰화된 DataFrame이므로 한 행씩 순회합니다.
    for row in gen_explode.itertuples():
        # ⭐️ 수정된 부분: row.text -> row.tokens
        word = str(row.tokens) # 현재 토큰(단어)
        
        # 중괄호를 제거하여 순수한 플레이스홀더 이름 추출 (예: {NAME} -> NAME)
        placeholder = word.strip('{}')

        # 현재 토큰이 pii_row에 있는 플레이스홀더인지 확인
        if placeholder in pii_row.index and pd.notna(pii_row[placeholder]):
            # pii_row에서 실제 faker 데이터를 가져옴
            fake_val = str(pii_row[placeholder])
            
            # faker 데이터 자체도 공백을 포함할 수 있으므로 토큰화
            pii_tokens = fake_val.split()
            
            # 분리된 토큰이 없는 경우(빈 문자열 등)를 대비
            if not pii_tokens:
                continue

            for i, tok in enumerate(pii_tokens):
                final_tokens.append(tok)
                # 마지막 토큰이 아니라면 뒤에 공백이 있다고 가정
                final_ws.append(True if i < len(pii_tokens) - 1 else row.trailing_whitespace)
                
                # B- (Begin), I- (Inside) 라벨 동적 생성
                label = f"B-{placeholder}" if i == 0 else f"I-{placeholder}"
                final_labels.append(label)
        else:
            # PII 플레이스홀더가 아닌 일반 단어 처리
            final_tokens.append(word)
            final_ws.append(row.trailing_whitespace)
            final_labels.append("O")

    # 결과를 DataFrame으로 재구성하여 반환
    return pd.DataFrame({
        "file_name": gen_explode['file_name'].iloc[0],
        "tokens": final_tokens,
        "trailing_whitespace": final_ws,
        "label": final_labels
    })

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

# 빈 문자열 오류 방지
try:
    from src.gendata_placeholder_mistral import pii_placeholders_cleaned as _pii_clean
except Exception:
    _pii_clean = None
    
# 1) 전역에 매핑/함수 정의
PH_MAP_TO_CSV = {
    'YOUR_NAME': 'NAME',
    'IDENTIFICATION_NUM': 'ID_NUM',
}

def normalize_ph_list(ph_list):
    return [PH_MAP_TO_CSV.get(p, p) for p in (ph_list or [])]

def normalize_placeholders_in_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    for k, v in PH_MAP_TO_CSV.items():
        s = s.replace('{' + k + '}', '{' + v + '}')
    return s
def pii_placeholders_cleaned(pii_phs, text, *args, **kwargs):
    """
    안전 래퍼:
    1) NFKC 정규화, BOM/선행공백 제거
    2) 비어 있으면 즉시 ""
    3) 원본 함수가 있으면 호출하되, IndexError 등 나오면 로컬 폴백
    4) 로컬 폴백: { ... } 자리표시자 정리
    """
    # 1) 문자열 정규화
    s = "" if text is None else str(text)
    s = unicodedata.normalize("NFKC", s).lstrip("\ufeff \t\r\n")

    # 2) 비어 있으면 즉시 반환 → 원본 호출 안 함
    if not s:
        return ""

    # 3) 원본 함수가 있으면 먼저 시도 (하지만 빈 문자열은 여기까지 못 옴)
    if _pii_clean is not None:
        try:
            return _pii_clean(pii_phs, s, *args, **kwargs)
        except IndexError:
            # 원본이 여전히 text[0] 인덱싱 등으로 터지면 폴백 사용
            pass
        except Exception:
            # 다른 예외도 폴백
            pass

    # 4) 로컬 폴백 로직
    ph_set = {p.upper() for p in (pii_phs or [])}

    # 전각 중괄호 → 반각
    s = s.replace("｛", "{").replace("｝", "}")

    # {{ ... }} → { ... }
    s = re.sub(r"\{\{+\s*", "{", s)
    s = re.sub(r"\s*\}\}+", "}", s)

    # {   something messy   } → {CLEANED_NAME}
    def _repl(m):
        inner = m.group(1)
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", inner).strip("_").upper()
        if cleaned in ph_set:
            return "{" + cleaned + "}"
        return "{" + cleaned + "}" if cleaned else m.group(0)

    # 과도 매칭 방지(최대 64자)
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
    pii_placeholders = list(
    pd.Series(df['fields_used'].apply(normalize_ph_list)).explode().dropna().unique())


    # Clean up messy placeholder names between curly braces
    df['full_text'] = df.apply(
        lambda x: pii_placeholders_cleaned(pii_phs=normalize_ph_list(x.fields_used), text=x.gen_response),
        axis=1
    )
    df['full_text'] = df['full_text'].map(normalize_placeholders_in_text)



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
  # Insert PII into Full Text
    df_final = None
    # tqdm을 사용하여 진행 상황을 표시합니다.
    from tqdm.auto import tqdm

    print("Injecting PII and generating labels...")
    for ii in tqdm(range(len(df)), desc="Processing documents"):
        gen = df.iloc[[ii]].reset_index(drop=True)
        pii_row = get_pii_row(ii)
        
        # --- 새로운 PII 주입 및 라벨링 로직 시작 ---

        # 1. 원본 텍스트에 PII 데이터를 먼저 삽입하여 최종 텍스트를 완성합니다.
        text_with_pii = gen.iloc[0].full_text
        
        # pii_row에 있는 값들 중 실제로 텍스트에 존재하는 플레이스홀더만 찾습니다.
        # 긴 플레이스홀더가 짧은 플레이스홀더에 포함되는 경우를 막기 위해 길이순으로 정렬합니다.
        placeholders_in_text = {
            ph: str(val) for ph, val in pii_row.items() 
            if ph in text_with_pii and pd.notna(val) and val != ''
        }
        sorted_placeholders = sorted(placeholders_in_text.keys(), key=len, reverse=True)

        for ph in sorted_placeholders:
            fake_val = placeholders_in_text[ph]
            text_with_pii = text_with_pii.replace(ph, fake_val)

        # 2. 모든 PII가 삽입된 최종 텍스트를 토큰화합니다.
        final_tokens, final_ws = tokenize_with_spacy(text_with_pii)
        
        # 3. 토큰화된 PII 값 시퀀스를 검색하여 정확한 라벨을 부여합니다.
        final_labels = ['O'] * len(final_tokens)
        
        for ph in sorted_placeholders:
            fake_val = placeholders_in_text[ph]
            pii_tokens, _ = tokenize_with_spacy(fake_val)
            
            if not pii_tokens: continue # PII 값이 비어있으면 건너뜁니다.
            
            # final_tokens 리스트에서 pii_tokens 시퀀스를 찾습니다.
            for i in range(len(final_tokens) - len(pii_tokens) + 1):
                # 해당 위치의 토큰들이 pii_tokens와 일치하는지 확인합니다.
                if final_tokens[i:i + len(pii_tokens)] == pii_tokens:
                    # 이미 다른 라벨이 할당되지 않았는지 확인하여 중복 라벨링을 방지합니다.
                    is_unlabeled = all(l == 'O' for l in final_labels[i:i + len(pii_tokens)])
                    if is_unlabeled:
                        # B- 라벨을 첫 토큰에 할당합니다.
                        final_labels[i] = f"B-{_norm_ph(ph)}"
                        # I- 라벨을 나머지 토큰에 할당합니다.
                        for j in range(1, len(pii_tokens)):
                            final_labels[i + j] = f"I-{_norm_ph(ph)}"
        
        # --- 로직 종료 ---

        # 처리된 결과를 임시 DataFrame에 저장합니다.
        tmp = pd.DataFrame({
            'file_name': [gen.iloc[0].file_name],
            'full_text': [text_with_pii],
            'tokens': [final_tokens],
            'trailing_whitespace': [final_ws],
            'label': [final_labels]
        })
        
        # 기존 DataFrame의 메타데이터와 결합합니다.
        keep_cols = [c for c in gen.columns if c not in ['tokens', 'trailing_whitespace', 'label', 'full_text']]
        new_gen = pd.merge(gen[keep_cols], tmp, on='file_name', how='left', validate='one_to_one')
        
        # 최종 DataFrame에 누적합니다.
        if df_final is None:
            df_final = new_gen
        else:
            df_final = pd.concat([df_final, new_gen], axis=0, ignore_index=True)


    # Document ID
    df_final['document'] = [DOC_PREFIX + f'_{i}' for i in range(len(df_final))]
    
    # merge 이후 full_text 접미사 정리(있을 때만)
    if 'full_text_y' in df_final.columns:
        df_final['full_text'] = df_final['full_text_y']
        df_final.drop(columns=['full_text_y'], inplace=True)
    if 'full_text_x' in df_final.columns:
        df_final.drop(columns=['full_text_x'], inplace=True)
    
    # label → labels로 통일(이미 labels가 있으면 건너뜀)
    if 'label' in df_final.columns and 'labels' not in df_final.columns:
        df_final.rename(columns={'label': 'labels'}, inplace=True)
    
    # (선택) 안전 가드
    assert 'full_text' in df_final.columns, f"missing full_text: {list(df_final.columns)}"
    assert 'labels' in df_final.columns, f"missing labels: {list(df_final.columns)}"
    
    # View results
    if DEBUG:
        verify_df(df=df_final.copy())
    print(f'df_final.shape: {df_final.shape}')
    
    # 필요한 컬럼만 선택
    df_final = df_final[['document', 'full_text', 'tokens', 'trailing_whitespace', 'labels']]
    print(f'df_final.shape: {df_final.shape}')
    
    # Save to disk (레코드 지향 + 줄단위 + 한글 그대로)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_final.to_json(
        save_path, orient='records', lines=True, force_ascii=False
    )
    import json
    with open(save_path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            json.loads(line)
    print("JSONL OK:", save_path)

    print('End of Script - Completed')

