# Add project root to Python path for package imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (load_cfg,
                      debugger_is_active)
from faker import Faker  # generates fake data
import ctypes
import argparse
import random
from pathlib import Path
from tqdm.auto import tqdm
import transformers
import numpy as np
import pandas as pd
import torch
import time
import gc
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Pytorch {torch.__version__}")


# Seed the same seed to all
libc = ctypes.CDLL("libc.so.6")


def seed_everything(*, seed=42):
    Faker.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_memory():
    libc.malloc_trim(0)
    torch.cuda.empty_cache()
    gc.collect()


def load_model(model_path: str, *, quantize: bool = False):
    model_pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",)
    return model_pipeline


def generate_texts(pipeline, generated_df, path_save):

    # Generate the texts
    for i in tqdm(range(len(generated_df))):
        start = time.time()
        # Get the prompt
        prompt = generated_df.prompt.iloc[i]
        max_new_tokens = generated_df['max_new_tokens'].iloc[i]
        temperature = generated_df['temperature'].iloc[i]
        top_p = generated_df['top_p'].iloc[i]
        top_k = int(generated_df['top_k'].iloc[i])
        repeat_penalty = generated_df['repetition_penalty'].iloc[i]
        file_name = generated_df['file_name'].iloc[i]
        writing_style = generated_df['writing_style'].iloc[i]
        fields_used = generated_df['fields_used'].iloc[i]

        # Tokenize the prompt
        prompt = pipeline.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate the outputs from prompt
        outputs = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
        )
        # print(outputs[0]["generated_text"][len(prompt):])
        generated_df.loc[i, 'generated_text'] = outputs[0]["generated_text"]

        # Partial save of data
        if i % 5 == 0:
            generated_df.to_csv(path_save, index=False, encoding="UTF-8")
        print(
            f"Complete the text for {i}-th student {time.time() - start: .1f} seconds")
    # Save generated_df to csv
    generated_df.to_csv(path_save, index=False, encoding="UTF-8")
    print(f'Saved at: {path_save}')
    
def generate_texts(pipeline, generated_df, path_save, batch_size=8):
    """배치 처리로 텍스트 생성"""

    import time
    from tqdm import tqdm

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 배치 단위로 데이터 분할
    for batch_start in tqdm(range(0, len(generated_df), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(generated_df))
        batch_df = generated_df.iloc[batch_start:batch_end]

        start_time = time.time()

        # 배치 프롬프트 준비
        batch_prompts = []
        batch_indices = []

        for idx, (_, row) in enumerate(batch_df.iterrows()):
            prompt_text = pipeline.tokenizer.apply_chat_template(
                row['prompt'],
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(prompt_text)
            batch_indices.append(batch_start + idx)

        try:
            # 배치 생성
            first_row = batch_df.iloc[0]
            outputs = pipeline(
                batch_prompts,
                max_new_tokens=first_row['max_new_tokens'],
                eos_token_id=terminators,
                do_sample=True,
                temperature=first_row['temperature'],
                top_p=first_row['top_p'],
                batch_size=min(batch_size, len(batch_prompts)),
                pad_token_id=pipeline.tokenizer.eos_token_id
            )

            # 결과 저장
            for i, output in enumerate(outputs):
                original_idx = batch_indices[i]
                try:
                    if isinstance(output, list):
                        generated_text = output[0].get("generated_text") or output[0].get("text", "")
                    elif isinstance(output, dict):
                        generated_text = output.get("generated_text") or output.get("text", "")
                    else:
                        raise ValueError("Unexpected output format")
                    
                    generated_df.loc[original_idx, 'generated_text'] = generated_text
                except Exception as format_error:
                    print(f"Failed to parse output for index {original_idx}: {format_error}")
                    generated_df.loc[original_idx, 'generated_text'] = ""

        except Exception as e:
            print(f"Batch processing failed, falling back to individual processing: {e}")
            for i, (_, row) in enumerate(batch_df.iterrows()):
                try:
                    prompt_text = pipeline.tokenizer.apply_chat_template(
                        row['prompt'],
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    output = pipeline(
                        prompt_text,
                        max_new_tokens=row['max_new_tokens'],
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=row['temperature'],
                    )

                    if isinstance(output, list):
                        generated_text = output[0].get("generated_text") or output[0].get("text", "")
                    elif isinstance(output, dict):
                        generated_text = output.get("generated_text") or output.get("text", "")
                    else:
                        raise ValueError("Unexpected output format")

                    original_idx = batch_start + i
                    generated_df.loc[original_idx, 'generated_text'] = generated_text

                except Exception as individual_error:
                    print(f"Failed to generate text for index {batch_start + i}: {individual_error}")
                    generated_df.loc[batch_start + i, 'generated_text'] = ""

        # 중간 저장 (매 5배치마다)
        if (batch_start // batch_size) % 5 == 0:
            generated_df.to_csv(path_save, index=False, encoding="UTF-8")

        batch_time = time.time() - start_time
        print(f"Completed batch {batch_start // batch_size + 1} ({len(batch_df)} texts) in {batch_time:.1f} seconds")

    # 최종 저장
    generated_df.to_csv(path_save, index=False, encoding="UTF-8")
    print(f'Saved at: {path_save}')
)

label_types = ['NAME', 'EMAIL', 'USERNAME', 'ID_NUM',
               'PHONE_NUM', 'URL_PERSONAL', 'STREET_ADDRESS']

if __name__ == '__main__':

    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = os.getenv('BASE_DIR') + '/gen-data/cfgs'
        args.name = 'cfg-auto-llama3-v0.yaml'
    else:
        arg_desc = '''This program points to input parameters for model training'''
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=arg_desc)
        parser.add_argument("-cfg_dir",
                            "--dir",
                            required=True,
                            help="Base Dir. for the YAML config. file")
        parser.add_argument("-cfg_filename",
                            "--name",
                            required=True,
                            help="File name of YAML config. file")
        args = parser.parse_args()
        print(args)
    # Load the configuration file
    CFG = load_cfg(base_dir=Path(args.dir),
                   filename=args.name)
    CFG.base_dir = os.getenv('BASE_DIR')
    CFG.gen_dir = os.getenv('GEN_DIR')
    CFG.llm_dir = os.getenv('LLM_MODELS')

    # Use Hugging Face model name directly if local path doesn't exist
    local_model_path = Path(CFG.llm_dir) / CFG.model
    if local_model_path.exists():
        MODEL_PATH = str(local_model_path)
    else:
        MODEL_PATH = CFG.model  # Use HF model name directly
    print(f'MODEL_PATH: {MODEL_PATH}')

    # Seed everything
    seed_everything(seed=CFG.seed)

    # Path to save generated csv
    save_gen_filename = (f'gen_{CFG.prompt_folder}_{CFG.model}_'
                         f'N{CFG.generate_text.N}_{CFG.filename}.csv')

    # List of topics
    with open('./gen-data/prompt-templates/topics-list.txt') as f:
        topics = f.read()
    topics = topics.split('\n')

    # List of majors
    with open('./gen-data/prompt-templates/majors.txt') as f:
        majors = f.read()
    majors = majors.split('\n')

    # Generate Placeholder Text from LLM
    cols = ['IDENTIFICATION_NUM', 'STREET_ADDRESS', 'PHONE_NUM',
            'USERNAME', 'URL_PERSONAL', 'EMAIL', 'DATE_OF_BIRTH', 'AGE',
            'CREDIT_CARD_INFO', 'BANKING_NUMBER', 'ORGANIZATION_NAME',
            'NATIONALITY', 'DATE', 'GENDER', 'MEDICAL_CONDITION',
            'PASSWORD', 'SECURE_CREDENTIAL']

    writing_style = [
        '업무 보고서',
        '프로젝트 계획서',
        '회의록',
        '제안서',
        '분석 리포트',
        '업무 메모',
        '이메일',
        '업무 질문',
        '내부 문서'
    ]
    fields_used = []
    writing_styles = []
    for _ in range(CFG.generate_text.N):
        fields_to_use = random.sample(cols, random.randint(2, 4))  # 2-4개 필드 사용
        random.shuffle(fields_to_use)
        fields_used.append(", ".join(['YOUR_NAME'] + fields_to_use))
        writing_styles.append(random.choice(writing_style))

    # Store in dataframe
    df = pd.DataFrame({'fields_used': fields_used,
                      'writing_style': writing_styles})
    del fields_to_use, fields_used, writing_styles

    # Generate model parameter settings
    df['max_new_tokens'] = [random.choice([2048]) for _ in range(len(df))]
    df['temperature'] = [random.choice(
        [10, 20, 30, 70]) / 100 for _ in range(len(df))]
    df['top_p'] = [random.randint(a=90, b=95) / 100 for _ in range(len(df))]
    df['top_k'] = [random.choice([40, 50]) for _ in range(len(df))]
    df['repetition_penalty'] = [random.choice(
        [1.1, 1.2]) for _ in range(len(df))]

    # Generate occupation
    df['occupation'] = [random.choice(majors).lower() for _ in range(len(df))]
    df['topic'] = [random.choice(topics).lower() for _ in range(len(df))]

    # Prompt fields to insert
    def prompt_placeholder(fields):
        fields = fields.split(', ')
        return '\n'.join(['{' + f'{field}' + '}' for field in fields])

    df['prompt_pii'] = df.apply(lambda x: prompt_placeholder(fields=x['fields_used']),
                                axis=1)

    # List of prompts
    prompt_files = {
        'mixed': (list(Path(f'./gen-data/prompt-templates/placeholder/mixed-llama3').glob('*.txt'))),
    }

    def create_prompt(files: dict, data: pd.Series):
        if random.random() >= 0.0:
            file = random.sample(files['mixed'], 1)[0]
        else:
            file = random.sample(files['names'], 1)[0]
        with open(file) as f:
            prompt = f.read()
        prompt = prompt.replace('{OCCUPATION}', data['occupation'])
        prompt = prompt.replace('{REPORT}', data['writing_style'])
        prompt = prompt.replace('{TOPIC}', data['topic'])

        system_prompt = prompt.split('%%%%%%%%%%%%%%%%%%%%%%%%%')[0].strip()
        user_prompt = prompt.split('%%%%%%%%%%%%%%%%%%%%%%%%%')[1].strip()

        prompt_defs = {
            'YOUR_NAME': "성명",
            'IDENTIFICATION_NUM': "사번/직원번호",
            'STREET_ADDRESS': "주소",
            'PHONE_NUM': "연락처",
            'USERNAME': "사용자명/계정",
            'URL_PERSONAL': "개인 웹사이트/SNS",
            'EMAIL': "이메일 주소",
            'DATE_OF_BIRTH': "생년월일",
            'AGE': "나이",
            'CREDIT_CARD_INFO': "신용카드 정보",
            'BANKING_NUMBER': "계좌번호",
            'ORGANIZATION_NAME': "소속 회사/기관명",
            'NATIONALITY': "국적",
            'DATE': "날짜 정보",
            'GENDER': "성별",
            'MEDICAL_CONDITION': "건강 상태",
            'PASSWORD': "비밀번호",
            'SECURE_CREDENTIAL': "보안 인증정보"}

        sys_pii = []
        for pii in data.prompt_pii.split('\n'):
            sys_pii.append(f'{pii}: {prompt_defs[pii[1:-1]]}')
        sys_pii = '\n'.join(sys_pii)

        system_prompt = system_prompt.replace('{INSERT_INFO_HERE}', sys_pii)
        user_prompt = user_prompt.replace(
            '{INSERT_INFO_HERE}', data['prompt_pii'])

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return file.name, prompt

    # Create prompt for the model
    df['file_name'], df['prompt'] = zip(*df.apply(lambda x: create_prompt(files=prompt_files,
                                                                          data=x), axis=1))

    # Model used to gen. text
    df['model'] = CFG.model
    
# 저장 파일 경로 지정 (파일명 포함)
save_gen_filename = "gen_placeholder_Bllossom/output.csv"  # 필요에 따라 파일명 변경 가능

# 전체 저장 경로 구성
full_save_path = Path(CFG.gen_dir) / 'placeholder' / save_gen_filename

# 필요한 상위 디렉토리까지 전부 생성
full_save_path.parent.mkdir(parents=True, exist_ok=True)

# 모델 로딩 (중복 제거)
model = load_model(model_path=MODEL_PATH,  quantize=True)

# 텍스트 생성 및 저장
generate_texts(pipeline=model,
               generated_df=df,
               path_save=str(full_save_path))
