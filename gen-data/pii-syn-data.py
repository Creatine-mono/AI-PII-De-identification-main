# Add project root to Python path for package imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.gendata import (tokenize_with_spacy,
                        assign_labels,
                        create_token_map,
                        verify_df)
from src.utils import (load_cfg,
                      debugger_is_active)
from src.load_data import LoadData
import src.create_datasets as create_datasets
import unicodedata
from typing import List
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from spacy.lang.en import English
from pathlib import Path
import random
import argparse
import sys
import ctypes
from faker import Faker  # generates fake data
import gc
import string
import re
import time
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Custom (cx) modules

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Pytorch {torch.__version__}")

# Ensure that all operations are deterministic on GPU (if used) for
# reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Seed the same seed to all
libc = ctypes.CDLL("libc.so.6")


def seed_everything(*, seed=42):
    Faker.seed(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_memory():
    libc.malloc_trim(0)
    torch.cuda.empty_cache()
    gc.collect()


def generate_korean_text_phone_number():
    """
    '공일공-일이삼사-오육칠팔' 형식의 한글 전화번호를 생성합니다.
    """
    num_to_korean = {'0': '공', '1': '일', '2': '이', '3': '삼', '4': '사', '5': '오', '6': '육', '7': '칠', '8': '팔', '9': '구'}

    p1 = "010"
    p2 = f"{random.randint(0, 9999):04d}"
    p3 = f"{random.randint(0, 9999):04d}"

    korean_p1 = "".join([num_to_korean[n] for n in p1])
    korean_p2 = "".join([num_to_korean[n] for n in p2])
    korean_p3 = "".join([num_to_korean[n] for n in p3])

    return f"{korean_p1}-{korean_p2}-{korean_p3}"


def generate_noisy_phone_number():
    """
    '공l0-ㅣz34-00oㅇ' 형식의 한글/영어/숫자 혼합 전화번호를 생성합니다.
    """
    noisy_map = {
        '0': ['0', '공', 'o', 'O', 'ㅇ'],
        '1': ['1', '일', 'l', 'I', 'ㅣ'],
        '2': ['2', '이', 'z', 'Z'],
        '3': ['3', '삼', 'ㅅ', 'E'],
        '4': ['4', '사', 'A'],
        '5': ['5', '오', 'o', 'O', 's', 'S'],
        '6': ['6', '육', 'b'],
        '7': ['7', '칠', 'ㅊ'],
        '8': ['8', '팔', 'ㅍ', 'B'],
        '9': ['9', '구', 'ㄱ', 'g', 'q', 'p']
    }

    # 기본 숫자 생성
    if random.random() >= 0.1:
        # 90% 휴대폰 번호
        prefixes = ['010', '011', '016', '017', '018', '019']
        prefix = random.choice(prefixes)
        middle = f"{random.randint(0, 9999):04d}"
        last = f"{random.randint(0, 9999):04d}"
    else:
        # 10% 지역번호
        area_codes = ['02', '031', '032', '033', '041', '042', '043', '044', '051', '052', '053', '054', '055', '061',
                      '062', '063', '064']
        prefix = random.choice(area_codes)
        middle = f"{random.randint(100, 999):03d}"
        last = f"{random.randint(1000, 9999):04d}"

    # 변조된 문자로 변환
    noisy_p1 = "".join([random.choice(noisy_map[d]) for d in prefix])
    noisy_p2 = "".join([random.choice(noisy_map[d]) for d in middle])
    noisy_p3 = "".join([random.choice(noisy_map[d]) for d in last])

    # 형식 랜덤 선택 (하이픈 있음/없음)
    formats = [
        f"{noisy_p1}-{noisy_p2}-{noisy_p3}",
        f"{noisy_p1}{noisy_p2}{noisy_p3}",
        f"{noisy_p1} {noisy_p2} {noisy_p3}",
    ]
    return random.choice(formats)


def generate_date_of_birth():
    """Generate Korean date of birth in various formats for all age groups"""
    # 1950-2025년 사이 생성 (모든 연령대 포함)
    year = random.randint(1950, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # 간단히 28일까지만

    formats = [
        f"{year}-{month:02d}-{day:02d}",
        f"{year}.{month:02d}.{day:02d}",
        f"{year}년 {month}월 {day}일",
        f"{year % 100:02d}{month:02d}{day:02d}",  # YYMMDD 형식
        f"{year}/{month:02d}/{day:02d}",
    ]
    return random.choice(formats)


def generate_age():
    """Generate age in Korean format for all age groups"""
    age = random.randint(0, 99)  # 모든 연령대 포함

    korean_numbers = {
        1: '한', 2: '두', 3: '세', 4: '네', 5: '다섯', 6: '여섯', 7: '일곱', 8: '여덟', 9: '아홉', 10: '열',
        11: '열한', 12: '열두', 13: '열세', 14: '열네', 15: '열다섯', 16: '열여섯', 17: '열일곱', 18: '열여덟', 19: '열아홉', 20: '스무',
        21: '스물한', 22: '스물두', 23: '스물세', 24: '스물네', 25: '스물다섯', 26: '스물여섯', 27: '스물일곱', 28: '스물여덟', 29: '스물아홉',
        30: '서른',
        31: '서른한', 32: '서른두', 33: '서른세', 34: '서른네', 35: '서른다섯', 36: '서른여섯',
        37: '서른일곱', 38: '서른여덟', 39: '서른아홉', 40: '마흔', 41: '마흔한', 42: '마흔두',
        43: '마흔세', 44: '마흔네', 45: '마흔다섯', 46: '마흔여섯', 47: '마흔일곱', 48: '마흔여덟',
        49: '마흔아홉', 50: '쉰', 60: '예순', 70: '일흔', 80: '여든', 90: '아흔'
    }

    formats = []

    # 기본 형태
    formats.extend([f"{age}세", f"{age}살"])

    # 한글 표현 (50세 이하 또는 10의 배수)
    if age in korean_numbers:
        formats.append(f"{korean_numbers[age]}살")
    elif age < 50:
        formats.append(f"{age}살")
    elif age % 10 == 0 and age <= 90:
        formats.append(f"{korean_numbers[age]}살")

    # 연령대 표현 (10세 이상만)
    if age >= 10:
        decade = age // 10
        if decade == 1:
            formats.extend([f"10대", f"십대"])
        elif decade <= 9:
            formats.extend([
                f"{decade}0대",
                f"{decade}0대 {'초반' if age % 10 <= 3 else '중반' if age % 10 <= 6 else '후반'}"
            ])

    # 유아/어린이/청소년 표현
    if age <= 7:
        formats.extend([f"{age}살 유아", f"{age}세 어린이"])
    elif age <= 12:
        formats.append(f"{age}세 어린이")
    elif age <= 19:
        formats.extend([f"{age}세 청소년", f"10대"])

    return random.choice(formats)


def generate_credit_card_info():
    """Generate Korean credit card information"""
    # 한국 주요 카드사 BIN
    card_bins = {
        '신한카드': ['4567', '5123', '4356'],
        'KB국민카드': ['4789', '5234', '4234'],
        '삼성카드': ['4123', '5345', '4445'],
        '현대카드': ['4567', '5456', '4678'],
        '롯데카드': ['4890', '5567', '4789'],
        'BC카드': ['4321', '5678', '4890'],
        '하나카드': ['4456', '5789', '4567'],
        '우리카드': ['4678', '5890', '4123']
    }

    card_company = random.choice(list(card_bins.keys()))
    bin_code = random.choice(card_bins[card_company])

    # 나머지 12자리 생성
    remaining = ''.join([str(random.randint(0, 9)) for _ in range(12)])
    card_number = bin_code + remaining

    formats = [
        f"{card_number[:4]}-{card_number[4:8]}-{card_number[8:12]}-{card_number[12:]}",
        f"{card_number[:4]} {card_number[4:8]} {card_number[8:12]} {card_number[12:]}",
        card_number,
        f"{card_company} {card_number[:4]}-****-****-{card_number[12:]}"
    ]
    return random.choice(formats)


def generate_banking_number():
    """Generate Korean banking account numbers"""
    banks = {
        '국민은행': ['123', '456', '789'],
        '신한은행': ['234', '567', '890'],
        '우리은행': ['1002', '1005', '1008'],
        'KB국민은행': ['456', '789', '012'],
        '하나은행': ['333', '444', '555'],
        '농협은행': ['301', '302', '303'],
        '기업은행': ['001', '002', '003'],
        '대구은행': ['504', '505', '506']
    }

    bank = random.choice(list(banks.keys()))
    prefix = random.choice(banks[bank])

    if bank in ['우리은행', '신한은행']:
        # 우리은행, 신한은행: XXXX-XXX-XXXXXX
        middle = str(random.randint(100, 999))
        suffix = str(random.randint(100000, 999999))
        account = f"{prefix}-{middle}-{suffix}"
    else:
        # 기타 은행: XXX-XXXXXX-XX-XXX
        middle = str(random.randint(100000, 999999))
        suffix1 = str(random.randint(10, 99))
        suffix2 = str(random.randint(100, 999))
        account = f"{prefix}-{middle}-{suffix1}-{suffix2}"

    formats = [
        account,
        account.replace('-', ''),
        f"{bank} {account}",
        account.replace('-', ' ')
    ]
    return random.choice(formats)


def generate_organization_name():
    """Generate Korean organization names"""
    universities = [
        '서울대학교', '연세대학교', '고려대학교', '성균관대학교', '한양대학교', '중앙대학교',
        '경희대학교', '한국외국어대학교', '서강대학교', '이화여자대학교', '홍익대학교',
        '건국대학교', '동국대학교', '국민대학교', '숭실대학교', '세종대학교'
    ]

    companies = [
        '삼성전자', 'LG전자', 'SK텔레콤', '현대자동차', '기아자동차', '포스코',
        'KB금융그룹', '신한은행', '우리은행', '하나금융그룹', '롯데그룹', 'GS그룹',
        '두산그룹', 'LS그룹', 'CJ그룹', '한화그룹', '대우조선해양', '현대중공업'
    ]

    organizations = universities + companies
    return random.choice(organizations)


def generate_date():
    """Generate general dates in various formats"""
    year = random.randint(2020, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)

    formats = [
        f"{year}-{month:02d}-{day:02d}",
        f"{year}.{month:02d}.{day:02d}",
        f"{year}년 {month}월 {day}일",
        f"{month}/{day}/{year}",
        f"{day}.{month}.{year}",
    ]
    return random.choice(formats)


def generate_password():
    """Generate various password formats"""
    # 일반적인 패스워드 패턴들
    patterns = [
        lambda: ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(8, 12))),
        lambda: f"{''.join(random.choices(string.ascii_letters, k=random.randint(4, 6)))}{random.randint(1000, 9999)}",
        lambda: f"{''.join(random.choices(string.ascii_letters, k=random.randint(3, 5)))}{''.join(random.choices('!@#$%', k=1))}{random.randint(100, 999)}",
        lambda: f"password{random.randint(123, 999)}",
        lambda: f"{''.join(random.choices(['qwer', 'asdf', 'zxcv'], k=1))}{random.randint(1234, 9999)}",
    ]

    return random.choice(patterns)()


def generate_secure_credential():
    """Generate secure credential information"""
    creds = [
        f"API-KEY-{random.randint(10000, 99999)}",
        f"{''.join(random.choices(string.ascii_uppercase + string.digits, k=32))}",  # API Key
        f"sk-{''.join(random.choices(string.ascii_letters + string.digits, k=48))}",  # OpenAI style
        f"Bearer {''.join(random.choices(string.ascii_letters + string.digits, k=40))}",  # Bearer token
        f"JWT-{''.join(random.choices(string.ascii_letters + string.digits, k=50))}",
    ]
    return random.choice(creds)


def generate_korean_phone_number():
    """Generate Korean phone number format with various styles including corrupted versions"""
    phone_type = random.choice(['normal', 'korean_text', 'noisy'])

    if phone_type == 'korean_text':
        return generate_korean_text_phone_number()
    elif phone_type == 'noisy':
        return generate_noisy_phone_number()
    else:
        # 일반 한국 전화번호 (기존 로직)
        if random.random() >= 0.1:
            # 90% 휴대폰 번호
            prefixes = ['010', '011', '016', '017', '018', '019']
            prefix = random.choice(prefixes)
            middle = str(random.randint(1000, 9999))
            last = str(random.randint(1000, 9999))

            # 형식 랜덤 선택 (하이픈 있음/없음/공백)
            formats = [
                f"{prefix}-{middle}-{last}",
                f"{prefix}{middle}{last}",
                f"{prefix} {middle} {last}",
            ]
            return random.choice(formats)
        else:
            # 10% 일반 전화번호 (지역번호)
            area_codes = ['02', '031', '032', '033', '041', '042', '043', '044', '051', '052', '053', '054', '055',
                          '061', '062', '063', '064']
            area_code = random.choice(area_codes)
            middle = str(random.randint(100, 999))
            last = str(random.randint(1000, 9999))

            formats = [
                f"{area_code}-{middle}-{last}",
                f"{area_code}{middle}{last}",
                f"{area_code} {middle} {last}",
            ]
            return random.choice(formats)


def generate_street_address(fake):
    """Generate Korean street address using faker with ko_KR locale"""
    # 기본적으로 ko_KR 로케일의 address 사용
    if random.random() >= 0.2:
        sa = str(fake.address()).replace("\n", random.choice([", ", " "]))
    else:
        # 10% 확률로 다양한 조합 시도
        sa = {0: fake.street_name() + ', ' + fake.city(),
              1: fake.street_address(),
              2: fake.street_address(),
              3: fake.street_address(),
              4: fake.street_address()}
        sa = sa[random.choice([0, 1, 2, 3, 4])]
    return sa


# Random generate 12 random number
def get_userid(length=16):
    """Generate userid - """
    if random.random() >= 0.30:
        # very common in training data 034626995785
        userid = str(random.randint(10 ** 11, 10 ** 12 - 1))
    else:
        if random.random() >= 0.5:
            # DM:705244534902
            userid = ("".join(random.choices(string.ascii_uppercase, k=2)) +
                      ':' + str(random.randint(10 ** 11, 10 ** 12 - 1)))
        else:
            if random.random() >= 0.25:
                # nMFtUVxSUI|33529258
                userid = ("".join(random.choices(string.ascii_letters, k=10)) +
                          '|' + str(random.randint(10 ** 8, 10 ** 9 - 1)))
            else:
                # ras21 or 51,00,23,0
                userid = ("".join(random.choices(string.ascii_letters, k=random.randint(
                    3, 5))) + str(random.randint(10 ** 4, 10 ** 6 - 1)))
    # Split id_num
    if random.random() >= 1.0:
        sep = random.choice([' ', '-'])
        n = int(len(userid) / random.choice([2, 3]))
        userid = sep.join([userid[i:i + n] for i in range(0, len(userid), n)])

    return userid


# Korean name combination logic
def combine_first_last(fn: str, ln: str, algo_num: int):

    # 한국어 이름 처리 - 공백 제거 및 안전성 체크
    fn = fn.strip() if fn else ""
    ln = ln.strip() if ln else ""
    
    # 빈 이름 처리
    if not fn or not ln:
        return fn, ln
    

    if algo_num == 0:
        # 이름 첫 글자 + 성
        fn = fn[0] if len(fn) > 0 else fn
        ln = ln
    elif algo_num == 1:
        # 이름 첫 글자 + 성 첫 글자 (한국어용)
        fn_initial = fn[0] if len(fn) > 0 else ""
        ln_initial = ln[0] if len(ln) > 0 else ""
        fn = fn_initial + ln_initial
        ln = ln
    return fn, ln


def social_media(username, prob):
    social_media_platforms = {
        'LinkedIn': 'linkedin.com/',
        'YouTube': 'youtube.com/',
        'Instagram': 'instagram.com/',
        'GitHub': 'github.com/',
        'Facebook': 'facebook.com/',
        'Twitter': 'twitter.com/',
        'KakaoStory': 'story.kakao.com/',
        'Band': 'band.us/',
    }

    if random.random() >= prob:
        platform, domain = random.choice(
            list(social_media_platforms.items())[0:2])
    else:
        platform, domain = random.choice(
            list(social_media_platforms.items())[2:])

    if platform == 'YouTube':
        post = {
            0: f'channel/UC{"".join(random.choices(string.ascii_letters + string.digits, k=random.randint(12, 14)))}',
            1: f'channel/watch?v={"".join(random.choices(string.ascii_letters + string.digits, k=random.randint(10, 12)))}',
            2: f'channel/user/{username}',
            3: f'c/{username}',
        }
        fake_url = f'https://www.{domain}{post[random.randint(0, 3)]}'
    elif platform == 'LinkedIn':
        post = {0: f'in/{username}', 1: f'{username}'}
        if random.random() >= 0.50:
            fake_url = f'https://www.{domain}{post[0]}'
        else:
            fake_url = f'https://www.{domain}{post[1]}'

    else:
        fake_url = f'https://{domain}{username}'
    return fake_url


def personal_site(first_name, last_name, username):
    print()
    fake = Faker()
    uri_path = fake.uri_path()
    tld = fake.tld()
    uri_ext = fake.uri_extension()
    domain_word = {0: f'{first_name}-{last_name}',
                   1: f'{first_name}',
                   2: f'{last_name}',
                   3: f'{first_name}{last_name}',
                   4: f'{username}'}
    www = random.choice(['', 'www.'])
    fake_url = f'https://{www}{domain_word[random.randint(0, 4)]}.{tld}/{uri_path}{uri_ext}'
    return fake_url.replace(' ', '').lower()


# Generate the personal url from social media
def generate_fake_social_media_url(first_name, last_name, algo):
    if random.random() >= 0.50:
        first_name, last_name, _ = get_name()

    username = generate_username(first_name, last_name, algo, 0.95)

    if random.random() >= 0.5:
        fake_url = social_media(username, 0.30)
    else:
        fake_url = personal_site(first_name, last_name, username)
    return fake_url


def generate_username(first_name, last_name, algo, prob):
    """Korean-optimized username generation"""

    if random.random() >= 0.50:
        first_name, last_name, _ = get_name()

    # 안전성 체크
    first_name = first_name.strip() if first_name else ""
    last_name = last_name.strip() if last_name else ""
    
    if not first_name or not last_name:
        return f"user{random.randint(1000, 9999)}"

    if algo is not None:
        first_name, last_name = combine_first_last(
            fn=first_name, ln=last_name, algo_num=algo)

    if random.random() >= prob:
        username = f"{first_name.lower()}{last_name.lower()}{random.randint(1, 999)}"
    else:
        username = f"{first_name}{last_name}"

    # 공백 제거 및 소문자 변환
    username = username.replace(' ', '').lower()
    return username


def generate_email(first_name, last_name, faker, algo):
    """Generate email using Faker for simplicity and reliability"""
    # 80% Faker 생성, 20% 커스텀 도메인 사용
    if random.random() >= 0.2:
        # Faker로 완전한 이메일 생성
        return faker.email()
    else:
        # 커스텀 도메인 사용
        local_part = faker.user_name()
        domain_name = random.choice(EMAIL_DOMAINS)
        return f"{local_part}@{domain_name}"


def get_name():
    """
    Faker 라이브러리를 사용해 한국어 성(last_name)과 이름(first_name)을 생성합니다.
    """
    # 'ko_KR' 로케일을 사용하여 한국어 이름 생성기를 만듭니다.
    faker = Faker("ko_KR")

    # 성과 이름을 각각 생성합니다.
    first_name = faker.first_name()
    last_name = faker.last_name()

    # 이름에 포함될 수 있는 하이픈(-)을 공백으로 바꿉니다. (이 부분은 필요에 따라 유지)
    first_name = first_name.replace('-', ' ')
    last_name = last_name.replace('-', ' ')

    # 생성된 이름, 성, faker 객체를 반환합니다.
    return first_name, last_name, faker


def generate_student_info():
    """Generates all the user info (name, eamil addresses, phone number, etc) together """
    first_name, last_name, faker = get_name()

    real = random.choice([True])
    # Select algorithm for combining first and last names
    algos = []
    for _ in range(3):
        if random.random() >= 0.25:
            algos.append(random.choices([0, 1], k=1)[0])
        else:
            algos.append(None)

    user_name = generate_username(first_name, last_name, algos[0], 0.80)
    fake_url = generate_fake_social_media_url(first_name, last_name, algos[1])
    fake_email = generate_email(first_name, last_name, faker, algos[2])
    street_address = generate_street_address(fake=faker)

    student = {}
    # 기존 PII 항목들
    student['ID_NUM'] = get_userid()  # User ID
    student['NAME'] = last_name + first_name
    student['EMAIL'] = fake_email
    student['USERNAME'] = user_name
    student['PHONE_NUM'] = generate_korean_phone_number()
    student['URL_PERSONAL'] = fake_url
    student['STREET_ADDRESS'] = street_address

    # 새로 추가된 PII 항목들
    student['DATE_OF_BIRTH'] = generate_date_of_birth()
    student['AGE'] = generate_age()
    student['CREDIT_CARD_INFO'] = generate_credit_card_info()
    student['BANKING_NUMBER'] = generate_banking_number()

    # 7:3 비율로 커스텀 함수 vs Faker 기본 함수 사용
    if random.random() >= 0.3:
        student['ORGANIZATION_NAME'] = faker.company()  # 70% 커스텀
    else:
        student['ORGANIZATION_NAME'] = generate_organization_name()  # 30% Faker

    if random.random() >= 0.3:
        student['DATE'] = generate_date()  # 70% 커스텀
    else:
        student['DATE'] = faker.date()  # 30% Faker

    student['PASSWORD'] = generate_password()
    student['SECURE_CREDENTIAL'] = generate_secure_credential()

    del faker
    clear_memory()
    #     print(student)
    return student


label_types = ['NAME', 'EMAIL', 'USERNAME', 'ID_NUM', 'PHONE_NUM',
               'URL_PERSONAL', 'STREET_ADDRESS', 'DATE_OF_BIRTH', 'AGE',
               'CREDIT_CARD_INFO', 'BANKING_NUMBER', 'ORGANIZATION_NAME',
               'DATE', 'PASSWORD', 'SECURE_CREDENTIAL']

if __name__ == '__main__':
    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = os.getenv('BASE_DIR') + '/gen-data/cfgs'
        args.name = 'cfg1.yaml'
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

    MODEL_PATH = str(Path(CFG.llm_dir) / CFG.model)
    print(f'MODEL_PATH: {MODEL_PATH}')

    # Seed everything
    seed_everything(seed=CFG.seed)

    # Training data
    # df_train = pd.read_json(
    #     Path(
    #         CFG.gen_dir) /
    #     'pii-detection-removal-from-educational-data/train.json')
    # df_train = df_train.explode(
    #     ['tokens', 'trailing_whitespace', 'labels']).reset_index(drop=True)




# Load top email domains
with open('./gen-data/top-domains.txt', 'r') as file:
    # Read the entire file content
    EMAIL_DOMAINS = file.read()
    EMAIL_DOMAINS = EMAIL_DOMAINS.split('\n')

# Create Syn. PII Data
TOTAL = 10000  # Generate 10,000
students = []
for i in tqdm(range(TOTAL)):
    students.append(generate_student_info())

# Store results in dataframe
df = pd.DataFrame(students)

# Reset index
df = df.reset_index(drop=True)
# Save to the csv file
df.to_csv(
    Path(
        CFG.gen_dir) /
    f"pii_syn_data.csv",
    index=False,
    encoding='UTF-8')
print(f'{Path(CFG.gen_dir) /f"pii_syn_data.csv"}')
print('End of Script - Complete')
