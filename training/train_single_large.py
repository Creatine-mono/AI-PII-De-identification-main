# Setup Env. Variables
import sys
import os
from pathlib import Path

# Add project root to Python path for package imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.cxmetrics import train_metrics
from src.cxmetrics import compute_metrics
from src.utils import (load_cfg,
                      debugger_is_active,
                      seed_everything)
from src.load_data import LoadData
import src.create_datasets as create_datasets
from pathlib import Path
import json
import argparse
from itertools import chain
from functools import partial
import math
import shutil
import pandas as pd
import numpy as np
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset, features, concatenate_datasets
import wandb
from scipy.special import softmax
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tokenizers import AddedToken
from huggingface_hub import HfApi, create_repo, upload_folder
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
from types import SimpleNamespace
from pytorch_lightning import seed_everything as pl_seed
import copy
import gc
import sys
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ['TOKENIZERS_PARALLELISM'] = 'True'
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'True'

# Do NOT log models to WandB
os.environ["WANDB_LOG_MODEL"] = "false"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"


# Custom (cx) modules


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class CustomTrainer(Trainer):
    def __init__(
            self,
            focal_loss_info: SimpleNamespace,
            *args,
            class_weights=None,
            **kwargs):
        super().__init__(*args, **kwargs)
        # Assuming class_weights is a Tensor of weights for each class
        self.class_weights = class_weights
        self.focal_loss_info = focal_loss_info

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # Reshape for loss calculation
        if self.focal_loss_info.apply:
            loss_fct = FocalLoss(alpha=5, gamma=2, reduction='mean')
            loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                            labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            if self.label_smoother is not None and "labels" in inputs:
                loss = self.label_smoother(outputs, inputs)
            else:
                loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                                labels.view(-1))
        return (loss, outputs) if return_outputs else loss


ALL_LABELS = ['B-EMAIL','B-ID_NUM','B-NAME_STUDENT','B-PHONE_NUM',
              'B-STREET_ADDRESS','B-URL_PERSONAL','B-USERNAME',
              'I-ID_NUM','I-NAME_STUDENT','I-PHONE_NUM',
              'I-STREET_ADDRESS','I-URL_PERSONAL','O']  # O는 마지막
id2label = {i: lab for i, lab in enumerate(ALL_LABELS)}
label2id = {lab: i for i, lab in id2label.items()}


if __name__ == '__main__':

    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = os.getenv('BASE_DIR') + '/cfgs/single-gpu'
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

    def seed_everything_local(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Load the configuration file
    CFG = load_cfg(base_dir=Path(args.dir), filename=args.name)
    pl_seed(getattr(CFG, "seed", 42))
    
    # Setup WandB (환경변수 키 없이 캐시 로그인)
    try:
        wandb.login(relogin=False)  # 이미 로그인돼 있으면 패스
    except Exception as e:
        print("[W&B] 먼저 터미널에서 `wandb login` 실행이 필요합니다:", e)
    run = wandb.init(project='PII')
    if CFG.debug:
        run.name = 'junk-debug'

    model_id = getattr(CFG.model, "name", None)
    if not model_id:
        raise ValueError("CFG.model.name이 비어있습니다. YAML에서 model.name을 설정하세요.")
    # 잘못된 축약형이면 보정
    if model_id == "deberta-v3-large":
        model_id = "microsoft/deberta-v3-large"


   # 0) ONLINE 전환 (모델/토크나이저 로드/리포 생성은 네트워크 필요)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    
    # 1) 로그인 확인
    api = HfApi()
    try:
        who = api.whoami()   # 캐시 토큰 사용
        username = who.get("name") or who.get("email") or "unknown"
        print(f"[HF] Logged in as: {username}")
    except Exception as e:
        raise SystemExit("[HF] 먼저 `huggingface-cli login` 하세요.") from e
    
    default_repo_name = f"{model_id}-pii-{run.name}".replace('/', '-')
    hf_repo_name = default_repo_name
    
    # fast 토크나이저 권장 (word_ids() 필요)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    def align_labels_with_tokens(batch):
        def normalize_tokens(toks):
            # 기대형태: list[str]
            if isinstance(toks, list):
                if len(toks) > 0 and isinstance(toks[0], dict):
                    # 예: [{"text": "Hello", ...}, ...] / [{"token": "Hello"}, ...]
                    if "text" in toks[0]:
                        return [t["text"] for t in toks]
                    if "token" in toks[0]:
                        return [t["token"] for t in toks]
                if len(toks) > 0 and isinstance(toks[0], str):
                    return toks
                return [str(x) for x in toks]
            elif isinstance(toks, str):
                # 공백 기준 토큰화(임시)
                return toks.split()
            else:
                return [str(toks)]
    
        def normalize_labels(labs):
            # 기대형태: list[str] (라벨 문자열)
            if isinstance(labs, list):
                if len(labs) > 0 and isinstance(labs[0], dict):
                    # 예: [{"label":"B-EMAIL"}, ...] or {"labels": "..."}
                    key = "label" if "label" in labs[0] else ("labels" if "labels" in labs[0] else None)
                    if key is not None:
                        return [l[key] for l in labs]
                if len(labs) > 0 and isinstance(labs[0], str):
                    return labs
                return [str(x) for x in labs]
            elif isinstance(labs, str):
                return labs.split()
            else:
                return [str(labs)]
    
        toks_batch, labs_batch = [], []
        for toks, labs in zip(batch["tokens"], batch["labels"]):
            toks = normalize_tokens(toks)
            labs = normalize_labels(labs)
            # 길이 안 맞으면 라벨을 O로 패딩/잘라내기 (응급처치)
            if len(labs) != len(toks):
                if len(labs) < len(toks):
                    labs = labs + ["O"] * (len(toks) - len(labs))
                else:
                    labs = labs[:len(toks)]
            toks_batch.append(toks)
            labs_batch.append(labs)
    
        tok = tokenizer(
            toks_batch,
            is_split_into_words=True,
            truncation=True,
            max_length=256,
            padding=False,
            add_special_tokens=True,
        )
    
        new_labels = []
        for i, labs in enumerate(labs_batch):
            word_ids = tok.word_ids(batch_index=i)
            ids = []
            for wid in word_ids:
                if wid is None:
                    ids.append(-100)
                else:
                    ids.append(label2id.get(labs[wid], label2id["O"]))
            new_labels.append(ids)
    
        tok["labels"] = new_labels
        return tok

    
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,     # ← CFG.model.name  → model_id
        num_labels=len(ALL_LABELS), id2label=id2label, label2id=label2id, use_safetensors=True 
    )
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
    # 학습용 JSONL 경로
    jsonl_path = str(Path(os.getenv('DATA_DIR')) / 'mdd-gen/llama3_placeholder_2.3K_v0.jsonl')
    
    # split 하지 말고 통째로 train으로 사용
    ds_dict = load_dataset("json", data_files={"train": jsonl_path})
    raw_train = ds_dict["train"]
    raw_val = None  # 평가 안 씀
    print(f"Loaded {len(raw_train)} examples")
    
    # 모델에不要한 원본 컬럼 제거(남길 컬럼만 유지)
    keep_cols = {"tokens", "labels"}
    cols_remove_train = [c for c in raw_train.column_names if c not in keep_cols]
    
    ds_train = raw_train.map(
        align_labels_with_tokens,
        batched=True,
        remove_columns=[c for c in raw_train.column_names if c not in {"tokens", "labels"}],
        desc="Tokenizing train"
    )
        
    # ==== 스텝 계산 (ds_train/ds_val 생성 직후) ====
    train_size = len(ds_train)
    bsz = CFG.train_args.per_device_train_batch_size
    ga  = CFG.train_args.gradient_accumulation_steps
    nep = CFG.train_args.num_train_epochs
    frac = getattr(CFG.train_args, "eval_epoch_fraction", 0.2)  # 없으면 0.2 기본값
    
    if bsz <= 0 or ga <= 0:
        raise ValueError("per_device_train_batch_size와 gradient_accumulation_steps는 1 이상이어야 합니다.")
    
    # 에폭당 스텝 수
    steps_per_epoch = math.ceil(train_size / (bsz * ga))
    
    # 전체 스텝 수
    num_steps = int(steps_per_epoch * nep)
    
    # 에폭의 frac 비율마다 평가
    eval_steps = max(1, int(math.ceil(steps_per_epoch * frac)))
    
    print(f"Train size: {train_size:,}")
    print(f"Steps/epoch: {steps_per_epoch:,}")
    print(f"My Calculated NUM_STEPS: {num_steps:,}")
    print(f"My Calculated eval_steps: {eval_steps:,}")

    
    # 안전 가드
    if len(ds_train) == 0:
        raise ValueError("Empty train/val dataset after tokenization. Check JSONL path and columns.")

    
    # 콜레이터
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
   

    
    # 4) 로컬 저장 (출력 디렉토리 준비)
    output_dir = Path(os.getenv('SAVE_DIR')) / f'{run.name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"[LOCAL] Saved tokenizer & model at: {str(output_dir)}")
    
    # 5) Hub 업로드 (git-lfs 자동, 대용량 파일 처리)
    #    - repo_id 에 사용자명 생략 가능(현재 로그인 네임스페이스로 업로드)
    try:
        upload_folder(
            repo_id=f"{username}/{hf_repo_name}",
            folder_path=str(output_dir),
            path_in_repo=".",
            commit_message="upload tokenizer & model artifacts",
        )
        print(f"[HF] Uploaded to: https://huggingface.co/{who.get('name', username)}/{hf_repo_name}")
    except Exception as e:
        print(f"[HF] upload_folder warning: {e}")
    
    # 6) (선택) 학습 중엔 네트워크 차단하고 싶으면 다시 OFFLINE
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
   
    # ↓↓↓ TrainingArguments는 v5 기준으로 eval_strategy 사용, 학습 중 push는 끈다(자동 create_repo 방지)
    gradient_checkpointing_kwargs = {'use_reentrant': getattr(CFG.train_args, 'use_reentrant', False)}
    
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,      # 무조건 1
        gradient_accumulation_steps=1,      # 무조건 1
        num_train_epochs=CFG.train_args.num_train_epochs,
    
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        load_best_model_at_end=False,
    
        weight_decay=0.0,
        warmup_ratio=0.0,
    
        # 메모리 절약 옵션
        fp16=True,                          # GPU가 fp16 지원 시
        bf16=False,                         # 둘 중 하나만 True
        gradient_checkpointing=True,
        push_to_hub=False,
    )

    class_weights = None
    if CFG.class_weights.apply or CFG.focal_loss.apply:
        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
            tokenizer=tokenizer,  # v5 경고 무시는 가능. 원하면 processing_class=tokenizer로 변경
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            data_collator=collator,
            tokenizer=tokenizer, )
    
    # 학습
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    
    # 학습 종료 후, 현재 로그인된 psh3333 네임스페이스로 수동 push
    trainer.push_to_hub(
        repo_id=hf_repo_name,     # 사용자명 없이 → 로그인 계정(psh3333) 네임스페이스
        private=True,
        commit_message="final model upload",
    )
    print(f"[HF] Pushed to: https://huggingface.co/{username}/{hf_repo_name}")

    ############################################
    # Log Metrics to WandB
    ############################################
    # Trainer optimal checkpoint steps
    best_ckpt = trainer.state.best_model_checkpoint
    best_val_metric = trainer.state.best_metric
    print(f'Best CKPT: {best_ckpt}')
    print(f'best_val_metric: {best_val_metric}')

    # Log F5 score for holdout
    run.log({'best_ckpt': best_ckpt,
             'best_val_metric': best_val_metric})

    # Num. steps for best checkpoint
    log_hist = copy.deepcopy(trainer.state.log_history)
    metric_name = f'eval_{CFG.train_args.metric_for_best_model}'
    optimal_steps = None
    for log_ in log_hist:
        for key, value in log_.items():
            if key == metric_name and value == best_val_metric:
                optimal_steps = log_['step']
    assert optimal_steps is not None, 'Error in Finding Optimal Steps'
    print(f'Optimal Steps: {optimal_steps:,}')
    print(f'trainer.state.max_steps: {trainer.state.max_steps:,}')
    del log_hist, metric_name, log_, key, value
    _ = gc.collect()

    # Final wandb log
    run.log({
        'optimal_steps_post': optimal_steps,
        'class_weights_approach': getattr(CFG.class_weights, 'approach', 'none'),
        'dataset_name': jsonl_path,           
        'model_name': CFG.model.name,
        'max_steps_post': trainer.state.max_steps
    })

    # Close wandb logger
    wandb.finish()
    ############################################
    # Clean up memory
    ############################################
    del run, trainer, model
    torch.cuda.empty_cache()
    _ = gc.collect()

    # ############################################
    # # Train on All Data
    # ############################################
    # # Create directory for saving all_data training
    # output_all_dir = output_dir / 'all_data'
    # output_tokenizer_dir = output_dir / 'tokenizer'
    # if not output_all_dir.exists():
    #     output_all_dir.mkdir(parents=False, exist_ok=True)
    #     output_tokenizer_dir.mkdir(parents=False, exist_ok=True)

    # # Combine train and val datasets
    # ds_all = concatenate_datasets([ds_train, ds_val])
    # ds_all = ds_all.shuffle(42)

    # # Model
    # model = AutoModelForTokenClassification.from_pretrained(
    #     str(Path(os.getenv('MODEL_DIR')) / CFG.model.name),
    #     num_labels=len(all_labels),
    #     id2label=id2label,
    #     label2id=label2id,
    #     ignore_mismatched_sizes=True,
    # )
    # # Resize model token embeddings if tokens were added
    # if CFG.tokenizer.add_tokens is not None:
    #     model.resize_token_embeddings(
    #         len(tokenizer),
    #         pad_to_multiple_of=CFG.tokenizer.pad_to_multiple_of,
    #     )

    # # Trainer Arguments
    # args = TrainingArguments(
    # output_dir=str(output_all_dir),
    # fp16=CFG.train_args.fp16,
    # learning_rate=CFG.train_args.learning_rate,
    # per_device_train_batch_size=CFG.train_args.per_device_train_batch_size,
    # gradient_accumulation_steps=CFG.train_args.gradient_accumulation_steps,
    # report_to="none",
    # lr_scheduler_type='cosine',
    # warmup_ratio=CFG.train_args.warmup_ratio,
    # weight_decay=CFG.train_args.weight_decay,
    # max_steps=optimal_steps,
    # evaluation_strategy="no",
    # save_total_limit=1,
    # )

    # # Initialize Trainer with custom class weights
    # if not CFG.class_weights.apply and not CFG.focal_loss.apply:
    #     trainer = Trainer(
    #         model=model,
    #         args=args,
    #         train_dataset=ds_all,
    #         data_collator=collator,
    #         tokenizer=tokenizer,
    #         compute_metrics=partial(train_metrics, all_labels=all_labels),
    #     )
    # else:
    #     trainer = CustomTrainer(
    #         model=model,
    #         args=args,
    #         train_dataset=ds_all,
    #         data_collator=collator,
    #         tokenizer=tokenizer,
    #         compute_metrics=partial(train_metrics, all_labels=all_labels),
    #         class_weights=class_weights,
    #         focal_loss_info=CFG.focal_loss,
    #     )
    # trainer.train()

    # # Save the trainer
    # trainer.save_model(output_dir=output_all_dir)
    # tokenizer.save_pretrained(save_directory=output_tokenizer_dir)

    print('checkpoint')
print('End of Script - Complete')
