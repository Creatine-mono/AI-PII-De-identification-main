# -*- coding: utf-8 -*-
import os
import sys
import math
import json
import random
from pathlib import Path
import argparse
import gc
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from huggingface_hub import HfApi, create_repo
import wandb

# =========================
# Config & Globals
# =========================
ALL_LABELS = [
    'B-NAME', 'I-NAME',
    'B-EMAIL', 'I-EMAIL',
    'B-USERNAME', 'I-USERNAME',
    'B-ID_NUM', 'I-ID_NUM',
    'B-PHONE_NUM', 'I-PHONE_NUM',
    'B-URL_PERSONAL', 'I-URL_PERSONAL',
    'B-STREET_ADDRESS', 'I-STREET_ADDRESS',
    'B-DATE_OF_BIRTH', 'I-DATE_OF_BIRTH',
    'B-AGE', 'I-AGE',
    'B-CREDIT_CARD_INFO', 'I-CREDIT_CARD_INFO',
    'B-BANKING_NUMBER', 'I-BANKING_NUMBER',
    'B-ORGANIZATION_NAME', 'I-ORGANIZATION_NAME',
    'B-DATE', 'I-DATE',
    'B-PASSWORD', 'I-PASSWORD',
    'B-SECURE_CREDENTIAL', 'I-SECURE_CREDENTIAL',
    'O'
]
# 정수 -> 문자열
id2label = {i: lab for i, lab in enumerate(ALL_LABELS)}
# 문자열 -> 정수
label2id = {lab: i for i, lab in id2label.items()}

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# Metrics (seqeval)
# =========================
# 스팬 기반이 아니라 토큰 레벨 F1(일반 NER) 기준. 필요 시 스팬 매칭으로 바꿔도 됨.
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_metrics_fn(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=-1)

    # -100 제거하고 라벨명으로 변환
    true_labels, true_preds = [], []
    for p, l in zip(preds, labels):
        tl, tp = [], []
        for pi, li in zip(p, l):
            if li == -100:
                continue
            tl.append(id2label[li])
            tp.append(id2label[pi])
        true_labels.append(tl)
        true_preds.append(tp)

    precision = precision_score(true_labels, true_preds)
    recall = recall_score(true_labels, true_preds)
    f1 = f1_score(true_labels, true_preds)
    acc = accuracy_score(true_labels, true_preds)

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

# =========================
# Loss (옵션): Focal
# =========================
# 운 샘플의 손실 비중을 줄이고, 어려운 샘플의 손실 비중을 키우는 로스
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
        
#  Transformers의 Trainer를 상속받아 손실 계산만 바꾼 버전
class CustomTrainer(Trainer):
    def __init__(self, use_focal=False, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.use_focal = use_focal
        self.class_weights = class_weights

    # 모델 출력 로짓과 라벨을 (배치*길이) 평탄화
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, L, C)
        # flatten
        logits = logits.view(-1, model.config.num_labels)
        labels = labels.view(-1)

        if self.use_focal:
            loss_fct = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
            loss = loss_fct(logits, labels)
        else:
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# =========================
# Data utils
# =========================
# 입력 JSONL의 tokens/labels가 리스트/딕셔너리/문자열 등 섞여도 리스트[str]로 통일
def normalize_tokens(toks):
    # 기대형태: list[str]
    if isinstance(toks, list):
        if len(toks) > 0 and isinstance(toks[0], dict):
            if "text" in toks[0]:  # [{"text": "안녕"}, ...]
                return [t["text"] for t in toks]
            if "token" in toks[0]:  # [{"token": "안녕"}, ...]
                return [t["token"] for t in toks]
        if len(toks) > 0 and isinstance(toks[0], str):
            return toks
        return [str(x) for x in toks]
    elif isinstance(toks, str):
        return toks.split()
    else:
        return [str(toks)]

def normalize_labels(labs):
    # 기대형태: list[str]
    if isinstance(labs, list):
        if len(labs) > 0 and isinstance(labs[0], dict):
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
        
# 원천 토큰/라벨 길이 차이 보정(O 패딩)
def build_align_fn(tokenizer, max_length=512):
    def align_labels_with_tokens(batch):
        toks_batch, labs_batch = [], []
        for toks, labs in zip(batch["tokens"], batch["labels"]):
            t = normalize_tokens(toks)
            l = normalize_labels(labs)
            if len(l) != len(t):
                if len(l) < len(t):
                    l = l + ["O"] * (len(t) - len(l))
                else:
                    l = l[:len(t)]
            toks_batch.append(t)
            labs_batch.append(l)

        tok = tokenizer(
            toks_batch,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
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
    return align_labels_with_tokens

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True, help="학습용 JSONL 경로")
    parser.add_argument("--project", default="PII-Detection-Korean-NER")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_private", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    args = parser.parse_args()

    seed_everything(42)

    # -------- Model / Tokenizer --------
    model_id = "klue/roberta-large"
    print(f"[Model] {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(ALL_LABELS),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # -------- HF Hub (optional) --------
    api = HfApi()
    username = None
    try:
        who = api.whoami()
        username = who.get("name")
    except Exception as e:
        print("[HF] whoami failed (not logged in?). You can still train locally.", e)

    run_name = f"{model_id.split('/')[-1]}-{datetime.now().strftime('%y%m%d-%H%M')}"
    hf_repo_id = None
    if args.push_to_hub and username:
        repo_name = f"{model_id.split('/')[-1]}-pii-{run_name}"
        hf_repo_id = f"{username}/{repo_name}"
        create_repo(repo_id=hf_repo_id, private=args.hf_private, exist_ok=True)
        print(f"[HF] Repo ready: {hf_repo_id}")

    # -------- W&B --------
    try:
        wandb.login(relogin=False)
    except Exception as e:
        print("[W&B] login skipped:", e)
    run = wandb.init(project=args.project, name=run_name)
    wandb.config.update(vars(args))

    # -------- Data --------
    # JSONL은 {"tokens": [...], "labels": [...]} 형태를 가정(프롬프트식이어도 위 정규화로 수용)
    full_ds = load_dataset("json", data_files={"data": args.jsonl_path})["data"]
    # 8:2 split
    split = full_ds.train_test_split(test_size=0.2, seed=42)
    raw_train, raw_val = split["train"], split["test"]
    print(f"[Data] train={len(raw_train)}, val={len(raw_val)}")

    align_fn = build_align_fn(tokenizer, max_length=args.max_length)
    ds_train = raw_train.map(align_fn, batched=True, remove_columns=raw_train.column_names, desc="Tokenize train")
    ds_val = raw_val.map(align_fn, batched=True, remove_columns=raw_val.column_names, desc="Tokenize val")
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # -------- Class Weights (optional) --------
    class_weights_t = None
    if args.use_class_weights:
        all_labels_flat = [lid for seq in ds_train["labels"] for lid in seq if lid != -100]
        if len(all_labels_flat) > 0:
            unique, counts = np.unique(all_labels_flat, return_counts=True)
            freq = {u: c for u, c in zip(unique, counts)}
            # inverse frequency (간단 가중)
            weights = np.ones(len(ALL_LABELS), dtype=np.float32)
            total = sum(counts)
            for i in range(len(ALL_LABELS)):
                weights[i] = total / (freq.get(i, 1.0) * len(ALL_LABELS))
            class_weights_t = torch.tensor(weights)
            if torch.cuda.is_available():
                class_weights_t = class_weights_t.cuda()
            print("[ClassWeights] enabled")
        else:
            print("[ClassWeights] skipped (no labels?)")

    # -------- TrainingArguments --------
    args_tr = TrainingArguments(
        output_dir=f"./results/{run_name}",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(),

        logging_strategy="steps",
        logging_steps=10,

        eval_strategy="steps",  # ← 중요!
        eval_steps=50,                # ← 중요!

        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        report_to=["wandb"],
        push_to_hub=bool(hf_repo_id is not None),
        hub_model_id=hf_repo_id if hf_repo_id else None,
    )

    # -------- Trainer --------
    trainer = CustomTrainer(
        model=model,
        args=args_tr,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        use_focal=args.use_focal,
        class_weights=class_weights_t,
    )

    # -------- Train --------
    print("[Train] start")
    trainer.train()
    print("[Train] done")

    # -------- Push / Finish --------
    if hf_repo_id:
        trainer.push_to_hub(commit_message="End of training")

    wandb.finish()
    del trainer, model
    torch.cuda.empty_cache()
    _ = gc.collect()
    print("[Done]")

if __name__ == "__main__":
    main()
