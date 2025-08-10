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
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainerCallback
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
        # --- 설정 및 시드 고정 (기존과 동일) ---
        is_debugger = debugger_is_active()
        if is_debugger:
            # (디버그용 설정)
            args = argparse.Namespace()
            args.dir = os.getenv('BASE_DIR') + '/cfgs/single-gpu'
            args.name = 'cfg1.yaml'
        else:
            # (실행용 설정)
            parser = argparse.ArgumentParser(description="NER 모델 학습 스크립트")
            parser.add_argument("--dir", required=True, help="YAML 설정 파일이 있는 디렉토리")
            parser.add_argument("--name", required=True, help="YAML 설정 파일 이름")
            args = parser.parse_args()
    
        CFG = load_cfg(base_dir=Path(args.dir), filename=args.name)
        pl_seed(getattr(CFG, "seed", 42))
    
        # --- 1. W&B 초기화 (수정 완료) ---
        # ✅ 스크립트 시작 시 한 번만 초기화
        try:
            wandb.login(relogin=False)
        except Exception as e:
            print(f"[W&B] W&B login failed: {e}. Please run 'wandb login' in your terminal.")
        
        run_name = f"{CFG.model.name.replace('/', '-')}-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}"
        run = wandb.init(
            project="PII-Detection-Korean-NER",
            name=run_name,
            config=vars(CFG)
        )
        print(f"✅ WandB run initialized: {run.name}")
    
        # --- 2. 모델 및 토크나이저 설정 (수정 완료) ---
        model_id = "klue/roberta-large"
        print(f"✅ Using Korean-specific model: {model_id}")
    
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=len(ALL_LABELS),
            id2label=id2label,
            label2id=label2id,
            use_safetensors=True
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
        # --- 3. Hugging Face Hub 레포지토리 설정 (❗️수정) ---
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        api = HfApi()
        who = api.whoami()
        username = who.get("name")
        
        # ✅ 'username/모델이름' 형태로 repo_id를 한 번만 올바르게 생성
        model_name_for_repo = f"{model_id.split('/')[-1]}-pii-{run.name}"
        hf_repo_id = f"{username}/{model_name_for_repo}"
        
        create_repo(
            repo_id=hf_repo_id,
            private=True,
            exist_ok=True,
        )
        print(f"✅ Hugging Face repo ready: {hf_repo_id}")
    
        # --- 4. 데이터 로드 및 전처리 (❗️수정) ---
        def align_labels_with_tokens(batch):
            def normalize_tokens(toks):
                if isinstance(toks, list):
                    if len(toks) > 0 and isinstance(toks[0], dict):
                        if "text" in toks[0]: return [t["text"] for t in toks]
                        if "token" in toks[0]: return [t["token"] for t in toks]
                    if len(toks) > 0 and isinstance(toks[0], str): return toks
                    return [str(x) for x in toks]
                elif isinstance(toks, str): return toks.split()
                else: return [str(toks)]
            def normalize_labels(labs):
                if isinstance(labs, list):
                    if len(labs) > 0 and isinstance(labs[0], dict):
                        key = "label" if "label" in labs[0] else ("labels" if "labels" in labs[0] else None)
                        if key is not None: return [l[key] for l in labs]
                    if len(labs) > 0 and isinstance(labs[0], str): return labs
                    return [str(x) for x in labs]
                elif isinstance(labs, str): return labs.split()
                else: return [str(labs)]
            toks_batch, labs_batch = [], []
            for toks, labs in zip(batch["tokens"], batch["labels"]):
                toks, labs = normalize_tokens(toks), normalize_labels(labs)
                if len(labs) != len(toks):
                    labs = (labs + ["O"] * len(toks))[:len(toks)]
                toks_batch.append(toks); labs_batch.append(labs)
            tok = tokenizer(toks_batch, is_split_into_words=True, truncation=True, max_length=256)
            new_labels = []
            for i, labs in enumerate(labs_batch):
                word_ids = tok.word_ids(batch_index=i)
                ids = [-100 if wid is None else label2id.get(labs[wid], label2id["O"]) for wid in word_ids]
                new_labels.append(ids)
            tok["labels"] = new_labels
            return tok
    
        jsonl_path = str(Path(os.getenv('DATA_DIR')) / 'mdd-gen/llama3_placeholder_2.3K_v0.jsonl')
        full_dataset = load_dataset("json", data_files={"train": jsonl_path})["train"]
    
        # ✅ 데이터를 학습용과 검증용으로 8:2 분리
        split_datasets = full_dataset.train_test_split(test_size=0.2, seed=42)
        raw_train, raw_val = split_datasets["train"], split_datasets["test"]
        print(f"✅ Data split: {len(raw_train)} for training, {len(raw_val)} for validation.")
    
        # ✅ 학습셋과 검증셋 모두에 전처리 적용
        ds_train = raw_train.map(align_labels_with_tokens, batched=True, remove_columns=raw_train.column_names)
        ds_val = raw_val.map(align_labels_with_tokens, batched=True, remove_columns=raw_val.column_names)
    
        # --- 5. 클래스 가중치 계산 (❗️추가) ---
        class_weights = None
        if CFG.class_weights.apply:
            print("Calculating class weights...")
            all_labels_flat = [label for sublist in ds_train['labels'] for label in sublist if label != -100]
            unique_labels = np.unique(all_labels_flat)
            class_weights_arr = compute_class_weight('balanced', classes=unique_labels, y=all_labels_flat)
            
            class_weights_map = {label: weight for label, weight in zip(unique_labels, class_weights_arr)}
            weights = torch.zeros(len(ALL_LABELS), dtype=torch.float32)
            for i, label_name in id2label.items():
                label_id = label2id[label_name]
                weights[i] = class_weights_map.get(label_id, 1.0) # 데이터에 없는 라벨은 가중치 1.0
            
            class_weights = weights.to('cuda' if torch.cuda.is_available() else 'cpu')
            print("✅ Class weights calculated and applied.")
    
        collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    
        # --- 6. TrainingArguments 설정 (❗️수정) ---
        # ✅ 평가 및 저장 전략 활성화
        args = TrainingArguments(
            output_dir=f"./results/{run.name}",
            per_device_train_batch_size=CFG.train_args.per_device_train_batch_size,
            gradient_accumulation_steps=CFG.train_args.gradient_accumulation_steps,
            num_train_epochs=CFG.train_args.num_train_epochs,
            learning_rate=CFG.train_args.learning_rate,
            warmup_ratio=CFG.train_args.warmup_ratio,
            weight_decay=CFG.train_args.weight_decay,
            fp16=torch.cuda.is_available(),
            
            logging_strategy="steps",
            logging_steps=10,
            
            eval_strategy="steps",
            evaluation_steps=50,
            
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            
            load_best_model_at_end=True,
            metric_for_best_model="f1", # compute_metrics 함수가 'f1' 키를 반환해야 함
            
            report_to=["wandb"],
            hub_model_id=hf_repo_id,
            push_to_hub=True,
        )
    
        # --- 7. Trainer 초기화 (❗️수정) ---
        # ✅ CustomTrainer 사용 및 모든 인자 전달
        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,           # ✅ 검증 데이터셋 전달
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics, # ✅ 평가 함수 전달
            class_weights=class_weights,     # ✅ 클래스 가중치 전달
            focal_loss_info=CFG.focal_loss,
        )
    
        # --- 8. 학습 및 정리 (❗️수정) ---
        # ❗️ 불필요한 로직(사전 업로드, 중복 저장, 콜백) 모두 제거
        print("🚀 Starting training...")
        trainer.train()
        print("✅ Training complete.")
    
        # ✅ 학습 종료 후 최종 모델 한 번만 Hub에 푸시 (선택사항, TrainingArguments가 이미 처리)
        # trainer.push_to_hub(commit_message="End of training, final model upload.")
        
        wandb.finish()
        print("✅ WandB run finished.")
    
        del run, trainer, model
        torch.cuda.empty_cache()
        _ = gc.collect()
    
        print('End of Script - Complete')
