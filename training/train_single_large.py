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
import torch
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset, features, concatenate_datasets
import wandb
from scipy.special import softmax
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from tokenizers import AddedToken
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import copy
import gc
import sys
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['TOKENIZERS_PARALLELISM'] = 'True'
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


ALL_LABELS = ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM',
              'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME',
              'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM',
              'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O']

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

    # Load the configuration file
    CFG = load_cfg(base_dir=Path(args.dir),
      정
    from huggingface_hub import HfApi, create_repo  # 반드시 import
    
    # Calculate num train steps
    num_steps = CFG.train_args.num_train_epochs * len(ds_train)
    num_steps = num_steps / CFG.train_args.per_device_train_batch_size
    num_steps = num_steps / CFG.train_args.gradient_accumulation_steps
    print(f'My Calculated NUM_STEPS: {num_steps:,.2f}')
    
    # Step per epoch to eval every 0.2 epochs
    eval_steps = int(math.ceil((num_steps / CFG.train_args.num_train_epochs) *
                               CFG.train_args.eval_epoch_fraction))
    print(f'My Calculated eval_steps: {eval_steps:,}')
    
    # Setup WandB (환경변수 키 없이 캐시 로그인)
    try:
        wandb.login(relogin=False)  # 이미 로그인돼 있으면 패스
    except Exception as e:
        print("[W&B] 먼저 터미널에서 `wandb login` 실행이 필요합니다:", e)
    run = wandb.init(project='PII')
    if CFG.debug:
        run.name = 'junk-debug'
    
    # ===== HF: 로그인 캐시로 현재 계정 확인 & 리포 설정 =====
    api = HfApi()
    try:
        who = api.whoami()  # 캐시된 로그인 정보 사용
        username = who.get("name")
        if not username:
            raise RuntimeError("no username")
    except Exception as e:
        raise SystemExit("[HF] 먼저 `huggingface-cli login` 하세요.") from e
    
    default_repo_name = f"{CFG.model.name}-pii-{run.name}".replace('/', '-')
    # 네 계정이 psh3333 라고 했으니, whoami 결과가 psh3333이어야 정상
    # repo_id는 '사용자명/리포명'으로 명시하거나, 사용자명 없이 리포명만 써도 현재 로그인 네임스페이스로 생성됨
    hf_repo_name = default_repo_name  # ← 사용자명 없이
    # hf_repo_id = f"{username}/{default_repo_name}"  # ← 이렇게 써도 OK
    
    # 출력 디렉토리
    output_dir = Path(os.getenv('SAVE_DIR')) / f'{run.name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    if run.name == 'junk-debug':
        os.system(f'rm -rf {str(output_dir)}/*')
    shutil.copyfile(str(Path(args.dir) / args.name), str(output_dir / args.name))
    
    # 업로드 위해 오프라인 해제 후 리포 생성(있으면 통과)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    try:
        create_repo(repo_id=hf_repo_name, private=True, exist_ok=True)
    except Exception as e:
        print(f"[HF] create_repo warning: {e}")
    
    # 토크나이저 로컬 저장
    tokenizer.save_pretrained(output_dir)
    
    # ↓↓↓ TrainingArguments는 v5 기준으로 eval_strategy 사용, 학습 중 push는 끈다(자동 create_repo 방지)
    gradient_checkpointing_kwargs = {'use_reentrant': CFG.train_args.use_reentrant}
    args = TrainingArguments(
        output_dir=str(output_dir),
        fp16=CFG.train_args.fp16,
        learning_rate=CFG.train_args.learning_rate,
        num_train_epochs=CFG.train_args.num_train_epochs,
        per_device_train_batch_size=CFG.train_args.per_device_train_batch_size,
        gradient_accumulation_steps=CFG.train_args.gradient_accumulation_steps,
        per_device_eval_batch_size=CFG.train_args.per_device_train_batch_size,
        report_to="wandb",
        eval_strategy="steps",           # ← evaluation_strategy 아님
        save_total_limit=2,
        logging_steps=eval_steps,
        save_steps=eval_steps,
        lr_scheduler_type=CFG.train_args.lr_scheduler_type,
        metric_for_best_model=CFG.train_args.metric_for_best_model,
        greater_is_better=CFG.train_args.greater_is_better,
        warmup_ratio=CFG.train_args.warmup_ratio,
        weight_decay=CFG.train_args.weight_decay,
        load_best_model_at_end=True,
        gradient_checkpointing=CFG.train_args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    
        push_to_hub=False,               # ← 학습 중 자동 업로드/리포생성 막기
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
            compute_metrics=partial(train_metrics, all_labels=ALL_LABELS),
            class_weights=class_weights,
            focal_loss_info=CFG.focal_loss,
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
            tokenizer=tokenizer,  # FutureWarning만 뜸
            compute_metrics=partial(train_metrics, all_labels=ALL_LABELS),
        )
    
    # 학습
    trainer.train()
    
    # 학습 종료 후, 현재 로그인된 psh3333 네임스페이스로 수동 push
    trainer.push_to_hub(
        repo_id=hf_repo_name,     # 사용자명 없이 → 로그인 계정(psh3333) 네임스페이스
        private=True,
        commit_message="final model upload",
    )
    print(f"[HF] Pushed to: https://huggingface.co/{username}/{hf_repo_name}")

    ############################################
    # F5 Score on Validation Dataset
    # Adjust Threshold
    ############################################

    # Predict on val dataset
    predictions = trainer.predict(ds_val).predictions
    weighted_preds = softmax(predictions, axis=-1) * 1.0
    preds = weighted_preds.argmax(-1)
    # preds_without_O = weighted_preds[:, :, :12].argmax(-1)
    # O_preds = weighted_preds[:, :, 12]
    preds_without_O = weighted_preds[:, :, :-1].argmax(-1)
    O_preds = weighted_preds[:, :, -1]

    # Test various threshold levels
    f5_scores = {}
    for threshold in [0.1, 0.2, 0.3, 0.4,
                      0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
        preds_final = np.where(O_preds < threshold, preds_without_O, preds)
        # Prepare to plunder the data for valuable triplets!
        triplets = []
        document, token, label, token_str = [], [], [], []
        # For each prediction, token mapping, offsets, tokens, and document in
        # the dataset
        for p, row in zip(preds_final, ds_val):
            token_map = row['token_map']
            offsets = row['offset_mapping']
            tokens = row['tokens']
            doc = row['document']

            # Iterate through each token prediction and its corresponding
            # offsets
            for token_pred, (start_idx, end_idx) in zip(p, offsets):
                label_pred = id2label[token_pred]  # Predicted label from token

                # If start and end indices sum to zero, continue to the next
                # iteration
                if start_idx + end_idx == 0:
                    continue

                # If the token mapping at the start index is -1, increment
                # start index
                if token_map[start_idx] == -1:
                    start_idx += 1

                # Ignore leading whitespace tokens ("\n\n")
                while start_idx < len(
                        token_map) and tokens[token_map[start_idx]].isspace():
                    start_idx += 1

                # If start index exceeds the length of token mapping, break the
                # loop
                if start_idx >= len(token_map):
                    break

                token_id = token_map[start_idx]   # Token ID at start index

                # Ignore "O" predictions and whitespace tokens
                if label_pred != "O" and token_id != -1:
                    # Form a triplet
                    triplet = (doc, token_id)  # Form a triplet

                    # If the triplet is not in the list of triplets, add it
                    if triplet not in triplets:
                        document.append(doc)
                        token.append(token_id)
                        label.append(label_pred)
                        # token_str.append(tokens[token_id])
                        token_str.append(tokens[token_id])
                        triplets.append(triplet)

        # Prediction dataframe
        df_pred = pd.DataFrame({"document": document,
                                "token": token,
                                "label": label,
                                "token_str": token_str})
        # Score val data
        df_ref = df_val.copy()
        df_ref['document'] = df_ref['document'].astype(str)
        df_ref = df_ref[df_ref['document'].isin(
            ds_val['document'])].reset_index(drop=True)
        df_ref = (df_ref.explode(['tokens', 'labels', 'trailing_whitespace'])
                  .reset_index(drop=True)
                  .rename(columns={'labels': 'label'}))
        df_ref['token'] = df_ref.groupby('document').cumcount()
        df_ref = df_ref[df_ref['label'] != 'O'].copy()
        df_ref = df_ref.reset_index().rename(columns={'index': 'row_id'})
        df_ref = df_ref[['row_id', 'document', 'token', 'label']].copy()
        m = compute_metrics(df_pred, df_ref)
        print(f'Threshold: {threshold}; F5: {m["ents_f5"]:.4f}')
        f5_scores[f'f5_{threshold}'] = m['ents_f5']

    # Best threshold for F5
    best_threshold = -1.0
    best_f5 = -1.0
    for name, key in f5_scores.items():
        if key > best_f5:
            best_f5 = key
            best_threshold = float(name.split('f5_')[-1])
    print(f'Best F5: {best_f5:.4f}; Threshold: {best_threshold}')

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
    run.log(f5_scores)
    run.log({'best_f5': best_f5, 'best_threshold': best_threshold})

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
    run.log({'optimal_steps_post': optimal_steps,
             'class_weights_approach': CFG.class_weights.approach,
             'dataset_name': '; '.join(CFG.paths.data.train),
             'model_name': CFG.model.name,
             'max_steps_post': trainer.state.max_steps})

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
