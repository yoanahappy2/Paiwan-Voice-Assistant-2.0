#!/usr/bin/env python3
"""
train_whisper_lora.py — Whisper-tiny LoRA 微調（帶 Loss 記錄）

從原專案複製的訓練腳本，加上：
- CSV Loss 日誌
- 訓練後自動生成 Loss 曲線圖

用於生成消融實驗的 Loss 曲線數據

作者: QA_Tester
日期: 2026-04-15
"""

import torch
import librosa
import pandas as pd
import glob
import numpy as np
import json
import os
from datetime import datetime
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ============================================
# 路徑配置（指向原專案，只讀）
# ============================================
ORIGINAL_PROJECT = "/Users/sbb-mei/paiwan_asr_project"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../docs/training_logs"

# 1. 載入模型
print("=" * 60)
print("Whisper-tiny LoRA 微調 — 排灣語 ASR")
print("=" * 60)

model_id = "openai/whisper-tiny"
print(f"載入模型: {model_id}")

# 嘗試從本地快取載入（不需要連網）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    processor = WhisperProcessor.from_pretrained(model_id, language="Chinese", task="transcribe", local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, local_files_only=True)
    print("✓ 從本地快取載入")
except Exception as e:
    print(f"本地快取失敗: {e}")
    print("嘗試從 hf-mirror 下載...")
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    processor = WhisperProcessor.from_pretrained(model_id, language="Chinese", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

# 2. LoRA 配置
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 3. 載入資料（從原專案）
os.chdir(ORIGINAL_PROJECT)
csv_files = glob.glob("族語彙整*.csv")
if not csv_files:
    print("❌ 找不到訓練數據（族語彙整*.csv）")
    exit(1)

csv_file = csv_files[0]
print(f"訓練數據: {csv_file}")

df = pd.read_csv(csv_file)
valid_data = []
for _, row in df.iterrows():
    path = row['file_path'].strip()
    if os.path.exists(path):
        valid_data.append({"audio_path": path, "sentence": row['transcription']})

print(f"有效樣本: {len(valid_data)} 筆")

dataset = Dataset.from_list(valid_data)

# 4. 資料處理
def prepare_dataset(batch):
    speech, sampling_rate = librosa.load(batch["audio_path"], sr=16000)
    batch["input_features"] = processor.feature_extractor(speech, sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print(f"處理資料中...")
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# 5. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 6. 訓練（帶 Loss 記錄）
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    max_steps=30,         # 30 步就夠畫曲線
    logging_steps=5,        # 每 5 步記錄一次
    save_steps=30,         # 最後才存
    save_total_limit=1,
    fp16=False,
    push_to_hub=False,
    report_to="none",
    use_cpu=False,
    remove_unused_columns=False,
)

# 自訂 Callback 記錄 Loss
from transformers import TrainerCallback

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            entry = {
                "step": state.global_step,
                "loss": logs["loss"],
                "learning_rate": logs.get("learning_rate", 0),
                "timestamp": datetime.now().isoformat()
            }
            self.losses.append(entry)
            if state.global_step % 50 == 0:
                print(f"  Step {state.global_step}/200 | Loss: {logs['loss']:.4f}")

loss_logger = LossLoggerCallback()

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    data_collator=data_collator,
    callbacks=[loss_logger],
)

print(f"\n🚀 開始訓練 (30 steps, 快速 Loss 曲線)...")
t0 = datetime.now()
trainer.train()
elapsed = (datetime.now() - t0).total_seconds()

print(f"\n✅ 訓練完成！耗時 {elapsed:.0f} 秒")

# 7. 儲存 Loss 曲線數據
loss_data = {
    "model": "whisper-tiny + LoRA",
    "total_steps": 30,
    "train_samples": len(valid_data),
    "elapsed_seconds": elapsed,
    "losses": loss_logger.losses,
    "final_loss": loss_logger.losses[-1]["loss"] if loss_logger.losses else None,
}

loss_json = os.path.join(OUTPUT_DIR, "whisper_lora_loss.json")
with open(loss_json, "w") as f:
    json.dump(loss_data, f, indent=2, ensure_ascii=False)

print(f"Loss 數據: {loss_json}")

# 8. 畫 Loss 曲線
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang TC', 'sans-serif']
    
    steps = [e["step"] for e in loss_logger.losses]
    losses = [e["loss"] for e in loss_logger.losses]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color='#3498db', linewidth=2)
    ax.fill_between(steps, losses, alpha=0.1, color='#3498db')
    ax.set_xlabel('Training Step', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(f'Whisper-tiny + LoRA 微調 Loss 曲線（排灣語 ASR）\n{len(valid_data)} 筆語料 × 30 steps，耗時 {elapsed:.0f}s', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 標注起點和終點
    ax.annotate(f'起始 Loss: {losses[0]:.2f}', xy=(steps[0], losses[0]), 
                fontsize=11, fontweight='bold', color='#e74c3c')
    ax.annotate(f'最終 Loss: {losses[-1]:.2f}', xy=(steps[-1], losses[-1]),
                fontsize=11, fontweight='bold', color='#2ecc71')
    
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "whisper_lora_loss_curve.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Loss 曲線: {chart_path}")
    
except ImportError:
    print("matplotlib 未安裝，跳過圖表生成")

print("\n完成！")
