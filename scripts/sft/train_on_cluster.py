#!/usr/bin/env python3
"""
train_on_cluster.py
在清華 GPU 集群上進行排灣語 SFT 微調

支持：
- Qwen2.5-7B-Instruct（推薦，開源免費）
- Llama-3-8B-Instruct
- 使用 LoRA 技術，顯存需求低

用法：
    # 單 GPU
    python train_on_cluster.py --model Qwen/Qwen2.5-7B-Instruct

    # 多 GPU (ddp)
    torchrun --nproc_per_node=4 train_on_cluster.py --model Qwen/Qwen2.5-7B-Instruct

    # DeepSpeed
    deepspeed --num_gpus=4 train_on_cluster.py --deepspeed ds_config.json

作者: Claude
日期: 2026-05-12
"""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "scripts" / "sft" / "data"
OUTPUT_DIR = BASE_DIR / "scripts" / "sft" / "checkpoints"


# ============================================
# 配置
# ============================================

@dataclass
class SFTConfig:
    """SFT 訓練配置"""

    # 模型
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # 推薦：Qwen2.5-7B
    trust_remote_code: bool = True

    # 數據
    train_file: str = str(DATA_DIR / "paiwan_sft_train.jsonl")
    val_file: str = str(DATA_DIR / "paiwan_sft_val.jsonl")
    max_length: int = 512

    # LoRA 配置
    use_lora: bool = True
    lora_r: int = 16          # LoRA rank
    lora_alpha: int = 32      # LoRA alpha
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",     # FFN
    ])

    # 訓練參數
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    # 量化（降低顯存）
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # 輸出
    output_dir: str = str(OUTPUT_DIR)
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # 其他
    seed: int = 42
    local_rank: int = -1


# ============================================
# 數據處理
# ============================================

def load_jsonl(file_path: str) -> List[Dict]:
    """載入 JSONL 檔案"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_conversation(messages: List[Dict], tokenizer: AutoTokenizer) -> str:
    """
    將 messages 格式化為模型輸入

    Qwen 格式：
    <|im_start|>system
    你是排灣語翻譯助手<|im_end|>
    <|im_start|>user
    請將「masalu」翻譯成中文<|im_end|>
    <|im_start|>assistant
    謝謝<|im_end|>
    """
    # Qwen2.5 使用 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return text


class PaiwanDataset:
    """排灣語 SFT 數據集"""

    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item.get("messages", [])

        # 格式化為對話格式
        text = format_conversation(messages, self.tokenizer)

        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        # 構造 labels（只計算 assistant 部分的 loss）
        labels = encodings["input_ids"].copy()

        # 找到 assistant 回應的位置
        # 簡單做法：只計算最後 assistant 的 loss
        assistant_start = text.rfind("<|im_start|>assistant")
        if assistant_start != -1:
            # 計算 assistant 開始的 token 位置
            assistant_text = text[assistant_start:]
            assistant_tokens = self.tokenizer(assistant_text)["input_ids"]
            assistant_start_idx = len(labels) - len(assistant_tokens)

            # 將 assistant 之前的 labels 設為 -100（忽略）
            labels[:assistant_start_idx] = [-100] * assistant_start_idx

        encodings["labels"] = labels

        return encodings


def load_datasets(config: SFTConfig, tokenizer: AutoTokenizer):
    """載入訓練和驗證數據集"""
    print("📖 載入數據集...")

    train_data = load_jsonl(config.train_file)
    val_data = load_jsonl(config.val_file)

    print(f"  - 訓練集: {len(train_data)} 樣本")
    print(f"  - 驗證集: {len(val_data)} 樣本")

    train_dataset = PaiwanDataset(train_data, tokenizer, config.max_length)
    val_dataset = PaiwanDataset(val_data, tokenizer, config.max_length)

    return train_dataset, val_dataset


# ============================================
# 模型準備
# ============================================

def load_model_and_tokenizer(config: SFTConfig):
    """載入模型和 tokenizer"""
    print(f"🔧 載入模型: {config.model_name}")

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code
    )

    # 設置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  - 詞表大小: {len(tokenizer)}")

    # 載入模型
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    # 4bit 量化
    if config.use_4bit:
        print("  - 使用 4bit 量化")
        from bitsandbytes import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )

    print(f"  - 模型載入完成")

    # 準備 LoRA
    if config.use_lora:
        print("🔧 配置 LoRA...")
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )

        model = get_peft_model(model, lora_config)

        # 打印可訓練參數
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - 可訓練參數: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    return model, tokenizer


# ============================================
# 訓練
# ============================================

def train(config: SFTConfig):
    """執行訓練"""
    print("=" * 60)
    print("🚀 開始 SFT 訓練")
    print("=" * 60)

    # 載入模型
    model, tokenizer = load_model_and_tokenizer(config)

    # 載入數據
    train_dataset, val_dataset = load_datasets(config, tokenizer)

    # 訓練參數
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=3,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        logging_first_step=True,
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],
        seed=config.seed,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 訓練
    print("\n🏋️ 開始訓練...")
    trainer.train()

    # 保存最終模型
    print("\n💾 保存模型...")
    final_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"✅ 模型已保存到: {final_path}")

    return trainer


# ============================================
# 主函數
# ============================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="排灣語 SFT 訓練")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="基礎模型名稱")
    parser.add_argument("--train-file", type=str,
                        default=str(DATA_DIR / "paiwan_sft_train.jsonl"))
    parser.add_argument("--val-file", type=str,
                        default=str(DATA_DIR / "paiwan_sft_val.jsonl"))
    parser.add_argument("--output-dir", type=str,
                        default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--no-4bit", action="store_true",
                        help="不使用 4bit 量化")
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    # 配置
    config = SFTConfig(
        model_name=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        use_4bit=not args.no_4bit,
        local_rank=args.local_rank,
    )

    # 創建輸出目錄
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 執行訓練
    train(config)


if __name__ == "__main__":
    main()
