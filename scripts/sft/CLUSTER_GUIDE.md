# 清華 GPU 集群 SFT 訓練指南

本指南適用於在清華大學 GPU 集群上進行排灣語模型微調。

## 📋 前置準備

### 1. 檢查集群環境

```bash
# 查看可用 GPU
sinfo -p gpu

# 查看隊列狀態
squeue

# 測試 Python 環境
python --version
```

### 2. 安裝依賴

```bash
# 創建虛擬環境（推薦）
python -m venv paiwan-sft-env
source paiwan-sft-env/bin/activate

# 安裝依賴
pip install -r scripts/sft/requirements_cluster.txt
```

## 🚀 快速開始

### 方法 1：交互式訓練（適合調試）

```bash
# 申請交互式 GPU 節點
srun --partition=gpu --gres=gpu:1 --pty bash

# 載入模組
module load cuda/12.1
module load python/3.10

# 激活環境
source paiwan-sft-env/bin/activate

# 運行訓練
python scripts/sft/train_on_cluster.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --batch-size 4
```

### 方法 2：批量任務（推薦）

```bash
# 提交任務
sbatch scripts/sft/submit_cluster.sh

# 查看任務狀態
squeue -u $USER

# 查看日誌
tail -f logs/sft-<jobid>.out
```

## ⚙️ 配置說明

### 模型選擇

| 模型 | 參數量 | 顯存需求 | 訓練時間 | 推薦場景 |
|------|--------|----------|----------|----------|
| Qwen2.5-7B-Instruct | 7B | ~16GB | ~30min | **推薦**，性價比最高 |
| Qwen2.5-14B-Instruct | 14B | ~32GB | ~60min | 更強能力 |
| Llama-3-8B-Instruct | 8B | ~16GB | ~40min | 英文能力強 |

### LoRA 配置

```bash
# 默認配置（推薦）
--lora-r 16          # Rank，越高越靈活但參數越多
--lora-alpha 32      # Alpha，通常 = 2 * r

# 顯存不足時
--lora-r 8
--batch-size 2

# 追求效果時
--lora-r 32
--batch-size 8
```

### 量化選項

```bash
# 4bit 量化（推薦，顯存需求最低）
# 默認啟用，無需額外參數

# 不使用量化（需要更多顯存）
--no-4bit
```

## 📊 監控訓練

### TensorBoard

```bash
# 在集群上啟動 TensorBoard
tensorboard --logdir scripts/sft/checkpoints --port 6006 --bind_all

# 在本地建立 SSH 隧道
ssh -L 6006:localhost:6006 user@cluster.tsinghua.edu.cn

# 在瀏覽器打開
# http://localhost:6006
```

### 日誌監控

```bash
# 實時查看輸出
tail -f logs/sft-<jobid>.out

# 查看錯誤
tail -f logs/sft-<jobid>.err
```

## 🔧 常見問題

### Q1: 顯存不足 (OOM)

**解決方案：**
```bash
# 減小 batch size
--batch-size 2

# 使用 4bit 量化
# 默認已啟用

# 使用梯度累積
--gradient-accumulation-steps 8
```

### Q2: 訓練速度慢

**解決方案：**
```bash
# 使用多 GPU
torchrun --nproc_per_node=4 scripts/sft/train_on_cluster.py

# 增加 batch size
--batch-size 8

# 使用 DeepSpeed
deepspeed --num_gpus=4 scripts/sft/train_on_cluster.py
```

### Q3: 模型下載失敗

**解決方案：**
```bash
# 設置 HF 鏡像
export HF_ENDPOINT=https://hf-mirror.com

# 或手動下載後上傳
# https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct
```

## 📁 預期輸出

訓練完成後，檔案結構：

```
scripts/sft/checkpoints/
├── checkpoint-100/          # 每 100 步保存一次
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── checkpoint-200/
├── final_model/             # 最終模型
│   ├── adapter_model.safetensors
│   ├── config.json
│   └── tokenizer_config.json
└── runs/                    # TensorBoard 日誌
    └── ...
```

## 🎯 下一步

訓練完成後，使用 `test_sft_model.py` 測試效果：

```bash
python scripts/sft/test_sft_model.py \
    --model scripts/sft/checkpoints/final_model \
    --interactive
```

## 📞 支持與資源

- 清華算力平台文檔：參考 `清華算力相關/` 目錄下的 PDF
- Qwen 模型文檔：https://github.com/QwenLM/Qwen2.5
- Transformers 文檔：https://huggingface.co/docs/transformers

## 🔄 與智譜 API 方案對比

| 特性 | 清華集群 | 智譜 API |
|------|----------|----------|
| 成本 | 免費（有算力券） | 免費（Flash） |
| 靈活性 | 高（可調參數） | 低（固定配置） |
| 模型選擇 | 開源模型任意 | 智譜系列 |
| 部署 | 需自行部署 | API 直接調用 |
| 數據隱私 | 本地處理 | 上傳雲端 |

**推薦：** 如果追求效果和靈活性，用清華集群；如果追求簡單方便，用智譜 API。
