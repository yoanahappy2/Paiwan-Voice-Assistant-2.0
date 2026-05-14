#!/bin/bash
#
# submit_cluster.sh
# 清華 GPU 集群任務提交腳本
#
# 使用方式：
#   sbatch submit_cluster.sh          # 使用默認配置
#   sbatch submit_cluster.sh --qos=h  # 使用高優先級隊列
#

#SBATCH --job-name=paiwan-sft                    # 任務名稱
#SBATCH --partition=gpu                          # GPU 分區
#SBATCH --nodes=1                                # 節點數
#SBATCH --ntasks-per-node=1                      # 每節點任務數
#SBATCH --cpus-per-task=8                        # CPU 核心數
#SBATCH --gres=gpu:1                             # GPU 數量
#SBATCH --mem=32G                                # 內存
#SBATCH --time=2:00:00                           # 最大運行時間
#SBATCH --output=logs/sft-%j.out                 # 標準輸出
#SBATCH --error=logs/sft-%j.err                  # 錯誤輸出

# ============================================
# 環境設置
# ============================================

# 加載模組（根據集群配置調整）
module load cuda/12.1
module load python/3.10

# 或使用 conda
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate paiwan-sft

# 設置 HF 鏡像（加速下載）
export HF_ENDPOINT=https://hf-mirror.com

# 設置緩存目錄
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/hf_home

# ============================================
# 創建日誌目錄
# ============================================

mkdir -p logs

# ============================================
# 啟動訓練
# ============================================

echo "========================================="
echo "開始訓練: $(date)"
echo "節點: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# 運行訓練
# 單 GPU
python scripts/sft/train_on_cluster.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --batch-size 4 \
    --output-dir scripts/sft/checkpoints

# 或多 GPU (如果有 4 塊 GPU)
# torchrun --nproc_per_node=4 scripts/sft/train_on_cluster.py \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --epochs 3 \
#     --batch-size 2 \
#     --output-dir scripts/sft/checkpoints

echo "========================================="
echo "訓練完成: $(date)"
echo "========================================="
