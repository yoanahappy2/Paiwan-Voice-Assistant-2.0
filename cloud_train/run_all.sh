#!/bin/bash
###############################################################################
# run_all.sh — 排灣語 XTTS v2 微調一鍵腳本
#
# 在清華 EasyCompute 容器雲 GPU 上執行
# 用法: bash run_all.sh
#
# 流程:
#   1. 環境檢查 (GPU, Python)
#   2. 安裝依賴 (TTS, torch, torchaudio)
#   3. 數據驗證 (50 筆排灣語音檔)
#   4. 微調訓練 (XTTS v2, ~1hr on 4090)
#   5. 測試推理 (生成所有詞彙 + 句子)
#
# 如果微調失敗，自動 fallback 到零樣本克隆（同樣可產出音檔）
###############################################################################

set -e

WORKSPACE="/workspace/paiwan_train"
PYTHON="python3"

echo "============================================================"
echo "  🌾 語聲同行 — 排灣語 XTTS v2 微調訓練"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---- Step 0: 環境 ----
echo ""
echo "[Step 0] 🔍 環境檢查"

if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    echo "  GPU: ${GPU}"
else
    echo "  ⚠️  未偵測到 GPU (nvidia-smi)"
fi

${PYTHON} --version

# ---- Step 1: 安裝依賴 ----
echo ""
echo "[Step 1] 📦 安裝依賴 (約 3-5 分鐘)"

pip install --upgrade pip --quiet

# PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet 2>/dev/null || \
pip install torch torchaudio --quiet

# TTS
pip install TTS pandas scikit-learn --quiet

# 驗證
${PYTHON} -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
${PYTHON} -c "from TTS.api import TTS; print('  ✓ TTS 庫就緒')"

# ---- Step 2: 解壓數據 ----
echo ""
echo "[Step 2] 📂 準備數據"

mkdir -p ${WORKSPACE}

# 如果 tar.gz 在當前目錄
if [ -f "paiwan_cloud_train.tar.gz" ]; then
    echo "  解壓 paiwan_cloud_train.tar.gz → ${WORKSPACE}"
    tar -xzf paiwan_cloud_train.tar.gz -C ${WORKSPACE}
elif [ -f "paiwan_cloud.tar.gz" ]; then
    echo "  解壓 paiwan_cloud.tar.gz → ${WORKSPACE}"
    tar -xzf paiwan_cloud.tar.gz -C ${WORKSPACE}
fi

# 驗證
if [ -d "${WORKSPACE}/dataset/wavs" ]; then
    WAV_COUNT=$(ls ${WORKSPACE}/dataset/wavs/*.wav 2>/dev/null | wc -l)
    echo "  ✓ ${WAV_COUNT} 個音檔"
else
    echo "  ❌ 找不到 dataset/wavs/"
    echo "  請手動上傳 dataset/ 到 ${WORKSPACE}/"
    exit 1
fi

# ---- Step 3: 訓練 ----
echo ""
echo "[Step 3] 🚀 開始訓練"
echo "  （這會花一些時間，請耐心等待）"
echo ""

cd ${WORKSPACE}
${PYTHON} train_paiwan.py \
    --dataset ${WORKSPACE}/dataset \
    --output ${WORKSPACE}/output \
    --epochs 50 \
    --method auto

# ---- Step 4: 測試 ----
echo ""
echo "[Step 4] 🎯 測試推理"

if [ -f "${WORKSPACE}/train_paiwan.py" ]; then
    ${PYTHON} ${WORKSPACE}/test_inference.py \
        ${WORKSPACE}/dataset \
        ${WORKSPACE}/output
fi

# ---- Done ----
echo ""
echo "============================================================"
echo "  🎉 全部完成！"
echo ""
echo "  輸出位置: ${WORKSPACE}/output/"
echo "  音檔位置: ${WORKSPACE}/output/paiwan_tts_outputs/"
echo ""
echo "  📥 下載到本地："
echo "  打包: cd ${WORKSPACE}/output && tar -czf paiwan_tts.tar.gz paiwan_tts_outputs/"
echo "  然後從 Web IDE 下載 paiwan_tts.tar.gz"
echo "============================================================"
