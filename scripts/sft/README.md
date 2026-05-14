# 排灣語 SFT 微調完整指南

本指南將幫助你從零開始完成智譜 GLM-4 模型的微調，讓模型學會排灣語的翻譯和語法規律。

## 📋 前置要求

- ✅ 智譜 API Key（你已經有）
- ✅ 排灣語平行語料（`data/merged_corpus.json`，~24,000 條）

## 🎯 預期效果

微調後的模型將能夠：
- 更準確地進行排灣語-中文雙向翻譯
- 理解排灣語的基本語法結構（詞序、時態等）
- 正確處理常見的詞綴變化

---

## 第一步：準備訓練數據

### 1.1 先分析語料（可選）

```bash
cd /Users/sbb-mei/Desktop/paiwan_competition_2026
python scripts/sft/prepare_sft_data.py --analyze
```

### 1.2 生成訓練數據

```bash
python scripts/sft/prepare_sft_data.py --output paiwan_sft.jsonl
```

生成的檔案：
- `scripts/sft/data/paiwan_sft_train.jsonl` - 訓練集（~90%）
- `scripts/sft/data/paiwan_sft_val.jsonl` - 驗證集（~10%）

### 1.3 驗證數據格式

```bash
python scripts/sft/run_sft.py --verify --stats
```

確保看到「✅ 格式正確！」的提示。

---

## 第二步：上傳到智譜平台進行微調

### 2.1 訪問智譜微調控制台

👉 https://open.bigmodel.cn/usercenter/fine-tuning

### 2.2 創建微調任務

1. 點擊「創建微調任務」
2. 選擇模型：**glm-4-flash**（免費）
3. 上傳訓練檔案：`paiwan_sft_train.jsonl`
4. （可選）上傳驗證檔案：`paiwan_sft_val.jsonl`
5. 設置參數：
   - 訓練輪次：3-5 epochs
   - 學習率：默認
   - Batch size：默認
6. 任務名稱：`paiwan-translator-v1`

### 2.3 等待訓練完成

- 預計時間：10-30 分鐘
- 可以在控制台實時查看訓練進度
- 完成後會收到通知

### 2.4 獲取微調模型 ID

訓練完成後，模型 ID 格式類似：
```
glm-4-flash-paiwan-20250512-123456
```

記下這個 ID，後續測試會用到。

---

## 第三步：測試微調模型

### 3.1 快速測試

```bash
# 測試單句
python scripts/sft/test_sft_model.py --model YOUR_MODEL_ID --single "masalu"

# 互動式測試
python scripts/sft/test_sft_model.py --model YOUR_MODEL_ID --interactive
```

### 3.2 完整對比測試

```bash
# 對比基礎模型和微調模型
python scripts/sft/test_sft_model.py --model glm-4-flash --finetuned YOUR_MODEL_ID
```

你會看到兩個模型的對比結果，包括：
- 翻譯準確率
- 各難度題目的表現
- 提升幅度百分比

---

## 第四步：整合到翻譯服務

微調模型驗證通過後，可以替換現有的 `glm-4-flash` 為微調模型：

```python
# 在 translate_service.py 中
MODEL_FAST = "glm-4-flash"  # 改為你的微調模型 ID
```

或者使用環境變量：

```bash
export PAIWAN_FINETUNED_MODEL="your-model-id"
```

---

## 📁 檔案結構

```
scripts/sft/
├── README.md              # 本指南
├── prepare_sft_data.py    # 數據準備腳本
├── run_sft.py             # 微調管理腳本
├── test_sft_model.py      # 測試腳本
└── data/                  # 訓練數據目錄
    ├── paiwan_sft_train.jsonl
    └── paiwan_sft_val.jsonl
```

---

## 🔧 常見問題

### Q1: 訓練需要多長時間？

**A:** 智譜 GLM-4-Flash 的微調通常在 10-30 分鐘內完成，取決於數據量。

### Q2: 訓練需要花錢嗎？

**A:** GLM-4-Flash 的微調目前是**免費**的。你只需要支付 API 調用費用（如果使用微調模型進行推理）。

### Q3: 數據格式有什麼要求？

**A:** 智譜要求的 JSONL 格式，每行一個 JSON 對象：

```json
{"messages": [
  {"role": "system", "content": "你是排灣語翻譯助手"},
  {"role": "user", "content": "請將「masalu」翻譯成中文"},
  {"role": "assistant", "content": "謝謝"}
]}
```

### Q4: 可以用其他模型嗎？

**A:** 可以！智譜還支持：
- `glm-4-plus` - 更強大，但需要付費
- `glm-4-air` - 性價比高

### Q5: 微調後如何部署？

**A:** 微調模型可以直接通過智譜 API 調用，無需自行部署。只需將模型 ID 傳給 `chat.completions.create` 即可。

---

## 📊 期待效果

根據你的 24,000 條平行語料，預期微調後：

| 指標 | 基礎模型 | 微調後 | 提升 |
|------|----------|--------|------|
| 常見詞彙翻譯 | ~70% | ~90%+ | +20% |
| 語法準確性 | ~60% | ~85%+ | +25% |
| 長句理解 | ~50% | ~75%+ | +25% |

---

## 🚀 下一步

1. ✅ 完成數據準備
2. ✅ 提交微調任務
3. ✅ 測試微調模型
4. ✅ 整合到 RAG 系統（SFT + RAG 混合）

微調完成後，我們可以進一步優化翻譯服務，讓 SFT 模型和 RAG 檢索協同工作！
