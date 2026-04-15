# 語聲同行 2.0 — 低資源瀕危語言的 AI 輔助教學框架

> **研究語言**：東排灣族語（Paiwan, ISO 639-3: pag）
> **研究問題**：如何在語料極度稀缺（<200 句）的條件下，構建完整的瀕危語言 AI 教學系統？

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 研究問題（Problem Statement）

全球超過 3000 種瀕危語言缺乏 AI 技術支持。排灣族語作為台灣原住民第二大族語，面臨三重技術困境：

1. **語料稀缺**：僅有數百筆標註語料，不足以訓練大規模模型
2. **LLM 語言能力缺失**：GPT-4、GLM-4 等主流 LLM 無法正確生成排灣語
3. **語音技術空白**：無商業 ASR/TTS 系統支持排灣語

本專案提出一套 **標準化的低資源語言 AI 教學管線**，並以排灣語為驗證案例。

---

## 技術貢獻（Technical Contributions）

### 1. ASR 微調 + 音韻規則補償方案
- Whisper-tiny + LoRA 微調，50 筆語料達 **81.05%** 準確率
- 三層音韻校正引擎（特定詞 / m-v 互換 / 音韻替換），校正成功率 **100%**
- 消融實驗驗證：微調 + 規則引擎 > 單獨微調

### 2. 「LLM 作為解釋者」範式
- LLM 不生成排灣語（生成不可控），只做中文教學解釋
- 所有排灣語內容 **100% 來自語料庫**，確保準確性
- 語料庫按語法結構分類（7 大類）+ 綴詞標註

### 3. 黏著語語料標準化方案
- 針對排灣語黏著語特性設計的綴詞體系映射
- 前綴時態標記（na/uri）+ 否定系統（ini/inika/maya）+ 語態標記
- 可複製至其他南島語系語言

### 4. 零樣本語音克隆應用於瀕危語言
- XTTS v2（518M 參數）零樣本語音克隆
- 用排灣族長老錄音作為參考音色
- 生成 50 詞彙 + 8 句子的排灣語語音

---

## 系統架構

```
┌──────────────────────────────────────────────────────────┐
│              低資源語言 AI 教學管線                         │
│                                                           │
│  Stage 1        Stage 2         Stage 3       Stage 4     │
│  語料構建        ASR 微調        語法分析       語音合成     │
│  ├ 結構化標註    ├ Whisper+LoRA  ├ 綴詞分析    ├ XTTS v2   │
│  ├ 語法分類      ├ 音韻校正      ├ 意圖分類    ├ 零樣本    │
│  └ 綴詞標註      └ 發音評測      └ LLM 解釋   └ 長老音色  │
│                                                           │
│           Stage 5: 整合評估                                │
│           消融實驗 │ 噪音容忍度 │ 可懂度評估                │
└──────────────────────────────────────────────────────────┘
```

---

## 實驗結果

### ASR 管線消融實驗

| 配置 | 準確率 | 提升 |
|------|--------|------|
| Whisper-tiny (Baseline) | ~30% | — |
| + LoRA 微調 | **81.05%** | +51% |
| + LoRA + 音韻校正 | **~90%** (估) | +9% |

### 音韻校正引擎

| 錯誤類型 | 校正成功率 |
|----------|-----------|
| m/v 混淆 | 100% |
| 常見辨識錯誤 | 100% |
| 整體 | **100%** |

### 噪音容忍度

| 噪音等級 | 音韻相似度 |
|----------|-----------|
| 無噪音 | 100.0% |
| 輕微（1字替換） | 96.2% |
| 中度（2字替換） | 94.5% |
| 嚴重（3字替換） | 90.0% |

### 系統性能

| 指標 | 數值 |
|------|------|
| ASR 準確率（LoRA） | 81.05% |
| 音韻校正成功率 | 100% |
| 意圖分類覆蓋率 | 95.2% |
| 綴詞偵測率 | 52.4% |
| 語料自比對 100 分率 | 100% |

---

## 專案結構

```
paiwan_competition_2026/
├── app.py                          # Gradio 主介面（4 個功能 Tab）
├── modules/
│   ├── asr_evaluator.py            # 音韻規則引擎 + 綴詞分析器 + 意圖分類
│   ├── grammar_explainer.py        # LLM 語法解釋器（只解釋不生成）
│   └── paiwan_corpus.json          # 結構化排灣語語料庫（7 類 42 句）
├── paiwan_tts/                     # XTTS v2 零樣本克隆語音
│   ├── *.wav                       # 50 個排灣語詞彙語音
│   └── sentences/                  # 8 個排灣語句子語音
├── tests/
│   ├── ablation_study.py           # 消融實驗腳本
│   └── generate_charts.py          # 圖表生成
├── docs/
│   ├── ablation_results.json       # 實驗數據
│   └── charts/                     # 可視化圖表
├── FRAMEWORK_METHODOLOGY.md        # 技術框架文件（方法論）
├── cloud_train/                    # 清華算力平台訓練腳本
└── README.md                       # 本文件
```

---

## 快速開始

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動 Web 介面
python app.py

# 跑消融實驗
python tests/ablation_study.py

# 生成圖表
python tests/generate_charts.py
```

**注意**：ASR 功能需要原專案後端運行在 `localhost:8000`。其他功能（語法解釋、語料庫瀏覽、測驗）可獨立使用。

---

## 技術棧

| 模組 | 技術 | 說明 |
|------|------|------|
| ASR | OpenAI Whisper-tiny + LoRA | 50 筆語料微調，81.05% |
| 音韻引擎 | 三層規則校正 | 特定詞 / m-v 互換 / 音韻替換 |
| 綴詞分析 | 排灣語前綴映射 | 時態 / 否定 / 語態 |
| 教學解釋 | 智譜 GLM-4-Flash | 只做中文解釋，不生成排灣語 |
| TTS | Coqui XTTS v2 | 零樣本克隆，長老音色 |
| 前端 | Gradio 4.x | 4 個功能 Tab |
| 訓練平台 | 清華 EasyCompute | NVIDIA RTX 4090 |

---

## 可複製性

本框架設計為可複製到其他低資源語言：

1. **語料標準化**（Stage 1）適用於任何黏著語
2. **音韻規則引擎**可替換為目標語言的規則
3. **XTTS 零樣本克隆**不依賴目標語言的 TTS 訓練數據
4. **LLM 解釋範式**適用於任何 LLM 不懂的語言

潛在推廣目標：阿美語、泰雅語、布農語等台灣原住民語言。

---

## 作者

**黃詠郁**（清華大學，學號 2026110008）

- 畢業專題：排灣族語 ASR 系統（Whisper + LoRA）
- 競賽：飛書 AI 校園挑戰賽 2026 — 開放創新賽道

---

## 參考文獻

1. Radford, A. et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision."
2. Hu, E.J. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models."
3. Casanova, E. et al. (2022). "XTTS: a massively multilingual zero-shot text-to-speech model."
4. 原住民族語言研究中心. "排灣族語言參考語法."
5. Li, P.J. (2004). "Selected Papers on Formosan Languages."

---

*語聲同行 2.0 | 2026 | 低資源語言 AI 教學框架*
