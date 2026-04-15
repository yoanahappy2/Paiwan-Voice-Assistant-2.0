# 語聲同行 2.0 — 排灣族語 Voice-to-Voice AI 語伴

> 飛書 AI 校園挑戰賽 2026（開放創新賽道）
>
> 為瀕危語言建立可互動的學習夥伴，讓每個排灣族孩子都能用母語跟 AI vuvu 對話。

---

## 技術亮點

| 模組 | 技術 | 數據 |
|------|------|------|
| ASR 語音辨識 | Whisper-tiny + LoRA 微調 | 81.05% 準確率 |
| RAG 語意檢索 | 智譜 embedding-3 + FAISS | Top-3 準確率 90% |
| LLM 對話 | 智譜 GLM-4-Flash | RAG 命中率 100% |
| 抗幻覺 | Chain-of-Thought | 低信心時自動標注 |
| 語料規模 | 118 句排灣語 | 東排灣方言 |

---

## 快速啟動

### 前置條件

1. **原專案 ASR 服務必須在運行**（提供 `/api/audio/recognize` 端點）
2. **智譜 API Key** 已填入 `.env`

### 啟動步驟

```bash
# 1. 進入比賽目錄
cd ~/Desktop/paiwan_competition_2026

# 2. 啟動虛擬環境
source venv/bin/activate

# 3. 確認原專案 ASR 服務正在運行
curl http://127.0.0.1:8000/health
# 應該看到: {"status":"ok",...}

# 4. 啟動 Gradio Demo
python app.py
# 瀏覽器打開 http://127.0.0.1:7860
```

### 如果 ASR 服務沒在跑

```bash
# 在另一個終端視窗啟動原專案後端
cd /Users/sbb-mei/paiwan_asr_project
source venv/bin/activate
python -m uvicorn api.server:app --reload --port 8000
```

---

## 三種使用模式

### 模式 1: Web 介面（推薦）
```bash
python app.py
# 打開 http://127.0.0.1:7860
```
- 🎤 錄音按鈕：按住錄排灣語，放開後點「送出語音」
- ⚡ 場景按鈕：一鍵觸發預設對話
- ⌨️ 文字輸入：直接打字跟 vuvu 聊天

### 模式 2: 終端機語音對話
```bash
python voice_chat.py
# 輸入音檔路徑進行語音辨識，或直接輸入文字
```

### 模式 3: 純文字對話（最快）
```bash
python llm_service.py
# 直接在終端打字跟 vuvu Maliq 聊天
```

---

## 指令說明

所有終端模式都支援以下指令：

| 指令 | 功能 |
|------|------|
| `/reset` | 重置對話（清除歷史） |
| `/quit` | 離開程式 |
| `/history` | 顯示對話歷史 |
| `/model` | 切換模型（flash ↔ plus） |

---

## 預設場景

Web 介面的 6 個場景按鈕：

| 按鈕 | 排灣語 | 中文意思 |
|------|--------|---------|
| 🏠 問候 | tjanu en | 你好嗎 |
| 🙏 謝謝 | masalu | 謝謝 |
| 👍 好 | nanguaq | 好 |
| 👋 再見 | pacunan | 再見 |
| 📖 說故事 | （中文） | 請 vuvu 講排灣族故事 |
| 📚 學單字 | （中文） | 請 vuvu 教排灣語日常用語 |

---

## 模型切換

| 模型 | 用途 | 速度 | 品質 |
|------|------|------|------|
| `glm-4-flash` | 日常開發測試 | ⚡ 快 | 一般 |
| `glm-4-plus` | Demo 錄製 | 🐢 慢 | 最好 |

在終端模式輸入 `/model` 切換。
在代碼中修改 `MODEL_DEFAULT` 變數。

---

## 推送到 GitHub

```bash
cd ~/Desktop/paiwan_competition_2026
git add -A
git commit -m "描述你的修改"
git push origin main
```

注意：需要開啟 Clash Verge VPN（已設 proxy 為 127.0.0.1:7897）

---

## 目錄結構

```
paiwan_competition_2026/
├── app.py                  # Gradio Web Demo（主入口）
├── llm_service.py          # LLM 對話服務（RAG 版）
├── rag_service.py          # RAG 檢索服務（embedding + FAISS）
├── voice_chat.py           # Voice-to-Voice 全鏈路
├── paiwan_phrases.txt      # 排灣語語料（118 句）
├── data/                   # RAG 向量索引快取
│   ├── faiss_index.bin
│   ├── corpus_metadata.json
│   └── paiwan_embeddings.npy
├── api/                    # ASR 後端
├── feishu_bot/             # 飛書 Bot 原型
│   ├── bot.py
│   └── ARCHITECTURE.md
├── tests/                  # 評測腳本
│   ├── asr_benchmark.py
│   └── results/
├── docs/                   # 技術文件
│   └── RAG_DESIGN.md
├── BENCHMARK.md            # 技術評測報告
├── ROADMAP.md              # 競賽優化路線圖
└── PROJECT_INFO.md         # 專題說明
```

---

## 飛書整合計畫

本專案已設計飛書機器人架構，入營後可快速部署：

- **架構**: 飛書 Webhook → Bot 後端 → RAG + LLM → 回覆
- **原型程式碼**: `feishu_bot/bot.py`
- **架構設計**: `feishu_bot/ARCHITECTURE.md`
- **擴展功能**: 每日排灣語推送、互動測驗、語音教學

---

## 目錄保護規則

- ✅ 可以修改: `~/Desktop/paiwan_competition_2026/` 內的所有檔案
- 🚫 禁止修改: `/Users/sbb-mei/paiwan_asr_project/` 內的任何檔案
- 原專案 ASR 服務透過 HTTP 呼叫（read-only）
