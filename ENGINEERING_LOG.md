# 語聲同行 2.0 — 工程紀錄

> 本文件記錄關鍵技術決策、架構設計、已知問題與解法

---

## 1. 系統架構

```
┌─────────────────────────────────────────────────────┐
│                 用戶介面 (Gradio)                      │
│          http://127.0.0.1:7860                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  🎤 錄音/上傳音檔  ──→  ASR Service                 │
│                            │                        │
│                   Whisper-tiny + LoRA               │
│                   (排灣語微調模型)                     │
│                   81.05% 準確率                      │
│                            │                        │
│                            ▼                        │
│                     辨識文字                         │
│                            │                        │
│                            ▼                        │
│                     LLM Service                     │
│                   智譜 GLM-4-Flash                   │
│                   (vuvu Maliq 角色對話)               │
│                   + 117 句排灣語語料                  │
│                   + Chain-of-Thought 推理            │
│                            │                        │
│                            ▼                        │
│                     雙語回覆                         │
│                   (排灣語 + 中文)                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 端口服用的

| 服務 | Port | 說明 |
|------|------|------|
| Gradio Web Demo | 7860 | 比賽用前端介面 |
| FastAPI ASR 後端 | 8000 | 原專案 ASR API（唯讀呼叫） |

---

## 2. 檔案結構

```
paiwan_competition_2026/
├── .env                        # 智譜 API Key (不入 Git)
├── .gitignore
├── requirements_comp.txt       # 獨立依賴清單
├── CHANGELOG.md                # 變更紀錄
├── DAILY_LOG.md                # 每日開發日誌
├── ENGINEERING_LOG.md          # 本文件：工程紀錄
├── README.md                   # 使用說明
├── PROJECT_INFO.md             # 競賽專題說明
│
├── app.py                      # Gradio Web Demo (主入口)
├── voice_chat.py               # Voice-to-Voice 全鏈路
├── llm_service.py              # LLM 對話服務 + System Prompt
├── paiwan_phrases.txt          # 排灣語語料清單 (117 句)
│
├── whisper-lora-paiwan-final/  # Whisper LoRA 權重 (6MB)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── api/                        # 從原專案複製的後端
│   ├── server.py               # FastAPI (10 個端點)
│   ├── asr_service.py          # Whisper LoRA ASR
│   └── data/
│       ├── keyword_weights.json
│       ├── intent_training.jsonl
│       ├── game_vocabulary.json
│       └── ...
│
├── venv/                       # 獨立虛擬環境 (不入 Git)
│
└── 族語彙整（固定格式） - 正式版本 .csv
```

---

## 3. 關鍵技術決策

### 3.1 為什麼用智譜而不是火山方舟？
- 火山方舟 API 要入營後才開放
- 智譜現在就能用，API 格式 OpenAI 兼容
- 入營後可以無縫切換到豆包（同樣是 OpenAI 格式）

### 3.2 為什麼 ASR 用 HTTP 呼叫而不直接載入模型？
- 原專案的 FastAPI 已經在 `localhost:8000` 運行
- 直接載入 Whisper 模型需要 ~1.8GB 記憶體
- HTTP 呼叫是 read-only，不會修改原專案
- 延遲可接受（ASR 推理 <1 秒）

### 3.3 為什麼用 Gradio 而不是自己寫前端？
- 開發速度快（30-50 行代碼就有完整介面）
- 內建錄音元件，不用處理 WebRTC
- `share=True` 可以產生公開連結（分享給評審）
- 原專案已有 React 前端，但比賽不需要那麼複雜

### 3.4 Chain-of-Thought 設計
- System Prompt 要求先輸出推理過程（匹配結果、信心度、策略）
- 再輸出最終回覆
- 程式碼解析 `🤔...EqualTo` 或 ` ```think...``` ` 標記
- 對話歷史只記錄最終回覆（不記錄思考，節省 context）

---

## 4. 已知問題

| 問題 | 等級 | 狀態 | 備註 |
|------|------|------|------|
| VPN 阻擋 GitHub push | 🟡 中 | 已解決 | 設 http.proxy 為 127.0.0.1:7897 |
| Gradio share link 未生成 | 🟡 中 | 暫緩 | VPN 環境下可能無法生成，Demo 時再處理 |
| TTS 語音回覆未完成 | 🟡 中 | 待做 | 目前只有文字回覆，需串接 XTTS v2 |
| vuvu 排灣語知識有限 | 🟡 中 | 持續改善 | 117 句語料已注入，但長尾場景仍可能幻覺 |
| ASR 辨識準確度 | 🟢 低 | 已知 | Whisper-tiny + LoRA, 81.05% WER |

---

## 5. 環境變數

```bash
# .env 檔案內容
ZHIPUAI_API_KEY=<你的智譜 API Key>
```

---

## 6. Git 配置注意事項

- Git proxy 設為 Clash Verge: `http://127.0.0.1:7897`
- 如果 push 失敗，檢查 VPN 是否開啟
- 如果不開 VPN，需要取消 proxy:
  ```bash
  git config --global --unset http.proxy
  git config --global --unset https.proxy
  ```
