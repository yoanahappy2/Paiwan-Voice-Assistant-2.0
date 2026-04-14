# 語聲同行 2.0 — 競賽版開發日誌

> 專案路徑: `~/Desktop/paiwan_competition_2026/`
> 原專案路徑: `/Users/sbb-mei/paiwan_asr_project/` （禁止修改）
> GitHub: https://github.com/yoanahappy2/Paiwan-Voice-Assistant-2.0

---

## 2026-04-14（Day 1）

### 概要
從零建立競賽沙盒環境，完成 Phase 0 到 Phase 3，跑通 Voice-to-Voice 全鏈路。

### 工作時段: 21:45 - 00:33

---

### Phase 0: 沙盒隔離（22:34-22:44）

**目的**: 保護原專案不被修改，所有比賽開發在獨立目錄進行

**操作**:
- 建立 `~/Desktop/paiwan_competition_2026/`
- 從原專案複製（非移動）:
  - `whisper-lora-paiwan-final/` (6MB, Whisper LoRA 權重)
  - `api/` (FastAPI 後端: server.py, asr_service.py, data/)
  - `族語彙整...csv` (語料)
- 建立 `.env`（智譜 API Key）、`.gitignore`、`requirements_comp.txt`
- 建立 `CHANGELOG.md` 變更紀錄
- 驗證: asr_service.py 使用相對路徑，新目錄下正確解析

**關鍵決策**: 使用複製而非 symlink，確保原專案零風險

---

### Phase 1: LLM 串接（22:44-23:25）

**目的**: 建立 vuvu Maliq 排灣族語 AI 語伴的 LLM 對話層

**技術選型**:
- LLM: 智譜 GLM-4-Flash（開發用，快速低成本）
- 備選: GLM-4-Plus（Demo 錄製時切換，效果最好）
- API: OpenAI 兼容格式，base_url = `https://open.bigmodel.cn/api/paas/v4`

**產出**: `llm_service.py`
- `VuvuService` 類別：管理對話歷史、切換模型
- `SYSTEM_PROMPT`：vuvu Maliq 角色設定（慈祥排灣族祖母）
- `main()`：終端機文字對話測試

**測試結果**:
| 輸入 | vuvu 回覆 | 驗證 |
|------|----------|------|
| `masalu` | "nanguaq masalu, ai~~ vuvu 高興..." | ✅ 排灣語優先 |
| `你好` | "nanguaq! 來，跟 vuvu 說一次 'tjanu en'" | ✅ 鼓勵學排灣語 |
| `nanguaq` | "nanguaq! 你好啊！" | ✅ 雙語教學 |

---

### Phase 2: Voice-to-Voice 串接（23:25-23:42）

**目的**: 串接 ASR（語音辨識）和 LLM（對話生成）

**技術架構**:
```
音檔 → FastAPI /api/audio/recognize (原專案, port 8000)
     → Whisper LoRA ASR (排灣語微調模型)
     → 辨識文字 → 智譜 GLM-4-Flash
     → vuvu 排灣語+中文回覆
```

**產出**: `voice_chat.py`
- `recognize_audio()`: 呼叫原專案 ASR API
- `voice_chat()`: 完整 ASR → LLM 流程
- `main()`: 終端機語音+文字測試

**關鍵**: ASR 服務透過 HTTP 呼叫原專案（read-only），不依賴本地模型載入

**測試結果**:
- `s01-akumaya.m4a` → ASR: "akumaya" (信心度 1.00) → vuvu 雙語回覆 ✅

---

### Git & GitHub 設置（23:42-00:04）

**問題**: VPN (Clash Verge, port 7897) 阻擋 SSH 和 HTTPS 連線

**解法**:
```bash
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

**GitHub 倉庫**: `https://github.com/yoanahappy2/Paiwan-Voice-Assistant-2.0`

---

### Phase 3: Gradio Web 介面（00:04-00:26）

**目的**: 建立可分享的 Web Demo 介面

**產出**: `app.py`
- Gradio Blocks 介面
- 錄音/上傳音檔 + 文字輸入
- 6 個預設場景按鈕（問候/謝謝/好/再見/說故事/學單字）
- 本地運行 `http://127.0.0.1:7860`

**問題**: `share=True` 因 VPN 未生成公開連結，暫用 `share=False` 本地測試

---

### 語料注入（00:22-00:26）

**目的**: 解決 vuvu 不認識排灣語句子的問題

**操作**:
- 從 `intent_training.jsonl` 提取 117 句不重複排灣語句子
- 注入 SYSTEM_PROMPT 的詞彙區塊

**測試**:
- `pida qadaw nu isiukang?`（一個星期有幾天？）→ 修復前: 誤以為 pida 是「請」→ 修復後: 正確理解 ✅

---

### Chain-of-Thought 抗幻覺機制（00:26-00:32）

**目的**: 讓 vuvu 展示推理過程，降低幻覺

**修改**:
- System Prompt 要求結構化輸出（推理 + 回覆分離）
- 新增 `chat_with_thinking()` 方法
- Gradio 介面新增「🧠 vuvu 的思考過程」區塊
- 溫度從 0.7 降到 0.5

**展示格式**:
```
🧠 思考: 匹配結果、信心度、回覆策略
👵 回覆: 排灣語 + 中文雙語
```

---

### 今日 Commit 歷史
```
7acf4c7 feat: 增加 chain-of-thought 推理過程展示
1643540 feat: 注入 117 句排灣語真實語料到 vuvu System Prompt
562385b feat: Phase 3 Gradio Web Demo 介面
163e4ea feat: Phase 2 Voice-to-Voice 全鏈路串接
b6aa1fe feat: Phase 0+1 初始化競賽沙盒環境
```

---

### 待辦（4/15-4/17）
- [ ] 測試更多排灣語句子，調優 System Prompt
- [ ] 錄製 Demo 影片（60-90 秒）
- [ ] 填寫飛書 AI 校園挑戰賽報名表
- [ ] 考慮加入飛書 Bot 整合（加分項）
- [ ] TTS 語音回覆（目前只有文字）
