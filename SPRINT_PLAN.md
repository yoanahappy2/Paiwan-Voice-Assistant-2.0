# 語聲同行 2.0 — 復賽 11 天衝刺計劃

> 建立日期：2026-04-25
> 最後更新：2026-04-25 22:00
> 執行者：黃詠郁 + 地陪
> 目標：5/6 前完成復賽所有投稿物

---

## 🎯 復賽勝利條件

1. **飛書 Bot 能現場演示** — 評審掃碼就能跟 vuvu Maliq 對話
2. **Benchmark 數據量化** — 每個技術環節有數字證明增益
3. **真實用戶反饋** — 排灣族朋友測試過，有截圖引述
4. **Demo 影片有感動力** — 60-90 秒，有故事線
5. **TTS 品質過關** — 15 句標準音檔，不出 bug

---

## 📊 現狀盤點

### 語料資產
| 來源 | 數量 | 格式 |
|------|------|------|
| paiwan_corpus.json | 42 句（7 類） | 結構化 JSON |
| paiwan_phrases.txt | 118 句 | 文本 |
| 錄音檔（audio/） | 173 筆 | WAV |
| RAG 語料（HF 版） | 117 句 | FAISS + embedding |
| CSV（新版） | 159 行 | 含 s03 系列 |

### 已完成模組
| 模組 | 狀態 | 位置 |
|------|------|------|
| ASR（Whisper+LoRA） | ✅ 81% | 原專案，本地 API :8000 |
| 音韻校正引擎 | ✅ 三層 | modules/asr_evaluator.py |
| 綴詞分析器 | ✅ 11種前綴 | modules/grammar_explainer.py |
| RAG（FAISS） | ✅ Top-3 90% | rag_service.py |
| LLM（GLM-4-Flash） | ✅ vuvu 角色 | llm_service.py |
| 意圖分類 | ✅ 95.2% | 12類 |
| TTS（XTTS v2） | ⚠️ 有 bug | cloud_train/regenerate_tts.py |
| 飛書 Bot | 🔲 原型 | feishu_bot/bot.py（骨架） |
| HF Spaces | ✅ 運行中 | hf_deploy/ |

### 缺失項目
- ❌ Benchmark 測試集（沒有獨立測試集）
- ❌ 飛書 Bot 真正可用（只有骨架）
- ❌ TTS 修正版音檔
- ❌ 用戶測試數據
- ❌ Demo 影片
- ❌ 復賽技術文件

---

## 📅 每日衝刺排程

### Phase 1：基礎建設（4/26-4/28）

#### Day 1 — 4/26（六）Benchmark 建置

**目標：** 建立 Benchmark 測試集 + 跑完主流模型 baseline

| 任務 | 產出 | 驗收標準 |
|------|------|----------|
| 從 173 筆語料抽出 40 筆做測試集 | `benchmark/test_set.json` | 40 組中排對照，覆蓋 7 類別 |
| 定義評測指標 | `benchmark/metrics.py` | WER（ASR）、BLEU（翻譯）、語法正確率 |
| 測試 GLM-4-Flash 零樣本翻譯 | `benchmark/results/glm4flash_baseline.json` | 記錄 BLEU 分數 |
| 測試 GLM-4-Plus 零樣本翻譯 | `benchmark/results/glm4plus_baseline.json` | 記錄 BLEU 分數 |
| 測試 GPT-4 零樣本翻譯（如有 key）| `benchmark/results/gpt4_baseline.json` | 記錄 BLEU 分數 |
| 彙整 baseline 數據 | `benchmark/BASELINE_REPORT.md` | 表格 + 分析 |

**自行迭代檢測點：**
- [ ] test_set.json 每類至少 5 句
- [ ] BLEU 腳本能自動跑完全部 40 句
- [ ] baseline 數據已記錄（數字，不是描述）

#### Day 2 — 4/27（日）Benchmark 收尾 + RAG 擴充

**目標：** baseline 數據完成 + RAG 翻譯能力擴充

| 任務 | 產出 | 驗收標準 |
|------|------|----------|
| Benchmark 補測（如有漏的模型）| `benchmark/results/` 補全 | 全部模型數據齊全 |
| 合併語料：corpus(42) + phrases(118) + CSV(159) | `data/expanded_corpus.json` | 統一格式，去重 |
| RAG 索引重建（擴充語料）| `data/faiss_index_v2.bin` | 新索引含所有語料 |
| 用 Benchmark 測 RAG 版翻譯 | `benchmark/results/rag_v2.json` | BLEU vs baseline 有提升 |
| 量化 RAG 增益 | 更新 `BASELINE_REPORT.md` | 數據對比表 |

**自行迭代檢測點：**
- [ ] expanded_corpus.json 去重後至少 200 條
- [ ] RAG v2 BLEU > baseline
- [ ] 增益數據有表格呈現

#### Day 3 — 4/28（一）RAG 翻譯功能 + 準備飛書

**目標：** RAG 翻譯功能完成 + 飛書 App 申請

| 任務 | 產出 | 驗收標準 |
|------|------|----------|
| RAG 翻譯 API（排灣→中、中→排灣）| `rag_service.py` 新增 translate() | 雙向翻譯能用 |
| 用 Benchmark 測翻譯 API | `benchmark/results/rag_translate.json` | 記錄 BLEU |
| 在飛書開放平台創建應用 | App ID + Secret | 拿到憑證 |
| 配置飛書 Bot 事件訂閱 | Webhook URL 驗證通過 | 飛書能發消息給 Bot |
| 本地 ngrok 測試通道 | ngrok URL 穩定 | Webhook 可達 |

**自行迭代檢測點：**
- [ ] 雙向翻譯各跑 20 句無報錯
- [ ] 飛書 App 已創建，拿到 App ID
- [ ] ngrok 隧道穩定運行

---

### Phase 2：飛書 Bot 核心（4/29-5/2）

#### Day 4 — 4/29（二）飛書 Bot 文字對話

**目標：** Bot 能接收文字訊息 → RAG + LLM → 回覆

| 任務 | 產出 | 驗收標準 |
|------|------|----------|
| 實作飛書 token 管理 | `feishu_bot/auth.py` | 自動刷新 tenant_access_token |
| 實作訊息接收+回覆 | `feishu_bot/bot.py` 完善 | 文字對話跑通 |
| 整合 RAG + LLM | Bot 回覆含排灣語+中文 | 回覆品質 ≥ Gradio 版 |
| vuvu Maliq 角色系統提示 | System prompt 優化 | 回覆有角色感 |
| 基本指令：/learn, /translate | 指令路由 | 指令能觸發對應功能 |

**自行迭代檢測點：**
- [ ] 在飛書私聊 Bot，5 句對話無報錯
- [ ] 回覆包含排灣語 + 中文
- [ ] /learn 回覆教學內容
- [ ] /translate 回覆翻譯結果

#### Day 5 — 4/30（三）Bot 語音輸入（ASR）

**目標：** Bot 能接收語音訊息 → ASR 轉文字 → 處理

| 任務 | 產出 | 驗收標準 |
|------|------|----------|
| 飛書語音訊息下載 | `feishu_bot/media.py` | 能下載 ogg → 轉 wav |
| 整合本地 ASR API | 語音→文字 | ASR 辨識準確 |
| 語音+文字雙模式 | Bot 同時支援兩種輸入 | 切換無縫 |
| 錯誤處理 | 語音太短/格式不對有提示 | 不會 crash |

**自行迭代檢測點：**
- [ ] 發語音給 Bot，收到正確辨識+回覆
- [ ] 發文字給 Bot，正常回覆
- [ ] 發空語音/太短語音，有友善提示
- [ ] 連續 10 次對話無 crash

#### Day 6 — 5/1（四）Bot 語音輸出（TTS）+ 教學場景

**目標：** Bot 能回覆語音訊息 + 教學互動場景

| 任務 | 產出 | 驗收標準 |
|------|------|----------|
| TTS 回覆功能（先用 edge-tts 中文）| `feishu_bot/tts_reply.py` | Bot 能發語音訊息 |
| 排灣語 TTS（預合成音檔匹配）| 匹配 hf_deploy/paiwan_tts/ | 已有音檔的詞能回語音 |
| /quiz 指令：出題+判斷對錯 | 互動測驗功能 | 能玩 5 輪問答 |
| /daily 指令：每日一句 | 每日排灣語推送 | 格式美觀 |
| 多輪對話上下文 | 對話歷史管理 | 能記住前文 |

**自行迭代檢測點：**
- [ ] Bot 回覆有時帶語音
- [ ] /quiz 能完整玩一輪
- [ ] /daily 回覆一句排灣語+翻譯+語法說明
- [ ] 連續 3 輪對話能記住上下文

#### Day 7 — 5/2（五）完整測試 + 用戶反饋

**目標：** Bot 穩定運行 + 開始收集用戶反饋

| 任務 | 產出 | 驗收標準 |
|------|------|----------|
| 全流程壓力測試 | 50 次對話無 crash | 穩定性 > 95% |
| 修復測試中發現的 bug | bug list 清零 | 無已知 critical bug |
| 撰寫用戶測試指南 | `docs/USER_TEST_GUIDE.md` | 簡單明瞭 |
| 邀請 2-3 位排灣族朋友測試 | 截圖+反饋記錄 | 至少 2 份反饋 |
| 記錄用戶反饋 | `docs/USER_FEEDBACK.md` | 有引述有截圖 |

**自行迭代檢測點：**
- [ ] 50 次對話成功率 > 95%
- [ ] 0 個 critical bug
- [ ] 至少 2 位用戶測試
- [ ] 反饋有具體引述（不是概括描述）

---

### Phase 3：Demo + TTS + 文件（5/3-5/5）

#### Day 8 — 5/3（六）TTS 修正

**目標：** 產出 15 句高品質 vuvu 音檔

| 任務 | 產出 | 驗收標準 |
|------|------|----------|
| 清華算力平台開機（RTX 4090）| conda Python 3.11 環境 | TTS 能 import |
| 上傳 regenerate_tts.py + IPA 映射 | 算力雲就緒 | 腳本可執行 |
| 生成 15 句標準音檔 | `tts_v2/*.wav` | 不重複、發音合理 |
| 音檔品質人工審核 | 15 句全部通過 | 無 bug、節奏自然 |
| 下載音檔到本地 | `assets/tts_v2/` | 本地備份完成 |

**自行迭代檢測點：**
- [ ] 15 個 wav 檔案大小合理（每個 2-10 秒）
- [ ] 每個音檔聽過無重複念三次
- [ ] IPA 映射生效（tj→ch, dj→j 等可感知）

> ⚠️ 如果算力平台數據已清空，需重新建環境（預留 2-3 小時）

#### Day 9 — 5/4（日）Demo 影片

**目標：** 60-90 秒 Demo 影片完成

| 任務 | 產出 | 驗收標準 |
|------|------|----------|
| Demo 腳本撰寫 | `docs/DEMO_SCRIPT_V2.md` | 有故事線 |
| 錄製飛書 Bot 演示 | 螢幕錄影 | 畫面清晰 |
| 錄製旁白（中文）| 語音軌 | 情感到位 |
| 剪輯成品 | `語聲同行2.0_Demo.mp4` | 60-90 秒 |
| 上傳至可訪問位置 | YouTube/Bilibili/雲端 | 連結可用 |

**Demo 故事線（建議）：**
1. 開場（5秒）：問題 — 排灣語正在消失
2. 解法（10秒）：vuvu Maliq，AI 祖母語伴
3. 展示文字對話（15秒）：飛書 Bot 聊天
4. 展示語音互動（20秒）：說排灣語 → Bot 回覆
5. 展示翻譯功能（15秒）：中排雙向翻譯
6. 用戶反饋（10秒）：真實引述
7. 願景（10秒）：16 族語言的未來

**自行迭代檢測點：**
- [ ] 影片 60-90 秒
- [ ] 畫面清晰、聲音清楚
- [ ] 故事線完整
- [ ] 上傳連結可正常播放

#### Day 10 — 5/5（一）技術文件 + 最終打磨

**目標：** 復賽提交物全部就緒

| 任務 | 產出 | 驗收標準 |
|------|------|----------|
| 復賽技術文件 | `語聲同行2.0_復賽技術文件.pdf` | 含所有新數據 |
| Benchmark 數據整理進文件 | 表格+圖表 | 每環節增益量化 |
| 用戶反饋整理進文件 | 引述+截圖 | 真實有說服力 |
| GitHub README 更新 | 反映最新架構 | 乾淨專業 |
| 飛書 Bot 截圖/操作指南 | `docs/BOT_GUIDE.md` | 評審能自行體驗 |
| 最終檢查清單 | 全部打勾 | 無遺漏 |

**自行迭代檢測點：**
- [ ] 技術文件 ≤ 15 頁，圖文並茂
- [ ] 所有數據有出處
- [ ] GitHub README 跟實際代碼一致
- [ ] 飛書 Bot 評審可自行體驗

#### Day 11 — 5/6（二）提交

| 任務 | 驗收標準 |
|------|----------|
| 提交復賽作品 | 所有文件+影片+連結已上傳 |
| 確認飛書 Bot 在線 | 評審能訪問 |
| 備份所有提交物 | 本地+雲端各一份 |

---

## 🔧 技術架構（復賽版）

```
飛書用戶端
    │
    ▼ 發送文字/語音
飛書開放平台（事件訂閱）
    │
    ▼ Webhook
語聲同行 Bot 後端（Flask + ngrok）
    │
    ├→ 文字訊息 → RAG 檢索 → LLM(GLM-4-Flash) → 回覆
    ├→ 語音訊息 → 下載音頻 → ASR(:8000) → RAG → LLM → 回覆
    ├→ /translate → RAG 翻譯（雙向）
    ├→ /learn → 教學模式（單詞+例句+語法）
    ├→ /quiz → 測驗模式（出題+判斷）
    └→ /daily → 每日一句
```

### 飛書 Bot 需要的權限
- `im:message` — 接收訊息
- `im:message:send_as_bot` — 發送訊息
- `im:resource` — 下載語音/圖片資源

---

## 📈 Benchmark 評測框架

### 測試集（40 組）
- 從 paiwan_phrases.txt（118句）+ corpus（42句）中抽樣
- 每類別至少 5 句（問候/身份/數量/位置/否定/文化/日常）
- 訓練集和測試集嚴格分離

### 評測指標
| 指標 | 用途 | 計算方式 |
|------|------|----------|
| BLEU | 翻譯品質 | 標準 BLEU-4 |
| WER | ASR 準確率 | 詞錯誤率 |
| 語法正確率 | 綴詞/語序 | 人工+規則判斷 |
| 回覆相關性 | LLM 品質 | 人工評分 1-5 |

### 對比實驗
| 實驗 | 模型 | 方法 |
|------|------|------|
| Exp 1 | GLM-4-Flash | 零樣本翻譯（baseline） |
| Exp 2 | GLM-4-Plus | 零樣本翻譯 |
| Exp 3 | GLM-4-Flash + RAG v1 | 117 句語料 |
| Exp 4 | GLM-4-Flash + RAG v2 | 擴充語料（200+句） |
| Exp 5 | GLM-4-Flash + RAG + 語法規則 | 加綴詞分析 |
| Exp 6 | ASR 原始 | Whisper-tiny + LoRA |
| Exp 7 | ASR + 音韻校正 | 三層校正 |

---

## 🚨 風險預案

| 風險 | 觸發條件 | 預案 |
|------|----------|------|
| 清華算力平台數據清空 | 開機後 /workspace 空了 | 本地重新打包上傳，預留 3h |
| TTS 修正效果仍不好 | IPA 映射後發音仍差 | 改用 edge-tts 中文版 + 預合成排灣詞 |
| 飛書審核不通過 | Bot 上線被拒 | 先用 Webhook 模式 + ngrok 演示 |
| 找不到排灣族朋友測試 | 5/2 前無反饋 | 用自己錄音模擬用戶測試，標注為「開發者自測」 |
| VPN 不穩定 | API 呼叫超時 | 提前測試，關鍵操作備離線方案 |
| 磁碟空間不足 | 剩餘 < 5GB | 先執行 conda/npm/pip cache 清理（7.7GB） |

---

## ✅ 自動檢測機制

### 每日收工前檢查（地陪自動執行）

```
□ 今日任務是否全部完成？
□ 有沒有新增 bug？
□ 產出檔案是否已 commit？
□ memory/YYYY-MM-DD.md 是否更新？
□ 明天的前置作業是否準備好？
```

### 階段性驗收（Phase 結束時）

```
Phase 1（4/28）：
□ Benchmark baseline 數據完整
□ RAG v2 BLEU > baseline
□ 擴充語料 200+ 條

Phase 2（5/2）：
□ 飛書 Bot 文字+語音雙模式可用
□ 50 次對話成功率 > 95%
□ 至少 2 份用戶反饋

Phase 3（5/5）：
□ 15 句 TTS 音檔品質通過
□ Demo 影片 60-90 秒
□ 技術文件完整
□ 提交物清單全部打勾
```

---

## 📦 最終提交物清單

| # | 提交物 | 格式 | 負責 |
|---|--------|------|------|
| 1 | 飛書 Bot（可體驗） | 在線服務 | 評審掃碼 |
| 2 | Demo 影片 | MP4, 60-90s | 雲端連結 |
| 3 | 復賽技術文件 | PDF | 附件 |
| 4 | GitHub 源碼 | Repo | 連結 |
| 5 | HF Spaces Demo | 在線 | 連結 |
| 6 | Benchmark 報告 | 含在技術文件 | 數據表 |
| 7 | 用戶反饋記錄 | 含在技術文件 | 截圖+引述 |
| 8 | 簡報/PPT（如需要） | PDF | 附件 |

---

*此計劃為活文件，每日更新進度。*
