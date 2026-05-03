# ASR/TTS 訓練資料清單與概要

**整理日期**：2026-05-03
**專案**：語聲同行 2.0 — 排灣族語智能教學系統

---

## 一、ASR（語音辨識）

### 1.1 模型資訊

- **基礎模型**：OpenAI Whisper-small
- **微調方式**：LoRA（r=32, alpha=64）
- **準確率**：96.8%（94 個高準確詞）
- **模型路徑**：`/Users/sbb-mei/paiwan_asr_project/whisper-lora-paiwan-final/`
  - adapter_model.safetensors（14MB）
  - adapter_config.json
  - tokenizer 等附屬檔案
- **舊模型備份**：`whisper-lora-paiwan-final-tiny-backup/`（Whisper-tiny，46% 準確率）
- **⚠️ 不可修改**：此目錄為畢業論文資料

### 1.2 ASR API 服務

- **腳本**：`/Users/sbb-mei/paiwan_asr_project/main.py`
- **venv**：`/Users/sbb-mei/paiwan_asr_project/venv/`
- **啟動方式**：`venv/bin/python main.py`
- **端點**：`http://127.0.0.1:8000`
  - `POST /api/audio/recognize` — 語音辨識
  - `GET /health` — 健康檢查
  - `GET /api/game/level/{id}/words` — 遊戲關卡詞彙
  - `POST /api/game/match` — 發音匹配

### 1.3 語料（s03 音檔）

- **路徑**：`/Users/sbb-mei/paiwan_asr_project/audio/`
- **數量**：101 個 s03 音檔
- **格式**：.m4a（ISO Media, MP4）
- **大小**：6.7MB
- **說話人**：s03（母語者）
- **命名**：s03-{詞彙}.m4a（例：s03-alu.m4a、s03-lima.m4a）

---

## 二、TTS（語音合成）

### 2.1 TTS V1 — 原文輸入（XTTS v2）

- **路徑**：`~/Desktop/paiwan_competition_2026/paiwan_tts_v2/paiwan_tts_output/`
- **數量**：51 個 wav 檔
- **大小**：5.3MB
- **模型**：Coqui XTTS v2（voice cloning）
- **輸入**：排灣語原文（language="en"）
- **參考音檔**：s03 前 5 個真人錄音
- **命名**：s03-{詞彙}_tts.wav
- **ASR 驗證結果**：26/50 完全正確（52%），平均準確率 68.7%

### 2.2 TTS V2 — IPA 轉換輸入

- **路徑**：`~/Desktop/paiwan_competition_2026/paiwan_tts_v2/paiwan_tts_output_v2/`
- **數量**：19 個 wav 檔（僅針對 V1 失敗的詞重新生成）
- **大小**：2.6MB
- **輸入**：排灣語→英文近似音素轉換（tj→ch, q→k, c→ts, lj→ly, dj→j）
- **命名**：s03-{詞彙}_tts_v2.wav
- **ASR 驗證結果**：2/19 額外正確（meljay、quma），淨提升 2 詞

### 2.3 舊版 TTS（競賽初期）

- **路徑**：`~/Desktop/paiwan_competition_2026/paiwan_tts/`
- **數量**：52 個檔案（含 sentences/ 子目錄）
- **已知問題**：重複念三次 bug、英文化發音
- **狀態**：已被 V1/V2 取代

### 2.4 分層語音策略（最終方案）

| 層級 | 來源 | 數量 | 用途 |
|------|------|------|------|
| Layer 1 | 真人錄音（s03） | 101 個 | 優先使用 |
| Layer 2 | TTS V1（ASR 驗證通過） | 28 個 | 擴充覆蓋 |
| Layer 3 | TTS V1 剩餘（未通過驗證） | 22 個 | 備用/降級 |

---

## 三、訓練腳本與工具

### 3.1 算力平台腳本（cloud_train/）

- **路徑**：`~/Desktop/paiwan_competition_2026/cloud_train/`
- **train_paiwan.py** — XTTS v2 微調訓練腳本（未使用，語料不足）
- **regenerate_tts.py** — TTS 音檔重新生成腳本
- **paiwan_ipa.py** — 排灣語→英文近似音素轉換
- **inference_paiwan_simple.py** — 簡易推理腳本
- **test_inference.py** — 測試腳本
- **run_all.sh** — 一鍵執行

### 3.2 Dataset

- **路徑**：`cloud_train/dataset/`
- **metadata.csv** — 50 筆（audio_file, text, speaker_id）
- **metadata_train.csv / metadata_eval.csv** — 訓練/評估分割
- **wavs/** — 50 個 wav 檔（從 s03 m4a 轉換）

### 3.3 評估工具

- **eval_tts.py** — ASR 反向驗證 TTS 品質腳本
  - 路徑：`~/Desktop/paiwan_competition_2026/eval_tts.py`
- **eval_results.json** — 50 詞完整評估數據
  - 路徑：`~/Desktop/paiwan_competition_2026/paiwan_tts_v2/eval_results.json`

---

## 四、算力平台資訊（已關閉）

- **平台**：Paratera（清華算力平台）
- **配置**：RTX 3090 24GB / 10 vCPU / 60GB RAM
- **PyTorch**：2.2.0+cu118
- **Python**：3.10.12（base）
- **TTS 版本**：0.22.0
- **transformers**：4.38.2（從預設降級以兼容 PyTorch 2.2）
- **學術加速 proxy**：http://u-UE25Z3:tXGJgV92@10.255.128.102:3128
- **HF_ENDPOINT**：https://hf-mirror.com

---

## 五、IPA 轉換規則（paiwan_ipa.py）

| 排灣語 | 英文近似 | 說明 |
|--------|----------|------|
| tj | ch | 清齦後塞擦音 |
| dj | j | 濁齦後塞擦音 |
| lj | ly | 硬顎邊音 |
| ng | ng | 軟顎鼻音（保持不變） |
| q | k | 小舌塞音 |
| c | ts | 清齒齦塞擦音 |

範例：ceqer → tseker、quma → kuma、tjelu → chelu

---

## 六、關鍵結論

1. **ASR 是強項**：Whisper-small + LoRA 達 96.8% 準確率，94 個高準確詞可用
2. **TTS 有限**：XTTS v2 無法真正學會排灣語發音，IPA 轉換僅改善 2 詞
3. **最佳策略**：真人錄音為主 + TTS 補充 + ASR 驗證閉環
4. **根本限制**：排灣語不在 XTTS v2 的 17 種支援語言中，voice cloning 只能複製音色不能複製音素
5. **學術價值**：分層語音策略 + ASR 反向驗證閉環，是低資源瀕危語言 TTS 的可行工程方案
