# 語聲同行 競賽版 — 變更日誌

> **規則**: 本目錄為比賽專用沙盒。禁止寫入 `/Users/sbb-mei/paiwan_asr_project`。

## 2026-04-14 Phase 0: 沙盒建立

### 執行者: 地陪 (OpenClaw main agent)

### 操作紀錄
| 操作 | 來源 | 目標 | 方式 |
|------|------|------|------|
| 建立 Workspace | — | `~/Desktop/paiwan_competition_2026/` | mkdir |
| 複製 LoRA 權重 | `paiwan_asr_project/whisper-lora-paiwan-final/` (6MB) | `whisper-lora-paiwan-final/` | cp -r |
| 複製 API 後端 | `paiwan_asr_project/api/` | `api/` | cp -r |
| 複製語料 CSV | `paiwan_asr_project/族語彙整...csv` | 根目錄 | cp |
| 清理 __pycache__ | — | 全目錄 | rm -rf |
| 建立 requirements | — | `requirements_comp.txt` | 新建 |
| 建立變更日誌 | — | `CHANGELOG.md` | 新建 |

### 路徑隔離驗證
- [x] asr_service.py 使用相對路徑 → 新目錄結構一致，無需修改
- [x] 無任何寫入操作觸及原專案路徑
- [x] __pycache__ 已清理

### 目前目錄結構
```
paiwan_competition_2026/
├── requirements_comp.txt
├── CHANGELOG.md
├── whisper-lora-paiwan-final/     # ASR LoRA 權重 (6MB)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── api/
│   ├── server.py                  # FastAPI 後端 (10 個端點)
│   ├── asr_service.py             # Whisper LoRA ASR 服務
│   ├── test_intent.py
│   └── data/
│       ├── keyword_weights.json
│       ├── intent_training.jsonl
│       ├── game_vocabulary.json
│       └── ...
└── 族語彙整（固定格式） - 正式版本 .csv
```

### 下一步
- Phase 1: LLM 串接（智譜 API + vuvu System Prompt）
