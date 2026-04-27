# 技術筆記：語料合併與分層翻譯策略

## 背景

原始語料庫僅 242 條（手動整理的日常用語+基礎詞彙），翻譯服務大量依賴 LLM 猜測，準確率低（如 `gacalju` 被翻成「食物」而非「起立」）。

## 語料來源

| 來源 | 數量 | 內容 |
|------|------|------|
| 原語料 (expanded_corpus.json) | 242 條 | 日常用語、基礎詞彙（手動整理） |
| 族語E樂園 (klokah.tw, d=23) | 4,687 條 | 詞彙、會話、文化、對話、歌謠、閱讀 |
| **合併去重後** | **4,844 條** | 東排灣語為主 |

### 方言代碼注意

- `d=23` = 東排灣語（masalu = 謝謝）
- `d=24` = 北排灣語（masalu = 相信）

**必須確認方言代碼，否則整批語料方向全錯。**

## 分層架構

翻譯服務採用兩層設計：

### 第一層：精確詞典（短詞匹配）

```
從 4,844 條中篩選 paiwan 文本 ≤ 3 個詞的條目 → 2,258 條
```

- 用途：`/translate gacalju` 的精確匹配
- 實作：字串比對（normalize 後比對），不需要向量索引
- 優勢：速度快、確定性高、不靠 LLM

```python
# 載入詞典
with open('data/paiwan_dictionary.json', 'r', encoding='utf-8') as f:
    dictionary = json.load(f)

# 建索引
keyword_map = {}
for entry in dictionary:
    for word in entry['paiwan'].lower().split():
        keyword_map.setdefault(word, []).append(entry)

# 查詢
def exact_match(query):
    normalized = query.lower().strip('!?.，。！？')
    return keyword_map.get(normalized)
```

### 第二層：RAG 語意檢索（LLM 兜底）

```
全部 4,844 條 → 智譜 embedding-3 (2048維) → FAISS IndexFlatL2
```

- 用途：精確匹配失敗時的語意搜索 + LLM 生成翻譯
- 實作：embedding → FAISS top-k → 構建 prompt → LLM 回覆
- 檔案：
  - `data/merged_faiss.index` — FAISS 索引（38MB）
  - `data/merged_embeddings.npy` — Embedding 矩陣

## 翻譯流程

```
用戶輸入 → normalize
    ↓
[精確詞典匹配] → 命中 → 直接返回（exact）
    ↓ 未命中
[FAISS top-k 檢索] → 構建 context prompt → LLM 生成（rag_llm）
    ↓
返回翻譯結果
```

## 合併邏輯

```python
# 原語料優先（手動整理的品質較高）
merged = {}
for e in klokah_entries:  # 先放 klokah
    merged[(e['paiwan'], e['chinese'])] = e
for e in original_entries:  # 原語料覆蓋
    merged[(e['paiwan'], e['chinese'])] = e
```

用 `(paiwan, chinese)` tuple 去重，原語料優先覆蓋（因為手動整理的分類和品質更好）。

## Embedding 建索引

```python
from openai import OpenAI
client = OpenAI(api_key=ZHIPUAI_KEY, base_url='https://open.bigmodel.cn/api/paas/v4')

texts = [f"{e['paiwan']} | {e['chinese']}" for e in entries]
# 分批 embedding（每批 16 條，避免 rate limit）
resp = client.embeddings.create(model='embedding-3', input=batch)
```

4,844 條約需 5-10 分鐘（取決於 API 速度）。

## 驗證結果

| 輸入 | 翻譯 | 方法 |
|------|------|------|
| gacalju | 起立！敬禮！(班長說) | exact ✅ |
| masalu | 謝謝！ | rag_llm ✅ |
| 水 | zaljumi | exact ✅ |
| 謝謝 | masalu | rag_llm ✅ |
| kama | 爸爸 | rag_llm ✅ |

## 檔案清單

| 檔案 | 用途 | 大小 |
|------|------|------|
| `data/merged_corpus.json` | 合併語料 | 749KB |
| `data/paiwan_dictionary.json` | 精確詞典 | 203KB |
| `data/merged_faiss.index` | FAISS 索引 | 38MB |
| `data/merged_embeddings.npy` | Embedding 矩陣 | 37.8MB |

## 日期
2026-04-27
