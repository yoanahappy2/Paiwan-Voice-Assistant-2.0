# RAG 重構技術方案

> 語聲同行 2.0 — 從硬編碼語料到向量檢索
> 建立：2026-04-15

---

## 一、問題分析

### 現狀
- 117 句排灣語全部硬編碼在 System Prompt 中
- LLM 每次對話都要「讀完」所有語料 → token 浪費 + 上下文污染
- 無法動態擴展，新增語料必須改 prompt
- 語意匹配完全依賴 LLM 自行判斷，沒有檢索輔助

### 目標
- 語料從 prompt 移到向量資料庫
- 每次對話只檢索相關的 3-5 句語料注入 context
- 支持語意相似度匹配，不限於精確匹配
- 擴展性：新增語料只需更新索引，不需改 prompt

---

## 二、Embedding 方案選擇

### 方案對比

| 方案 | 模型 | 維度 | 中文能力 | 排灣語 | 成本 | 延遲 |
|------|------|------|----------|--------|------|------|
| **智譜 embedding-3** | embedding-3 | 2048 | ✅ 優秀 | ⚠️ 一般（罕見語言） | 免費額度 | ~200ms |
| 本地 sentence-transformers | paraphrase-multilingual | 768 | ✅ 好 | ⚠️ 一般 | 免費 | ~50ms |
| OpenAI text-embedding-3-small | 1536 | ✅ 好 | ⚠️ 一般 | 付費 | ~300ms |

### 最終選擇：**智譜 embedding-3**

理由：
1. **API 已有** — 智譜 API Key 已配好，零額外配置
2. **免費額度** — 117 句語料的 embedding 生成成本為零
3. **OpenAI 兼容** — 同一套 SDK，無需額外依賴
4. **中文理解好** — 排灣語語料中包含大量中文翻譯，中文語意匹配能力是關鍵

### 排灣語特殊考量
排灣語是極低資源語言，任何 embedding 模型都不會特別優化。但我們的策略是：
- 同時 embedding **排灣語句子 + 中文翻譯**
- 檢索時同時匹配用戶輸入的排灣語和可能的中文意圖
- 利用中文翻譯的語意空間來「代理」排灣語的語意理解

---

## 三、向量資料庫方案

### 選擇：**FAISS**（Facebook AI Similarity Search）

理由：
1. **輕量** — 純 Python，無需額外服務（不需要 Pinecone / Milvus 伺服器）
2. **快速** — 117 句語料量級，FAISS 毫秒級檢索
3. **離線** — 索引存本地檔案，不依賴網路
4. **比賽友好** — 評審可以在本地完整重現

### 替代方案（被否決）
- ChromaDB：稍重，但也可以接受（備選）
- Pinecone / Milvus：需要額外服務，過度工程化

---

## 四、架構設計

### 重構前（現狀）
```
用戶輸入 → LLM（System Prompt 含 117 句全部語料）→ 回覆
```

### 重構後（RAG）
```
用戶輸入
    │
    ├─→ Embedding（智譜 embedding-3）
    │       │
    │       ▼
    │   FAISS 向量檢索（Top-K=5）
    │       │
    │       ▼
    │   相關語料（3-5 句排灣語 + 翻譯）
    │       │
    ▼       ▼
  LLM（精簡 System Prompt + 檢索到的語料作為 context）→ 回覆
```

### 新的 System Prompt（精簡版）
```
你是 vuvu Maliq（排灣族祖母）...
[角色描述 - 不變]

## 本次對話相關語料
以下是根據用戶輸入檢索到的排灣語句子，作為回覆參考：
{rag_context}

## 回覆規則
- 優先參考上述語料進行回覆
- 如果檢索結果不相關，誠實告知
...
```

---

## 五、數據結構

### 語料向量索引結構

每筆語料的 metadata：
```json
{
  "paiwan": "masalu",
  "chinese": "謝謝",
  "category": "greeting",
  "intent": "thank",
  "embedding": [0.012, -0.034, ...],
  "id": 0
}
```

### 索引檔案
```
data/
├── paiwan_corpus.json       # 語料原文（117 句）
├── paiwan_embeddings.npy    # 預計算的 embedding 向量
├── faiss_index.bin           # FAISS 索引
└── corpus_metadata.json     # 語料 metadata
```

---

## 六、核心模組設計

### 新增檔案：`rag_service.py`

```python
class PaiwanRAG:
    """排灣語 RAG 檢索服務"""
    
    def __init__(self):
        self.index = None          # FAISS 索引
        self.metadata = []         # 語料 metadata
        self.embed_client = ...    # 智譜 embedding client
    
    def build_index(self):
        """建構向量索引（首次 or 語料更新時）"""
        # 1. 讀取 paiwan_phrases.txt
        # 2. 每句生成 embedding（排灣語 + 中文拼接）
        # 3. 存入 FAISS
    
    def search(self, query: str, top_k: int = 5) -> list:
        """語意檢索"""
        # 1. query → embedding
        # 2. FAISS 搜尋 Top-K
        # 3. 返回相關語料 + 相似度分數
    
    def format_context(self, results: list) -> str:
        """格式化檢索結果為 LLM context"""
```

### 修改檔案：`llm_service.py`

```python
class VuvuService:
    def __init__(self):
        self.rag = PaiwanRAG()  # 新增
        # System Prompt 移除 117 句硬編碼
    
    def chat_with_thinking(self, user_input: str) -> dict:
        # 1. RAG 檢索相關語料
        rag_context = self.rag.search(user_input)
        context_str = self.rag.format_context(rag_context)
        
        # 2. 動態注入 context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(
                rag_context=context_str
            )},
            *self.conversation_history
        ]
        
        # 3. 呼叫 LLM
        ...
```

---

## 七、預期效果

| 指標 | 硬編碼（現狀） | RAG（重構後） |
|------|---------------|--------------|
| System Prompt tokens | ~3000 | ~500（+ 動態 context ~300） |
| 語料匹配精確度 | 依賴 LLM 記憶 | 向量相似度 + LLM 判斷 |
| 擴展性 | 改 prompt | 更新索引即可 |
| 回覆延遲 | 基線 | +200ms（embedding）+ ~5ms（FAISS） |
| Token 成本 | 高（每次帶全部語料） | 低（只帶相關語料） |

---

## 八、實施步驟

1. **建構語料 JSON**（從 paiwan_phrases.txt 解析）
2. **生成 embedding**（呼叫智譜 embedding-3，批次處理）
3. **建 FAISS 索引**（L2 距離，IndexFlatL2）
4. **實作 rag_service.py**（PaiwanRAG 類別）
5. **重構 llm_service.py**（整合 RAG，移除硬編碼）
6. **A/B 測試**（硬編碼 vs RAG 回覆品質對比）
7. **更新 app.py**（如需要）

---

*此文件為技術設計稿，實作時可能微調。*
