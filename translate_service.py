"""
translate_service.py
語聲同行 2.0 — 排灣語↔中文 雙向翻譯服務

基於 RAG 檢索 + LLM 生成，支援：
- 排灣語 → 中文
- 中文 → 排灣語
- 混合 RAG（關鍵詞 + 向量）

作者: 地陪
日期: 2026-04-26
"""

import os
import re
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

client = OpenAI(
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

MODEL_FAST = "glm-4-flash"
EMBEDDING_MODEL = "embedding-3"
EMBEDDING_DIM = 2048

BASE_DIR = Path(__file__).parent

# ============================================
# 翻譯 System Prompt
# ============================================

TRANSLATE_SYSTEM_PROMPT = """你是一個專業的**東排灣語**-中文翻譯助手。

## 任務
根據提供的參考語料，將輸入翻譯為目標語言。

## 參考語料（RAG 檢索結果）
{rag_context}

## ⚠️ 排灣語硬性語法規範（必須遵守）

### 1. 語序：VSO（動詞-主詞-賓語）
排灣語的基本語序是 VSO，不是中文的 SVO。
- 中文「我 吃 飯」→ 排灣語「吃 我 飯」
- 中文「他 想 家人」→ 排灣語「想 他 家人」

### 2. 格位標記（Case Markers）
賓語前面必須加格位標記：
- **tua**：一般名詞賓語（事物、概念）
  例：sengelit aken tua taqumaqanan（我想念家人）
- **tjai / tjanu**：人名或人稱賓語
  例：sengelit aken tjai vuvu（我想念祖母）
- **ti**：專有名詞前（人名）
  例：ti savan a ku ngadan（我的名字是 savan）
- **a**：不定名詞標記（在數量詞或描述後）
  例：izua a drangi（有一個朋友）

### 3. 代名詞綁定
人稱代名詞是獨立詞，緊隨動詞（不與動詞融合）：
- **aken** = 我（主格）
- **sun** = 你（主格）
- **timadju / tiamadju** = 他/她
- **itjen / itjenan** = 我們（含/不含聽者）
- **mun** = 你們
- **tiamadju** = 他們

代名詞位置：動詞 + 代名詞 + 賓語
例：qemavilj aken tua kakanen（我買食物）

### 4. 所有格標記
- **ku** = 我的（ku ngadan = 我的名字）
- **su** = 你的（su ngadan = 你的名字）
- **nia** = 他/她的
- **ta / ita** = 我們的
- **u** = 所有格標記（放在名詞前）
  例：tua u taqumaqanan = 的家人

### 5. 疑問句
疑問詞放在句首或動詞前：
- **inuan** = 哪裡
- **anema** = 什麼
- **pida** = 多少
- **tima** = 誰
- **nungida** = 什麼時候
- **namakuda** = 如何

## 組合翻譯範例（Few-shot）

範例 1：中文「我很想念我的家人」→ 排灣語
拆解：想念(V) + 我(S) + 家人(O)
→ sengelit（想念）+ aken（我）+ tua（格位）+ u（所有格）+ taqumaqanan（家人）
→ **sengelit aken tua u taqumaqanan.**

範例 2：中文「他今天去山上工作」→ 排灣語
拆解：工作(V) + 他(S) + 今天 + 山上
→ masengeseng（工作）+ tiamadju（他）+ nusauni（今天）+ i gadu（在山上）
→ **masengeseng tiamadju nusauni i gadu.**

範例 3：中文「你叫什麼名字」→ 排灣語
拆解：名字 + 你 + 什麼
→ **tima su ngadan?**

範例 4：中文「我想喝水」→ 排灣語
拆解：想喝(V) + 我(S) + 水(O)
→ tjengelay aken a kium tu zaljum.

## 翻譯規則（優先級從高到低）
1. 參考語料中有完全匹配 → 直接使用該翻譯
2. 有部分匹配 → 根據語料中的詞彙，嚴格按照上述 VSO 語法和格位標記規則組合
3. 組合時必須包含格位標記（tua/tjai/tjanu），不得省略
4. 不要編造語料中沒有的排灣語詞彙
5. 如果無法確定翻譯，在前面加 [不確定]

## 輸出格式（嚴格遵守）
只輸出翻譯結果本身，不要任何解釋或多餘文字。

範例：
輸入「masalu」→ 輸出「謝謝！」
輸入「你好嗎」→ 輸出「na tarivak sun?」
輸入「我很想念我的家人」→ 輸出「sengelit aken tua u taqumaqanan."
"""


# ============================================
# 翻譯服務
# ============================================

class PaiwanTranslator:
    """排灣語↔中文 雙向翻譯服務"""

    def __init__(self):
        self.index = None
        self.corpus = []
        self.dictionary = None  # 精確詞典
        self.keyword_map_paiwan = {}  # 排灣語關鍵詞索引
        self.keyword_map_chinese = {}  # 中文關鍵詞索引
        self._loaded = False

    def load(self):
        """載入擴充語料和 FAISS 索引"""
        if self._loaded:
            return

        # 優先載入合併語料（含 klokah 東排灣）
        merged_file = BASE_DIR / "data" / "merged_corpus.json"
        fallback_file = BASE_DIR / "data" / "expanded_corpus.json"

        if merged_file.exists():
            corpus_file = merged_file
        elif fallback_file.exists():
            corpus_file = fallback_file
        else:
            raise FileNotFoundError("語料檔案不存在")

        with open(corpus_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, dict) and "entries" in raw:
                self.corpus = raw["entries"]
            elif isinstance(raw, list):
                self.corpus = raw
            else:
                raise ValueError(f"未知的語料格式: {type(raw)}")

        # 過濾掉 undefined 中文
        self.corpus = [e for e in self.corpus if e.get('chinese', '').strip() 
                       and e['chinese'].strip().lower() != 'undefined']

        # 語料清洗：統一翻譯格式
        for e in self.corpus:
            c = e.get('chinese', '').strip()
            # 移除前綴標記 (B: A: 等)
            import re
            c = re.sub(r'^[A-Za-z]:\s*', '', c)
            # 移除句尾標點
            c = c.rstrip('。！？：；,. ')
            e['chinese'] = c

        # 載入核心詞彙表（最高優先級，直接作為精確匹配的首選）
        self._core_vocab = {}  # {paiwan_lower: chinese}
        core_vocab_file = BASE_DIR / "data" / "core_vocab.tsv"
        if core_vocab_file.exists():
            with open(core_vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        p, c = parts[0].strip(), parts[1].strip()
                        if p and c:
                            self._core_vocab[p.lower()] = c
            if self._core_vocab:
                print(f"  核心詞彙表：{len(self._core_vocab)} 筆")

        # 載入精確詞典（短詞優先匹配）
        dict_file = BASE_DIR / "data" / "paiwan_dictionary.json"
        if dict_file.exists():
            with open(dict_file, "r", encoding="utf-8") as f:
                self.dictionary = json.load(f)
        else:
            self.dictionary = None

        # 載入 FAISS 向量索引（優先合併版）
        merged_index = BASE_DIR / "data" / "merged_faiss.index"
        fallback_index = BASE_DIR / "data" / "expanded_faiss.index"
        if merged_index.exists():
            self.index = faiss.read_index(str(merged_index))
        elif fallback_index.exists():
            self.index = faiss.read_index(str(fallback_index))
        else:
            print("⚠️ 向量索引不存在，只使用關鍵詞檢索")
            self.index = None

        # 建立關鍵詞索引
        self._build_keyword_maps()

        self._loaded = True
        print(f"✅ 翻譯服務載入完成：{len(self.corpus)} 筆語料")

    def _build_keyword_maps(self):
        """建立雙向關鍵詞索引 + 精確匹配索引"""
        self.keyword_map_paiwan = {}
        self.keyword_map_chinese = {}
        # 精確匹配索引：排灣語/中文原文 → 所有匹配的翻譯
        self._exact_paiwan = {}  # {paiwan_lower: [chinese1, chinese2, ...]}
        self._exact_chinese = {}  # {chinese: [paiwan1, paiwan2, ...]}

        for item in self.corpus:
            paiwan = item.get("paiwan", "").strip()
            chinese = item.get("chinese", "").strip()

            # 精確匹配索引
            p_key = paiwan.lower().rstrip("?.!")
            if p_key:
                if p_key not in self._exact_paiwan:
                    self._exact_paiwan[p_key] = []
                # 拆分多義翻譯（分號、斜線分隔）
                for c_part in re.split(r'[；;/]', chinese):
                    c_part = c_part.strip(' 。！？：，,')
                    if c_part and c_part not in self._exact_paiwan[p_key]:
                        self._exact_paiwan[p_key].append(c_part)

            if chinese:
                c_key = chinese.rstrip("?.！")
                if c_key:
                    if c_key not in self._exact_chinese:
                        self._exact_chinese[c_key] = []
                    if paiwan and paiwan not in self._exact_chinese[c_key]:
                        self._exact_chinese[c_key].append(paiwan)

            # 關鍵詞索引（用於 RAG 模糊檢索）
            if paiwan:
                for word in paiwan.lower().split():
                    word = word.strip("?.!,")
                    if word and len(word) > 1:
                        if word not in self.keyword_map_paiwan:
                            self.keyword_map_paiwan[word] = []
                        self.keyword_map_paiwan[word].append(item)

            if chinese:
                for i in range(len(chinese)):
                    for length in [2, 3, 4]:
                        if i + length <= len(chinese):
                            phrase = chinese[i:i+length]
                            if phrase not in self.keyword_map_chinese:
                                self.keyword_map_chinese[phrase] = []
                            self.keyword_map_chinese[phrase].append(item)

        print(f"  精確匹配索引：{len(self._exact_paiwan)} 排灣語詞、{len(self._exact_chinese)} 中文詞")

        # 建立變體索引（處理拼寫差異）
        # 常見變體：sa/saka, ta/taka 等
        self._build_variant_index()

    def _is_paiwan(self, text: str) -> bool:
        """判斷輸入是否為排灣語"""
        # 排灣語特徵：拉丁字母 + 常見排灣語詞綴
        paiwan_markers = [
            "na ", "su ", "aken", "sun?", "mun", "tima", "pida",
            "anema", "inuan", "ini", "maya", "masalu", "tarivak",
            "nanguaq", "pacunan", "tjengelay", "vuvu", "kama",
            "cavilj", "ngadan", "umaq", "zua", "tiyamadju",
            "tjaljaljak", "vavayan", "uqaljay", "siruvetjek",
            "taqumaqanan", "qadupu", "milingan", "kakedrian",
        ]
        text_lower = text.lower().strip()
        # 檢查是否有拉丁字母（非中文）
        has_latin = any(c.isalpha() and ord(c) < 128 for c in text_lower)
        if not has_latin:
            return False
        # 如果包含中文字，不是排灣語
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
        if has_chinese:
            return False
        return True

    def _keyword_search(self, query: str, direction: str, top_k: int = 5) -> list:
        """關鍵詞檢索"""
        results = []
        seen_ids = set()

        if direction == "p2c":
            # 排灣語 → 中文：在排灣語關鍵詞中搜索
            query_words = query.lower().split()
            for word in query_words:
                word = word.strip("?.!,")
                matches = self.keyword_map_paiwan.get(word, [])
                for m in matches:
                    mid = id(m)  # 用物件 id 去重
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        results.append(m)
        else:
            # 中文 → 排灣語：在中文關鍵詞中搜索
            for i in range(len(query)):
                for length in [4, 3, 2]:
                    if i + length <= len(query):
                        phrase = query[i:i+length]
                        matches = self.keyword_map_chinese.get(phrase, [])
                        for m in matches:
                            mid = id(m)
                            if mid not in seen_ids:
                                seen_ids.add(mid)
                                results.append(m)

        return results[:top_k]

    def _vector_search(self, query: str, top_k: int = 5) -> list:
        """向量語意檢索"""
        if self.index is None:
            return []

        try:
            embed_client = OpenAI(
                api_key=ZHIPUAI_API_KEY,
                base_url="https://open.bigmodel.cn/api/paas/v4"
            )
            response = embed_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query,
            )
            query_vec = np.array([response.data[0].embedding], dtype=np.float32)

            distances, indices = self.index.search(query_vec, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.corpus):
                    item = self.corpus[idx].copy()
                    item["distance"] = float(dist)
                    results.append(item)

            return results
        except Exception as e:
            print(f"⚠️ 向量檢索失敗: {e}")
            return []

    def _hybrid_search(self, query: str, direction: str, top_k: int = 5) -> list:
        """混合檢索（關鍵詞 + 向量）"""
        # 關鍵詞結果
        kw_results = self._keyword_search(query, direction, top_k=top_k)

        # 向量結果
        vec_results = self._vector_search(query, top_k=top_k)

        # 合併去重（關鍵詞優先）
        seen = set()
        merged = []

        for item in kw_results:
            key = f"{item.get('paiwan', '')}|{item.get('chinese', '')}"
            if key not in seen:
                seen.add(key)
                item["source"] = "keyword"
                merged.append(item)

        for item in vec_results:
            key = f"{item.get('paiwan', '')}|{item.get('chinese', '')}"
            if key not in seen:
                seen.add(key)
                item["source"] = "vector"
                merged.append(item)

        return merged[:top_k]

    def _format_rag_context(self, results: list) -> str:
        """格式化 RAG 結果給 LLM"""
        if not results:
            return "（未找到相關語料）"

        lines = []
        for i, r in enumerate(results, 1):
            src = r.get("source", "")
            src_tag = f" [{src}]" if src else ""
            line = f"{i}. 排灣語: {r.get('paiwan', '')} ｜中文: {r.get('chinese', '')}{src_tag}"
            lines.append(line)
        return "\n".join(lines)

    def translate(self, text: str, direction: str = "auto", top_k: int = 5) -> dict:
        """
        翻譯入口

        Args:
            text: 要翻譯的文字
            direction: "p2c"(排灣→中), "c2p"(中→排灣), "auto"(自動偵測)
            top_k: RAG 檢索數量

        Returns:
            {
                "input": 原文,
                "direction": 方向,
                "translation": 翻譯結果,
                "rag_context": RAG 檢索結果,
                "rag_results": 原始檢索結果,
                "method": "exact" | "rag_llm"
            }
        """
        if not self._loaded:
            self.load()

        # 自動偵測方向
        if direction == "auto":
            direction = "p2c" if self._is_paiwan(text) else "c2p"

        # 混合 RAG 檢索
        rag_results = self._hybrid_search(text, direction, top_k=top_k)

        # 檢查是否有精確匹配
        exact = self._check_exact_match(text, direction, rag_results)
        if exact:
            return {
                "input": text,
                "direction": direction,
                "translation": exact,
                "rag_context": self._format_rag_context(rag_results),
                "rag_results": rag_results,
                "method": "exact",
            }

        # RAG + LLM 翻譯
        context_str = self._format_rag_context(rag_results)
        system_prompt = TRANSLATE_SYSTEM_PROMPT.format(rag_context=context_str)

        # 方向提示
        if direction == "p2c":
            user_msg = f"請將以下排灣語翻譯為中文：{text}"
        else:
            user_msg = f"請將以下中文翻譯為排灣語：{text}"

        try:
            response = client.chat.completions.create(
                model=MODEL_FAST,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,  # 翻譯用低溫度
                max_tokens=200,
            )
            translation = response.choices[0].message.content.strip()

            return {
                "input": text,
                "direction": direction,
                "translation": translation,
                "rag_context": context_str,
                "rag_results": rag_results,
                "method": "rag_llm",
            }
        except Exception as e:
            return {
                "input": text,
                "direction": direction,
                "translation": f"[翻譯失敗: {e}]",
                "rag_context": context_str,
                "rag_results": rag_results,
                "method": "error",
            }

    def _check_exact_match(self, text: str, direction: str, results: list = None) -> str | None:
        """在全語料庫中做精確匹配（O(1) hash 查找）"""
        text_clean = text.strip().lower().rstrip("?.!")
        if not text_clean:
            return None

        # 1. 核心詞彙表（最高優先級）
        if direction == "p2c" and text_clean in self._core_vocab:
            return self._core_vocab[text_clean]

        # 2. 全語料精確匹配
        if direction == "p2c":
            matches = self._exact_paiwan.get(text_clean, [])
        else:
            matches = self._exact_chinese.get(text_clean, [])

        if not matches:
            return None

        return self._pick_best_translation(matches)

    def _pick_best_translation(self, translations: list) -> str:
        """從多個翻譯中選最佳的一個
        
        策略：純中文 >=2字 且出現次數最多的優先
        """
        from collections import Counter
        
        counter = Counter()
        for t in translations:
            t_clean = t.strip(' 。！？：；,:')
            if t_clean:
                counter[t_clean] += 1
        
        # 分組：純中文 >=2 字 / 純中文 1 字 / 其他
        chinese_2plus = [(t, c) for t, c in counter.items() 
                        if all('\u4e00' <= ch <= '\u9fff' for ch in t) and len(t) >= 2]
        chinese_1 = [(t, c) for t, c in counter.items() 
                     if all('\u4e00' <= ch <= '\u9fff' for ch in t) and len(t) == 1]
        others = [(t, c) for t, c in counter.items() 
                  if not all('\u4e00' <= ch <= '\u9fff' for ch in t)]
        
        if chinese_2plus:
            chinese_2plus.sort(key=lambda x: (-x[1], len(x[0])))
            return chinese_2plus[0][0]
        if chinese_1:
            chinese_1.sort(key=lambda x: (-x[1],))
            return chinese_1[0][0]
        if others:
            others.sort(key=lambda x: (-x[1],))
            return others[0][0]
        
        return translations[0] if translations else ""

    def _build_variant_index(self):
        """建立變體拼寫索引：如果用戶輸入 sa，也能匹配 saka 的結果"""
        # 收集所有含 'saka' 的 key，建立 'sa' 的 alias
        variant_pairs = {}  # {變體key: [原key1, 原key2]}
        
        for key in list(self._exact_paiwan.keys()):
            # saka → sa 變體
            if 'saka ' in key:
                alt = key.replace('saka ', 'sa ')
                if alt not in self._exact_paiwan:
                    variant_pairs.setdefault(alt, []).append(key)
            # taka → ta 變體  
            if 'taka ' in key:
                alt = key.replace('taka ', 'ta ')
                if alt not in self._exact_paiwan:
                    variant_pairs.setdefault(alt, []).append(key)
        
        # 把變體指向原 key 的結果
        for variant, originals in variant_pairs.items():
            merged = []
            for orig in originals:
                merged.extend(self._exact_paiwan[orig])
            self._exact_paiwan[variant] = merged
        
        if variant_pairs:
            print(f"  變體索引：{len(variant_pairs)} 個拼寫變體")

    def _merge_translations(self, translations: list) -> str:
        """合併多個翻譯：去重、清理、排序後用「/」連接
        
        例如 ['媽媽', '媽', '母親', '阿姨'] → '媽媽 / 母親 / 阿姨'
        例如 ['天', '太陽', '太陽天', '一天'] → '太陽 / 天'
        """
        if not translations:
            return ""
        if len(translations) == 1:
            return translations[0].strip("。！：； ")

        # Step 1: 清理每個翻譯
        cleaned = []
        for t in translations:
            t = t.strip("。！：；,， ").strip()
            if not t or len(t) == 0:
                continue
            cleaned.append(t)

        # Step 2: 精確去重（保留順序）
        seen = set()
        unique = []
        for t in cleaned:
            t_lower = t.lower().strip()
            if t_lower not in seen:
                seen.add(t_lower)
                unique.append(t)

        # Step 3: 只做精確去重，不做子串去重
        # （中文子串關係不代表冗餘，「媽」和「媽媽」是不同意義）
        final = unique[:]

        # Step 4: 排序 — 優先核心意義
        # 策略：純漢字且 >=2 字的排最前，然後純數字，然後其他
        def _sort_key(x):
            is_pure_chinese = all('\u4e00' <= c <= '\u9fff' for c in x)
            is_pure_number = all(c.isdigit() for c in x)
            if is_pure_chinese and len(x) >= 2:
                return (0, len(x), x)  # 最優先：2字以上的純中文
            elif is_pure_chinese:
                return (1, len(x), x)  # 其次：1字中文
            elif is_pure_number:
                return (2, len(x), x)  # 再次：純數字
            else:
                return (3, len(x), x)  # 最後：混合

        final.sort(key=_sort_key)

        # Step 5: 限制數量（多義詞最多顯示前 8 個）
        if len(final) > 8:
            final = final[:8]

        return " / ".join(final)

    def batch_translate(self, texts: list, direction: str = "auto") -> list:
        """批量翻譯"""
        results = []
        for i, text in enumerate(texts):
            print(f"  [{i+1}/{len(texts)}] {text[:30]}...")
            result = self.translate(text, direction)
            results.append(result)
        return results


# ============================================
# 終端機測試
# ============================================

def main():
    print("=" * 50)
    print("  語聲同行 2.0 — 雙向翻譯測試")
    print("=" * 50)

    translator = PaiwanTranslator()
    translator.load()

    test_cases = [
        ("masalu", "auto"),
        ("na tarivak sun?", "auto"),
        ("tima su ngadan?", "auto"),
        ("你好嗎", "auto"),
        ("謝謝", "auto"),
        ("你叫什麼名字", "auto"),
        ("你幾歲", "auto"),
        ("pida anga su cavilj?", "auto"),
        ("再見", "auto"),
        ("pacunan", "auto"),
    ]

    for text, direction in test_cases:
        result = translator.translate(text, direction)
        method_tag = "🎯" if result["method"] == "exact" else "🤖"
        dir_tag = "排→中" if result["direction"] == "p2c" else "中→排"
        print(f"\n{method_tag} [{dir_tag}] {result['input']}")
        print(f"   → {result['translation']}")
        print(f"   方法: {result['method']}")

    print("\n" + "=" * 50)
    print("測試完成")


if __name__ == "__main__":
    main()
