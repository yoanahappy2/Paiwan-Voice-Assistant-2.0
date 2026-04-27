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

TRANSLATE_SYSTEM_PROMPT = """你是一個專業的排灣語-中文翻譯助手。

## 任務
根據提供的參考語料，將輸入翻譯為目標語言。

## 參考語料（RAG 檢索結果）
{rag_context}

## 規則
1. 如果參考語料中有完全匹配，直接使用該翻譯
2. 如果沒有完全匹配，根據語料中的詞彙和語法結構推導翻譯
3. 不要編造語料中沒有的排灣語詞彙
4. 如果無法確定翻譯，在前面加 [不確定]

## 輸出格式（嚴格遵守）
只輸出翻譯結果本身，不要任何解釋或多餘文字。

範例：
輸入「masalu」→ 輸出「謝謝！」
輸入「你好嗎」→ 輸出「na tarivak sun?」
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
        """建立雙向關鍵詞索引"""
        self.keyword_map_paiwan = {}
        self.keyword_map_chinese = {}

        for item in self.corpus:
            paiwan = item.get("paiwan", "").strip().lower()
            chinese = item.get("chinese", "").strip()

            if paiwan:
                # 排灣語關鍵詞（拆詞）
                for word in paiwan.split():
                    word = word.strip("?.!,")
                    if word and len(word) > 1:
                        if word not in self.keyword_map_paiwan:
                            self.keyword_map_paiwan[word] = []
                        self.keyword_map_paiwan[word].append(item)

            if chinese:
                # 中文關鍵詞（逐字+雙字組合）
                for i in range(len(chinese)):
                    for length in [2, 3, 4]:
                        if i + length <= len(chinese):
                            phrase = chinese[i:i+length]
                            if phrase not in self.keyword_map_chinese:
                                self.keyword_map_chinese[phrase] = []
                            self.keyword_map_chinese[phrase].append(item)

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

    def _check_exact_match(self, text: str, direction: str, results: list) -> str | None:
        """檢查是否有精確匹配"""
        text_clean = text.strip().lower().rstrip("?.!")

        for item in results:
            if direction == "p2c":
                paiwan_clean = item.get("paiwan", "").strip().lower().rstrip("?.!")
                if text_clean == paiwan_clean:
                    return item.get("chinese", "")
            else:
                chinese_clean = item.get("chinese", "").strip().rstrip("?.！")
                if text_clean == chinese_clean:
                    return item.get("paiwan", "")

        return None

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
