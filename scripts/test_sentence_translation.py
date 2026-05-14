#!/usr/bin/env python3
"""
test_sentence_translation.py — 測試句子級翻譯正確率
用 klokah 已知正確的句子當 ground truth，跑系統翻譯，比對結果
零 token？不，會調 translate_service（裡面有 RAG + LLM）
但如果只想測 RAG 召回品質，可以只看 rag_results 不調 LLM
"""

import sys
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# 已知正確的排灣語-中文對照（來自 klokah 東排灣 d=23）
# 包含單詞 + 句子
TEST_CASES = [
    # 單詞
    {"paiwan": "masalu", "chinese": "謝謝", "type": "word"},
    {"paiwan": "aiyanga", "chinese": "你好", "type": "word"},
    {"paiwan": "na tarivak sun?", "chinese": "你好嗎？", "type": "word"},
    {"paiwan": "djavadjavai", "chinese": "你們好", "type": "word"},
    {"paiwan": "zaljum", "chinese": "水", "type": "word"},
    {"paiwan": "drangi", "chinese": "朋友(女的)", "type": "word"},
    # 句子（來自 klokah conversation）
    {"paiwan": "na keman anga ken", "chinese": "我吃過了", "type": "sentence"},
    {"paiwan": "ruqivu sun ta penayuanan?", "chinese": "你常說排灣語嗎？", "type": "sentence"},
    {"paiwan": "ui, payuan aken a kacalisian.", "chinese": "是，我是排灣族的。", "type": "sentence"},
    {"paiwan": "tiaken na paiwan a aljak.", "chinese": "我是排灣族的小孩。", "type": "sentence"},
    {"paiwan": "Payuan itjen a pimapuljatan.", "chinese": "我們都是排灣族。", "type": "sentence"},
    {"paiwan": "anema su kinatjengelayan ta vangavangaven?", "chinese": "你最喜歡的運動是什麼？", "type": "sentence"},
    {"paiwan": "pacun aken ta tilibi.", "chinese": "我在看電視。", "type": "sentence"},
    {"paiwan": "izua, ku pinaljiljiljan anga katiaw.", "chinese": "有，我已經安排好了。", "type": "sentence"},
    # 中文→排灣語 測試
    {"chinese": "謝謝", "expected_paiwan": "masalu", "direction": "c2p", "type": "word"},
    {"chinese": "你好", "expected_paiwan": "aiyanga", "direction": "c2p", "type": "word"},
    {"chinese": "我吃過了", "expected_paiwan": "na keman anga ken", "direction": "c2p", "type": "sentence"},
    {"chinese": "我是排灣族的小孩", "expected_paiwan": "tiaken na paiwan a aljak", "direction": "c2p", "type": "sentence"},
    {"chinese": "你常說排灣語嗎？", "expected_paiwan": "ruqivu sun ta penayuanan?", "direction": "c2p", "type": "sentence"},
    {"chinese": "我在看電視", "expected_paiwan": "pacun aken ta tilibi", "direction": "c2p", "type": "sentence"},
]


def test_rag_only():
    """只測 RAG 召回，不調 LLM（零 token）"""
    print("=" * 70)
    print("  📊 RAG 召回品質測試（零 token 消耗）")
    print("=" * 70)
    
    sys.path.insert(0, str(BASE_DIR))
    from translate_service import PaiwanTranslator
    
    translator = PaiwanTranslator()
    translator.load()
    
    correct = 0
    partial = 0
    wrong = 0
    not_found = 0
    results = []
    
    for tc in TEST_CASES:
        direction = tc.get("direction", "p2c")
        
        if direction == "c2p":
            text = tc["chinese"]
            expected = tc.get("expected_paiwan", "")
        else:
            text = tc["paiwan"]
            expected = tc.get("chinese", "")
        
        print(f"\n--- {text} ({tc['type']}, {direction}) ---")
        print(f"  期望: {expected}")
        
        # 只跑 RAG 檢索，不調 LLM
        rag_results = translator._hybrid_search(text, direction, top_k=5)
        
        if not rag_results:
            print(f"  ❌ RAG 無結果")
            not_found += 1
            results.append({"input": text, "expected": expected, "rag_top": [], "verdict": "not_found"})
            continue
        
        # 看前 3 個結果
        top_matches = [(r.get("paiwan", ""), r.get("chinese", ""), r.get("similarity", 0)) for r in rag_results[:3]]
        for p, c, s in top_matches:
            print(f"  RAG: {p} = {c} (sim={s:.3f})")
        
        # 檢查是否有匹配
        if direction == "p2c":
            # 排灣→中文：看 RAG 返回的中文是否包含期望
            matched = any(expected in c or c in expected for _, c, _ in top_matches)
            exact = any(c == expected for _, c, _ in top_matches)
        else:
            # 中文→排灣：看 RAG 返回的排灣語是否包含期望
            matched = any(expected in p or p in expected for p, _, _ in top_matches)
            exact = any(p == expected for p, _, _ in top_matches)
        
        if exact:
            print(f"  ✅ 精確匹配")
            correct += 1
            verdict = "correct"
        elif matched:
            print(f"  ⚠️ 部分匹配")
            partial += 1
            verdict = "partial"
        else:
            print(f"  ❌ 不匹配")
            wrong += 1
            verdict = "wrong"
        
        results.append({"input": text, "expected": expected, "rag_top": top_matches, "verdict": verdict})
    
    # 統計
    total = len(results)
    print("\n" + "=" * 70)
    print(f"  📊 RAG 召回結果 ({total} 筆)")
    print("=" * 70)
    print(f"  ✅ 精確匹配: {correct} ({correct/total*100:.1f}%)")
    print(f"  ⚠️ 部分匹配: {partial} ({partial/total*100:.1f}%)")
    print(f"  ❌ 不匹配:   {wrong} ({wrong/total*100:.1f}%)")
    print(f"  ❓ 無結果:    {not_found} ({not_found/total*100:.1f}%)")
    
    # 分類統計
    words = [r for r, tc in zip(results, TEST_CASES) if tc["type"] == "word"]
    sents = [r for r, tc in zip(results, TEST_CASES) if tc["type"] == "sentence"]
    
    print(f"\n  📊 按類型:")
    w_correct = sum(1 for r in words if r["verdict"] == "correct")
    s_correct = sum(1 for r in sents if r["verdict"] == "correct")
    print(f"  單詞: {w_correct}/{len(words)} ({w_correct/max(len(words),1)*100:.0f}%)")
    print(f"  句子: {s_correct}/{len(sents)} ({s_correct/max(len(sents),1)*100:.0f}%)")


if __name__ == "__main__":
    test_rag_only()
