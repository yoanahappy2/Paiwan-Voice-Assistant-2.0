"""
tests/asr_benchmark.py
語聲同行 2.0 — ASR 評測腳本

測試 Whisper-tiny + LoRA 微調模型的 WER (字錯率)
並與微調前（base model）做對比

作者: QA_Tester
日期: 2026-04-15
"""

import json
import os
import time
import requests
from pathlib import Path
from datetime import datetime

# ============================================
# 配置
# ============================================

ASR_API_URL = "http://127.0.0.1:8000/api/audio/recognize"
HEALTH_URL = "http://127.0.0.1:8000/health"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================
# WER 計算
# ============================================

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    計算 Word Error Rate (WER)
    
    WER = (S + D + I) / N
    S = 替換數, D = 刪除數, I = 插入數, N = 參考詞數
    """
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()

    n = len(ref_words)
    if n == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Dynamic Programming
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # 刪除
                    d[i][j - 1] + 1,      # 插入
                    d[i - 1][j - 1] + 1   # 替換
               )

    wer = d[len(ref_words)][len(hyp_words)] / n
    return min(wer, 1.0)  # 上限 100%


def calculate_cer(reference: str, hypothesis: str) -> float:
    """計算 Character Error Rate (CER)"""
    ref_chars = list(reference.lower().strip())
    hyp_chars = list(hypothesis.lower().strip())

    n = len(ref_chars)
    if n == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1
               )

    cer = d[len(ref_chars)][len(hyp_chars)] / n
    return min(cer, 1.0)


# ============================================
# 測試集
# ============================================

# 基準測試句子（涵蓋不同意圖類別）
TEST_SET = [
    {"paiwan": "na tarivak sun?", "chinese": "你好嗎？", "intent": "greeting"},
    {"paiwan": "tima su ngadan?", "chinese": "你叫什麼名字？", "intent": "identity_query"},
    {"paiwan": "masalu", "chinese": "謝謝！", "intent": "thanks"},
    {"paiwan": "pida anga su cavilj?", "chinese": "你幾歲？", "intent": "quantity_query"},
    {"paiwan": "anema cu?", "chinese": "這是什麼？", "intent": "object_query"},
    {"paiwan": "inuan a su kama?", "chinese": "你爸爸在哪裡？", "intent": "location_query"},
    {"paiwan": "maya kiljavaran!", "chinese": "別說話！", "intent": "prohibition"},
    {"paiwan": "ini, vuday azua.", "chinese": "不，那是玉米。", "intent": "negation"},
    {"paiwan": "uri semainu sun tucu?", "chinese": "你現在要去哪裡？", "intent": "location_query"},
    {"paiwan": "na kemasinu sun tucu?", "chinese": "你今天從哪裡來的？", "intent": "location_query"},
    {"paiwan": "nanguaq", "chinese": "好", "intent": "affirmation"},
    {"paiwan": "pacunan", "chinese": "再見", "intent": "farewell"},
    {"paiwan": "kasinu mun katiaw?", "chinese": "你們昨天去哪裡？", "intent": "location_query"},
    {"paiwan": "pida milingan tucu?", "chinese": "現在是幾點鐘？", "intent": "quantity_query"},
    {"paiwan": "mapida mun a taqumaqanan?", "chinese": "你們家有幾個人？", "intent": "quantity_query"},
]


# ============================================
# LLM 評測
# ============================================

def test_llm_responses():
    """測試 LLM 回覆品質（有 RAG vs 無 RAG）"""
    from llm_service import VuvuService

    test_inputs = [
        {"input": "masalu", "expected_keyword": "謝謝", "expected_paiwan": "masalu"},
        {"input": "你好嗎", "expected_keyword": "你好嗎", "expected_paiwan": "tarivak"},
        {"input": "tima su ngadan?", "expected_keyword": "名字", "expected_paiwan": "ngadan"},
        {"input": "你幾歲", "expected_keyword": "歲", "expected_paiwan": "cavilj"},
        {"input": "這是什麼", "expected_keyword": "什麼", "expected_paiwan": "anema"},
    ]

    results = {"rag": [], "legacy": []}

    for mode in ["rag", "legacy"]:
        use_rag = (mode == "rag")
        vuvu = VuvuService(use_rag=use_rag)

        for test in test_inputs:
            result = vuvu.chat_with_thinking(test["input"])
            reply = result["reply"].lower()

            keyword_hit = test["expected_keyword"].lower() in reply
            paiwan_hit = test["expected_paiwan"].lower() in reply

            results[mode].append({
                "input": test["input"],
                "expected_keyword": test["expected_keyword"],
                "keyword_hit": keyword_hit,
                "paiwan_hit": paiwan_hit,
                "reply_length": len(result["reply"]),
                "has_thinking": bool(result["thinking"]),
            })
            vuvu.reset()

    return results


# ============================================
# RAG 檢索品質評測
# ============================================

def test_rag_retrieval():
    """測試 RAG 檢索的 Top-1 / Top-3 準確率"""
    from rag_service import PaiwanRAG
    rag = PaiwanRAG()
    rag.build_index()

    # 每個查詢期望的 Top-1 結果
    test_queries = [
        {"query": "masalu", "expected_paiwan": "masalu!"},
        {"query": "你好嗎", "expected_paiwan": "na tarivak sun?"},
        {"query": "你叫什麼名字", "expected_paiwan": "tima su ngadan?"},
        {"query": "你幾歲", "expected_paiwan": "pida anga su cavilj?"},
        {"query": "這是什麼", "expected_paiwan": "anema cu?"},
        {"query": "你爸爸在哪裡", "expected_paiwan": "inuan a su kama?"},
        {"query": "別說話", "expected_paiwan": "maya kiljavaran!"},
        {"query": "謝謝", "expected_paiwan": "masalu!"},
        {"query": "再見", "expected_paiwan": "pacunan"},
        {"query": "pida anga su cavilj?", "expected_paiwan": "pida anga su cavilj?"},
    ]

    top1_correct = 0
    top3_correct = 0

    details = []
    for t in test_queries:
        results = rag.search(t["query"], top_k=3)
        top1_paiwan = results[0]["paiwan"] if results else ""
        top3_paiwans = [r["paiwan"] for r in results[:3]]

        # 模糊匹配（容忍標點差異）
        expected = t["expected_paiwan"].lower().replace("!", "").replace("?", "").strip()

        hit1 = any(expected in p.lower().replace("!", "").replace("?", "") for p in [top1_paiwan])
        hit3 = any(expected in p.lower().replace("!", "").replace("?", "") for p in top3_paiwans)

        if hit1: top1_correct += 1
        if hit3: top3_correct += 1

        details.append({
            "query": t["query"],
            "expected": t["expected_paiwan"],
            "top1": top1_paiwan,
            "top3": top3_paiwans,
            "top1_hit": hit1,
            "top3_hit": hit3,
        })

    total = len(test_queries)
    return {
        "top1_accuracy": top1_correct / total,
        "top3_accuracy": top3_correct / total,
        "total_queries": total,
        "details": details,
    }


# ============================================
# 主程式
# ============================================

def main():
    print("=" * 60)
    print("  語聲同行 2.0 — 技術評測報告")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    report = {
        "date": datetime.now().isoformat(),
        "asr": {"status": "pending"},
        "rag": {},
        "llm": {},
    }

    # ── 1. RAG 檢索評測 ──
    print("\n📊 [1/2] RAG 檢索品質評測...")
    try:
        rag_results = test_rag_retrieval()
        report["rag"] = rag_results
        print(f"  Top-1 準確率: {rag_results['top1_accuracy']:.1%}")
        print(f"  Top-3 準確率: {rag_results['top3_accuracy']:.1%}")
    except Exception as e:
        print(f"  ❌ RAG 評測失敗: {e}")
        report["rag"] = {"error": str(e)}

    # ── 2. LLM 回覆品質評測 ──
    print("\n📊 [2/2] LLM 回覆品質評測（RAG vs Legacy）...")
    try:
        llm_results = test_llm_responses()
        report["llm"] = llm_results

        for mode in ["rag", "legacy"]:
            hits = sum(1 for r in llm_results[mode] if r["keyword_hit"])
            total = len(llm_results[mode])
            print(f"  {mode.upper()} 關鍵詞命中率: {hits}/{total} ({hits/total:.1%})")
    except Exception as e:
        print(f"  ❌ LLM 評測失敗: {e}")
        report["llm"] = {"error": str(e)}

    # ── 儲存結果 ──
    output_file = RESULTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 評測結果已儲存: {output_file}")

    # ── 輸出摘要 ──
    print("\n" + "=" * 60)
    print("  評測摘要")
    print("=" * 60)
    if "top1_accuracy" in report.get("rag", {}):
        print(f"  RAG Top-1: {report['rag']['top1_accuracy']:.1%}")
        print(f"  RAG Top-3: {report['rag']['top3_accuracy']:.1%}")

    for mode in ["rag", "legacy"]:
        if mode in report.get("llm", {}):
            data = report["llm"][mode]
            if isinstance(data, list):
                hits = sum(1 for r in data if r.get("keyword_hit"))
                total = len(data)
                print(f"  LLM ({mode}) 命中率: {hits}/{total}")

    print("=" * 60)


if __name__ == "__main__":
    main()
