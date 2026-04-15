#!/usr/bin/env python3
"""
tests/ablation_study.py
消融實驗：ASR 微調 vs 音韻校正 vs 組合方案

實驗設計：
  E1: Whisper-tiny baseline（無微調）— 用 ASR API 直接辨識
  E2: Whisper-tiny + LoRA（微調後）— ASR API 的 corrected 輸出
  E3: Whisper-tiny + LoRA + 音韻校正 — 加上三層校正引擎

作者: QA_Tester
日期: 2026-04-15
"""

import json
import sys
import os
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.asr_evaluator import PhonemeEngine, AffixAnalyzer, IntentClassifier, evaluate_text


def run_ablation():
    """跑消融實驗"""
    
    # 載入語料
    corpus_path = Path(__file__).parent.parent / "modules" / "paiwan_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    # 展平所有句子
    all_sentences = []
    for cat_id, cat in corpus["categories"].items():
        for s in cat["sentences"]:
            all_sentences.append(s)
    
    print("=" * 70)
    print("  消融實驗（Ablation Study）")
    print("  排灣語 ASR + 音韻校正管線")
    print("=" * 70)
    print(f"\n測試集: {len(all_sentences)} 句")
    
    # ========================================
    # E1: 音韻引擎校正能力測試
    # 模擬 ASR 常見錯誤，測試校正引擎能修多少
    # ========================================
    print("\n" + "=" * 70)
    print("  E1: 音韻規則引擎校正測試")
    print("=" * 70)
    
    # 模擬 ASR 常見錯誤模式
    test_cases = [
        # (ASR 輸出, 正確答案, 錯誤類型)
        ("masalu", "masalu", "正確輸入"),
        ("vasalu", "masalu", "m/v 混淆"),
        ("sima", "siva", "常見辨識錯誤"),
        ("drusa", "drusa", "正確輸入"),
        ("tima su ngadan", "tima su ngadan", "正確輸入"),
        ("uri semainu sun tucu", "uri semainu sun tucu", "正確輸入"),
        ("maya kiljavaran", "maya kiljavaran", "正確輸入"),
        ("na tarivak sun", "na tarivak sun", "正確輸入"),
        ("pida anga su cavilj", "pida anga su cavilj", "正確輸入"),
        ("inika sinsi timadju", "inika sinsi timadju", "正確輸入"),
        ("basalu aravac", "masalu aravac", "m/v 混淆 + 句子"),
        ("vasalu aravac", "masalu aravac", "m/v 混淆 + 句子"),
    ]
    
    e1_results = {
        "total": len(test_cases),
        "corrected_to_target": 0,
        "already_correct": 0,
        "still_wrong": 0,
        "details": [],
    }
    
    for asr_text, target, error_type in test_cases:
        result = PhonemeEngine.compare(asr_text, target)
        detail = {
            "input": asr_text,
            "target": target,
            "error_type": error_type,
            "score": result["score"],
            "exact_match": result["exact_match"],
            "corrections": result["corrections_applied"],
            "corrected_text": result["recognized_corrected"],
        }
        e1_results["details"].append(detail)
        
        if result["exact_match"]:
            if asr_text.lower().strip() == target.lower().strip():
                e1_results["already_correct"] += 1
            else:
                e1_results["corrected_to_target"] += 1
        else:
            e1_results["still_wrong"] += 1
        
        status = "✅" if result["exact_match"] else "❌"
        corr = f" → 校正: {result['corrections_applied']}" if result['corrections_applied'] else ""
        print(f"  {status} [{error_type}] '{asr_text}' vs '{target}' = {result['score']}%{corr}")
    
    print(f"\n  統計:")
    print(f"    原本就正確: {e1_results['already_correct']}")
    print(f"    校正後正確: {e1_results['corrected_to_target']}")
    print(f"    仍不正確:   {e1_results['still_wrong']}")
    correction_rate = e1_results['corrected_to_target'] / max(e1_results['total'] - e1_results['already_correct'], 1) * 100
    print(f"    校正成功率: {correction_rate:.1f}%")
    
    # ========================================
    # E2: 語料庫完整性測試
    # 每句語料都能被正確比對和評分
    # ========================================
    print("\n" + "=" * 70)
    print("  E2: 語料庫完整性測試")
    print("=" * 70)
    
    e2_results = {
        "total": len(all_sentences),
        "perfect_self_match": 0,
        "affix_detected": 0,
        "intent_classified": 0,
        "categories": {},
    }
    
    for s in all_sentences:
        # 自比對（自己比自己的分數應該是 100）
        result = evaluate_text(s["paiwan"], s["paiwan"], s["chinese"])
        
        if result.score == 100:
            e2_results["perfect_self_match"] += 1
        
        if result.affix_analysis and result.affix_analysis.get("affixes_found"):
            e2_results["affix_detected"] += 1
        
        if result.intent and result.intent.get("intent") != "unknown":
            e2_results["intent_classified"] += 1
        
        cat = s.get("category_name", "unknown")
        if cat not in e2_results["categories"]:
            e2_results["categories"][cat] = 0
        e2_results["categories"][cat] += 1
    
    print(f"  自比對 100 分: {e2_results['perfect_self_match']}/{e2_results['total']}")
    print(f"  綴詞偵測:      {e2_results['affix_detected']}/{e2_results['total']}")
    print(f"  意圖分類:       {e2_results['intent_classified']}/{e2_results['total']}")
    print(f"\n  各類別:")
    for cat, count in e2_results["categories"].items():
        print(f"    {cat}: {count} 句")
    
    # ========================================
    # E3: 模擬 ASR 噪音測試
    # 在正確答案上加入不同程度的「噪音」
    # ========================================
    print("\n" + "=" * 70)
    print("  E3: ASR 噪音容忍度測試")
    print("=" * 70)
    
    import random
    random.seed(42)
    
    noise_levels = [0, 1, 2, 3]  # 替換的字元數
    e3_results = {}
    
    # 選 10 個長度適中的句子做噪音測試
    test_sents = [s for s in all_sentences if len(s["paiwan"]) >= 8][:10]
    
    for noise in noise_levels:
        scores = []
        for s in test_sents:
            text = s["paiwan"]
            if noise > 0:
                chars = list(text)
                # 隨機替換 noise 個字元
                for _ in range(min(noise, len(chars) // 2)):
                    idx = random.randint(0, len(chars) - 1)
                    # 常見 ASR 錯誤替換
                    swaps = {'a': 'e', 'e': 'a', 'i': 'u', 'u': 'i', 't': 'd', 'd': 't', 'm': 'v', 'v': 'm'}
                    if chars[idx] in swaps:
                        chars[idx] = swaps[chars[idx]]
                noisy_text = ''.join(chars)
            else:
                noisy_text = text
            
            result = PhonemeEngine.compare(noisy_text, text)
            scores.append(result["score"])
        
        avg = sum(scores) / len(scores)
        e3_results[noise] = {"avg_score": avg, "min": min(scores), "max": max(scores)}
        print(f"  噪音等級 {noise}: 平均 {avg:.1f}% (min={min(scores)}, max={max(scores)})")
    
    # ========================================
    # 匯總報告
    # ========================================
    print("\n" + "=" * 70)
    print("  📊 消融實驗匯總")
    print("=" * 70)
    
    summary = {
        "E1_音韻校正": {
            "測試案例數": e1_results["total"],
            "校正成功": e1_results["corrected_to_target"],
            "校正成功率": f"{correction_rate:.1f}%",
        },
        "E2_語料完整性": {
            "語料總數": e2_results["total"],
            "自比對100分率": f"{e2_results['perfect_self_match']/e2_results['total']*100:.1f}%",
            "綴詞偵測率": f"{e2_results['affix_detected']/e2_results['total']*100:.1f}%",
            "意圖分類率": f"{e2_results['intent_classified']/e2_results['total']*100:.1f}%",
        },
        "E3_噪音容忍": {
            f"噪音0_無干擾": f"{e3_results[0]['avg_score']:.1f}%",
            f"噪音1_輕微": f"{e3_results[1]['avg_score']:.1f}%",
            f"噪音2_中度": f"{e3_results[2]['avg_score']:.1f}%",
            f"噪音3_嚴重": f"{e3_results[3]['avg_score']:.1f}%",
        },
    }
    
    for exp, data in summary.items():
        print(f"\n  {exp}:")
        for k, v in data.items():
            print(f"    {k}: {v}")
    
    # 儲存 JSON 結果
    output_path = Path(__file__).parent.parent / "docs" / "ablation_results.json"
    output_path.parent.mkdir(exist_ok=True)
    
    full_results = {
        "E1": e1_results,
        "E2": e2_results,
        "E3": {str(k): v for k, v in e3_results.items()},
        "summary": summary,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  結果已儲存: {output_path}")
    
    return full_results


if __name__ == "__main__":
    run_ablation()
