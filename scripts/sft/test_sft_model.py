#!/usr/bin/env python3
"""
test_sft_model.py
測試微調後的排灣語翻譯模型

比較原始模型 vs 微調模型 vs RAG 的翻譯效果

作者: Claude
日期: 2026-05-12
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent
ZHIPU_API_KEY = os.getenv("ZHIPUAI_API_KEY")

client = OpenAI(
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)


# ============================================
# 測試用例
# ============================================

TEST_CASES = [
    # 基礎詞彙（應該在訓練數據中）
    {
        "input": "masalu",
        "direction": "排灣語→中文",
        "expected": "謝謝",
        "difficulty": "簡單"
    },
    {
        "input": "你好嗎",
        "direction": "中文→排灣語",
        "expected": "na tarivak sun?",
        "difficulty": "簡單"
    },
    {
        "input": "na masalu su vuvu",
        "direction": "排灣語→中文",
        "expected": "你的祖母很好",
        "difficulty": "中等"
    },
    {
        "input": "tima su ngadan?",
        "direction": "排灣語→中文",
        "expected": "你叫什麼名字？",
        "difficulty": "中等"
    },
    {
        "input": "你幾歲",
        "direction": "中文→排灣語",
        "expected": "pida anga su cavilj?",
        "difficulty": "中等"
    },
    # 語法變化（測試 SFT 效果）
    {
        "input": "uri namakuda a kalevelevan i tjanumun?",
        "direction": "排灣語→中文",
        "expected": "你們那邊天氣如何？",
        "difficulty": "困難"
    },
    {
        "input": "如果明天不下雨，我們就去山里",
        "direction": "中文→排灣語",
        "expected": None,  # 開放式，檢查流暢度
        "difficulty": "困難"
    },
    # 複雜句
    {
        "input": "macaqu sun aravac a penayuanan, tima na temulu tjanusun?",
        "direction": "排灣語→中文",
        "expected": "你的族語說的很好，都是誰教你的？",
        "difficulty": "困難"
    },
]


# ============================================
# 翻譯測試
# ============================================

def translate_with_model(model: str, text: str, direction: str = "auto") -> dict:
    """
    使用指定模型進行翻譯

    Args:
        model: 模型名稱（如 "glm-4-flash" 或微調模型 ID）
        text: 輸入文本
        direction: 翻譯方向

    Returns:
        {"translation": "...", "raw_response": ...}
    """
    # 判斷方向
    is_paiwan = any(c.isalpha() and ord(c) < 128 for c in text) and not any('一' <= c <= '鿿' for c in text)

    if direction == "auto":
        direction = "排灣語→中文" if is_paiwan else "中文→排灣語"

    system_prompt = """你是專業的排灣語-中文翻譯助手。請準確地進行雙向翻譯。

規則：
1. 直接輸出翻譯結果，不要任何解釋
2. 保持語句的自然和準確
3. 如果是問句，保留問句標點"""

    if direction == "排灣語→中文":
        user_msg = f"請將以下排灣語翻譯成中文：{text}"
    else:
        user_msg = f"請將以下中文翻譯成排灣語：{text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=200,
        )

        translation = response.choices[0].message.content.strip()

        return {
            "translation": translation,
            "raw_response": response,
            "model": model
        }
    except Exception as e:
        return {
            "translation": f"[錯誤: {e}]",
            "raw_response": None,
            "model": model
        }


def calculate_score(predicted: str, expected: str) -> float:
    """計算翻譯準確率分數（簡單版）"""
    if not expected:
        return None  # 開放式問題，不評分

    predicted = predicted.strip("。！？：；，, ")
    expected = expected.strip("。！？：；，, ")

    # 完全匹配
    if predicted == expected:
        return 1.0

    # 包含關鍵詞
    if expected in predicted or predicted in expected:
        return 0.7

    # 部分匹配（字符重疊度）
    overlap = len(set(predicted) & set(expected))
    total = len(set(expected))
    if total > 0:
        return overlap / total * 0.5

    return 0.0


def run_test_suite(model: str, model_name: str = "模型") -> dict:
    """運行完整測試套件"""
    print(f"\n{'=' * 60}")
    print(f"🧪 測試模型: {model_name} ({model})")
    print(f"{'=' * 60}")

    results = []
    scores = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}] {case['difficulty']} - {case['direction']}")
        print(f"    輸入: {case['input']}")

        result = translate_with_model(model, case['input'], case['direction'])
        translation = result['translation']

        print(f"    輸出: {translation}")

        # 評分
        if case['expected']:
            score = calculate_score(translation, case['expected'])
            scores.append(score)
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.3 else "❌"
            print(f"    預期: {case['expected']}")
            print(f"    分數: {score:.1f} {status}")
        else:
            score = None
            print(f"    預期: [開放式]")

        results.append({
            "case": case,
            "result": result,
            "score": score
        })

    # 總結
    print(f"\n{'─' * 60}")
    print(f"📊 測試總結: {model_name}")
    print(f"{'─' * 60}")

    if scores:
        avg_score = sum(scores) / len(scores)
        pass_rate = sum(1 for s in scores if s >= 0.7) / len(scores)

        print(f"平均分數: {avg_score:.2f}")
        print(f"通過率 (≥0.7): {pass_rate:.1%}")
    else:
        print(f"無法計算分數（開放式問題）")

    return {
        "model": model,
        "model_name": model_name,
        "results": results,
        "avg_score": sum(scores) / len(scores) if scores else None,
        "pass_rate": sum(1 for s in scores if s >= 0.7) / len(scores) if scores else None
    }


def compare_models(base_model: str = "glm-4-flash", finetuned_model: str = None):
    """比較基礎模型和微調模型"""

    print("=" * 60)
    print("🔄 排灣語翻譯模型對比測試")
    print("=" * 60)

    # 測試基礎模型
    base_results = run_test_suite(base_model, "基礎模型 (glm-4-flash)")

    # 測試微調模型
    if finetuned_model:
        ft_results = run_test_suite(finetuned_model, f"微調模型 ({finetuned_model})")

        # 對比
        print(f"\n{'=' * 60}")
        print("📈 對比結果")
        print(f"{'=' * 60}")

        if base_results['avg_score'] and ft_results['avg_score']:
            improvement = ft_results['avg_score'] - base_results['avg_score']
            print(f"基礎模型平均分: {base_results['avg_score']:.2f}")
            print(f"微調模型平均分: {ft_results['avg_score']:.2f}")
            print(f"提升幅度: {improvement:+.2f} ({improvement/base_results['avg_score']*100:+.1f}%)")


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="測試微調模型")
    parser.add_argument("--model", type=str, default="glm-4-flash",
                        help="要測試的模型（默認：glm-4-flash）")
    parser.add_argument("--finetuned", type=str,
                        help="微調模型 ID（用於對比）")
    parser.add_argument("--single", type=str,
                        help="單句測試")
    parser.add_argument("--interactive", action="store_true",
                        help="互動式測試")

    args = parser.parse_args()

    # 單句測試
    if args.single:
        result = translate_with_model(args.model, args.single)
        print(f"\n輸入: {args.single}")
        print(f"輸出: {result['translation']}")
        return

    # 互動式測試
    if args.interactive:
        print("\n🎮 互動式翻譯測試 (Ctrl+C 退出)")
        print(f"使用模型: {args.model}\n")

        while True:
            try:
                text = input("請輸入要翻譯的文本: ").strip()
                if not text:
                    continue

                result = translate_with_model(args.model, text)
                print(f"→ {result['translation']}\n")
            except KeyboardInterrupt:
                print("\n\n👋 再見！")
                break

    # 對比測試
    compare_models(
        base_model=args.model,
        finetuned_model=args.finetuned
    )


if __name__ == "__main__":
    main()
