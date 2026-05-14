#!/usr/bin/env python3
"""
klokah_verify.py — 用 klokah API 驗證系統翻譯正確率

用法：
    python3 scripts/klokah_verify.py [--sample 50] [--output results/klokah_verify.json]

從語料庫抽樣，拿中文詞去 klokah 查東排灣語標準答案，
再跟系統語料庫裡的排灣語比對。
"""

import json
import sys
import time
import random
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "merged_corpus.json"
KLOKAH_URL = "https://web.klokah.tw/php/multiSearchResult.php?d=23&txt={}"


def query_klokah(chinese_word: str) -> list[dict]:
    """查詢 klokah API，返回 [{paiwan, chinese}] 列表"""
    url = KLOKAH_URL.format(urllib.parse.quote(chinese_word))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_text = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  ⚠️ API 錯誤: {e}")
        return []

    results = []
    try:
        root = ET.fromstring(xml_text)
        # 搜尋所有分類
        for category in ["conversation", "vocabulary", "custom", "speech", "nine", "twelve"]:
            cat = root.find(category)
            if cat is None:
                continue
            count = cat.find("count")
            if count is not None and int(count.text or "0") == 0:
                continue
            for item in cat.findall("item"):
                paiwan = item.find("text")
                chinese = item.find("chinese")
                if paiwan is not None and chinese is not None:
                    results.append({
                        "paiwan": (paiwan.text or "").strip(),
                        "chinese": (chinese.text or "").strip(),
                    })
    except ET.ParseError:
        pass
    return results


def extract_key_word(chinese: str) -> str:
    """從中文句子中提取關鍵詞（取短詞優先）"""
    # 去掉標點
    clean = chinese.strip("。，！？、：；""''「」『』（）()")
    if len(clean) <= 4:
        return clean
    # 取前 2-3 個字作為關鍵詞
    return clean[:3]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50, help="抽樣數量")
    parser.add_argument("--output", type=str, default="results/klokah_verify.json")
    args = parser.parse_args()

    # 載入語料庫
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    # 過濾：只要短詞（≤6字中文），因為 klokah 適合查詞不適合查長句
    short_entries = [
        e for e in corpus
        if len(e.get("chinese", "")) <= 6
        and len(e.get("paiwan", "")) <= 20
    ]
    print(f"語料庫總數: {len(corpus)}, 短詞: {len(short_entries)}")

    # 抽樣
    sample_size = min(args.sample, len(short_entries))
    sample = random.sample(short_entries, sample_size)

    results = []
    correct = 0
    partial = 0
    not_found = 0
    wrong = 0

    for i, entry in enumerate(sample):
        chinese = entry["chinese"]
        our_paiwan = entry["paiwan"]

        print(f"\n[{i+1}/{sample_size}] {chinese} → 系統: {our_paiwan}")

        # 查 klokah
        klokah_results = query_klokah(chinese)
        time.sleep(0.3)  # 溫和請求

        if not klokah_results:
            # 嘗試用關鍵詞搜
            keyword = extract_key_word(chinese)
            if keyword != chinese:
                klokah_results = query_klokah(keyword)
                time.sleep(0.3)

        if not klokah_results:
            print(f"  ❌ klokah 查無")
            not_found += 1
            results.append({
                "chinese": chinese,
                "our_paiwan": our_paiwan,
                "klokah_results": [],
                "verdict": "not_found",
            })
            continue

        # 比對
        klokah_paiwans = [r["paiwan"] for r in klokah_results]
        print(f"  klokah: {klokah_paiwans[:5]}")

        # 精確匹配
        if any(our_paiwan == kp for kp in klokah_paiwans):
            print(f"  ✅ 精確匹配")
            correct += 1
            verdict = "correct"
        elif any(our_paiwan in kp or kp in our_paiwan for kp in klokah_paiwans):
            print(f"  ⚠️ 部分匹配")
            partial += 1
            verdict = "partial"
        else:
            print(f"  ❌ 不匹配")
            wrong += 1
            verdict = "wrong"

        results.append({
            "chinese": chinese,
            "our_paiwan": our_paiwan,
            "klokah_results": klokah_results[:5],
            "verdict": verdict,
        })

    # 統計
    total = len(results)
    print("\n" + "=" * 60)
    print(f"  📊 klokah 比對結果 ({total} 筆抽樣)")
    print("=" * 60)
    print(f"  ✅ 精確匹配: {correct} ({correct/total*100:.1f}%)")
    print(f"  ⚠️ 部分匹配: {partial} ({partial/total*100:.1f}%)")
    print(f"  ❌ 不匹配:   {wrong} ({wrong/total*100:.1f}%)")
    print(f"  ❓ klokah查無: {not_found} ({not_found/total*100:.1f}%)")
    print(f"\n  有效比對正確率（排除查無）: {correct}/{total-not_found} = {correct/(total-not_found)*100:.1f}%")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "correct": correct,
                "partial": partial,
                "wrong": wrong,
                "not_found": not_found,
                "accuracy": f"{correct/(total-not_found)*100:.1f}%" if total > not_found else "N/A",
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 結果已保存到 {output_path}")


if __name__ == "__main__":
    main()
