#!/usr/bin/env python3
"""
verify_loop_translations.py — 從 checkpoint 提取 Loop 翻譯結果，用 klokah 驗證
零 token 消耗，只調 klokah API
"""

import json
import time
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from ast import literal_eval

PROJECT_ROOT = Path(__file__).parent.parent
KLOKAH_URL = "https://web.klokah.tw/php/multiSearchResult.php?d=23&txt={}"


def query_klokah(word: str) -> list[dict]:
    """查詢 klokah API"""
    url = KLOKAH_URL.format(urllib.parse.quote(word))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_text = resp.read().decode("utf-8")
    except Exception as e:
        return []

    results = []
    try:
        root = ET.fromstring(xml_text)
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


def extract_translations_from_reply(reply: str) -> list[tuple]:
    """從 Orchestrator 回覆中提取翻譯對 (排灣語, 中文)"""
    pairs = []

    # Pattern 1: **排灣語** - 中文 / **排灣語** → 中文
    for m in re.finditer(r'\*\*([a-zāáǎàēéěèīíǐìōóǒòūúǔù]+[.!？?]?)\*\*\s*[-–→→]\s*([^\n*]+)', reply):
        paiwan = m.group(1).strip()
        chinese = m.group(2).strip()
        pairs.append((paiwan, chinese))

    # Pattern 2: 排灣語 → 中文 (no bold)
    for m in re.finditer(r'([a-zāáǎàēéěèīíǐìōóǒòūúǔù]{2,}[.!？?]?)\s*→\s*([^\n*,]+)', reply):
        paiwan = m.group(1).strip()
        chinese = m.group(2).strip()
        if paiwan and chinese and len(paiwan) > 1:
            pairs.append((paiwan, chinese))

    return pairs


def main():
    # 讀取 checkpoint
    ckpt_path = PROJECT_ROOT / "agent_framework/storage/state/ckpt_sess_cb55361c_1778655436.json"
    with open(ckpt_path) as f:
        ckpt = json.load(f)

    iteration_results = ckpt['state']['iteration_results']
    print(f"共 {len(iteration_results)} 個 Loop\n")

    # 收集所有翻譯對
    all_pairs = []
    for item in iteration_results:
        if isinstance(item, str):
            try:
                data = literal_eval(item)
            except:
                continue
        else:
            data = item

        loop_num = data.get('loop', '?')
        result = data.get('result', {})
        data_inner = result.get('data', {})
        reply = data_inner.get('reply', '')

        if not reply:
            continue

        pairs = extract_translations_from_reply(reply)
        for p in pairs:
            all_pairs.append((loop_num, p[0], p[1], reply))

    # 去重
    seen = set()
    unique_pairs = []
    for loop_num, paiwan, chinese, reply in all_pairs:
        key = (paiwan, chinese)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((loop_num, paiwan, chinese))

    print(f"提取到 {len(all_pairs)} 個翻譯對，去重後 {len(unique_pairs)} 個\n")
    print("=" * 70)

    # 逐個驗證
    correct = 0
    partial = 0
    wrong = 0
    not_found = 0
    verified = []

    for i, (loop_num, paiwan, chinese) in enumerate(unique_pairs):
        print(f"\n[{i+1}/{len(unique_pairs)}] Loop #{loop_num}: {paiwan} = {chinese}")

        # 用排灣語查 klokah
        klokah_results = query_klokah(paiwan)
        time.sleep(0.3)

        if not klokah_results:
            # 用中文查
            klokah_results = query_klokah(chinese)
            time.sleep(0.3)

        if not klokah_results:
            print(f"  ❓ klokah 查無")
            not_found += 1
            verified.append({"paiwan": paiwan, "chinese": chinese, "loop": loop_num,
                            "verdict": "not_found", "klokah": []})
            continue

        # 檢查是否有匹配
        klokah_chinese = [r["chinese"] for r in klokah_results]
        klokah_paiwan = [r["paiwan"] for r in klokah_results]

        # 中文含義是否匹配
        match = False
        partial_match = False
        for kc in klokah_chinese:
            if chinese in kc or kc in chinese:
                match = True
                break
            # 檢查是否有部分詞匹配
            chinese_chars = set(re.findall(r'[\u4e00-\u9fff]', chinese))
            kc_chars = set(re.findall(r'[\u4e00-\u9fff]', kc))
            if chinese_chars and kc_chars and chinese_chars & kc_chars:
                partial_match = True

        if match:
            print(f"  ✅ 匹配（klokah: {klokah_chinese[:3]}）")
            correct += 1
            verdict = "correct"
        elif partial_match:
            print(f"  ⚠️ 部分匹配（klokah: {klokah_chinese[:3]}）")
            partial += 1
            verdict = "partial"
        else:
            print(f"  ❌ 不匹配！klokah 說 {paiwan} = {klokah_chinese[:3]}，Loop 說 = {chinese}")
            wrong += 1
            verdict = "wrong"

        verified.append({
            "paiwan": paiwan, "chinese": chinese, "loop": loop_num,
            "verdict": verdict,
            "klokah_chinese": klokah_chinese[:5],
            "klokah_paiwan": klokah_paiwan[:5],
        })

    # 統計
    total = len(verified)
    print("\n" + "=" * 70)
    print(f"  📊 Loop 翻譯驗證結果 ({total} 筆)")
    print("=" * 70)
    print(f"  ✅ 正確:     {correct} ({correct/total*100:.1f}%)")
    print(f"  ⚠️ 部分匹配: {partial} ({partial/total*100:.1f}%)")
    print(f"  ❌ 不匹配:   {wrong} ({wrong/total*100:.1f}%)")
    print(f"  ❓ klokah查無: {not_found} ({not_found/total*100:.1f}%)")

    if wrong > 0:
        print(f"\n  🔴 錯誤翻譯列表：")
        for v in verified:
            if v["verdict"] == "wrong":
                print(f"    Loop #{v['loop']}: {v['paiwan']} ≠ {v['chinese']}")
                print(f"      klokah 說: {v.get('klokah_chinese', [])}")

    # 保存
    output = PROJECT_ROOT / "results/loop_translation_verify.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"total": total, "correct": correct, "partial": partial,
                       "wrong": wrong, "not_found": not_found},
            "details": verified,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 保存到 {output}")


if __name__ == "__main__":
    main()
