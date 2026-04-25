#!/usr/bin/env python3
"""
paiwan_ipa.py — 排灣語拼字 → IPA 映射表

排灣語使用拉丁字母書寫系統，其中有幾個特殊拼法對應非英文音素。
XTTS v2 不支援排灣語（pag），以英文模式（en）合成時，
會按照英文發音規則讀排灣語拼字，導致嚴重偏差。

本映射表將排灣語拼字轉換為 IPA，再用英文近似音素重新拼寫，
讓 XTTS v2 在英文模式下輸出更接近排灣語的發音。

排灣語音系參考：
- 輔音：p, t, k, q（小舌塞音）, '（喉塞音）, c（清齒齦塞擦音）, s, z, m, n, ng（ŋ）, l, lj（ʎ）, lj, r, v, y, w
- 元音：a, i, u, e
- 特殊拼字：tj（清齒齦塞音，接近 tʃ）, dj（濁齒齦塞音）, lj（硬顎邊音 ʎ）, ng（ŋ）, q（小舌音）

作者: 地陪
日期: 2026-04-18
"""

# ============================================
# 排灣語拼字 → 英文近似音素映射
# ============================================
# 策略：將排灣語特殊音素轉換為英文拼字中發音最接近的組合
# 這不是精確的 IPA，而是「XTTS 英文模式能正確發音的近似拼法」

PAIWAN_TO_ENGLISH_APPROX = {
    # --- 特殊輔音組合（必須先處理）---
    "tj": "ch",      # 清齦後塞擦音 → ch (如 tjelu → chelu)
    "dj": "j",       # 濁齦後塞擦音 → j (如 ladjap → lajap)  
    "lj": "ly",      # 硬顎邊音 → ly (如 ljavek → lyavek)
    "ng": "ng",      # 軟顎鼻音 ŋ → ng 保持不變（英文模式能處理）

    # --- 單輔音 ---
    "q": "k",        # 小舌塞音 → k（XTTS 會把 q 發成奇怪的音）
    "c": "ts",       # 清齒齦塞擦音 → ts (如 ceqer → tseker)
    "z": "z",        # 濁齒齦擦音 → 保持

    # --- 元音（排灣語只有 a, i, u, e 四個，較規則）---
    # 大部分保持不變，英文模式讀得還行
}


def paiwan_to_english_approx(text: str) -> str:
    """
    將排灣語文字轉換為英文近似拼寫，供 XTTS v2 英文模式使用。
    
    處理順序很重要：先處理多字符組合（tj, dj, lj, ng），再處理單字符（q, c）。
    """
    result = text.lower().strip()
    
    # 先處理多字符映射
    for paiwan, english in PAIWAN_TO_ENGLISH_APPROX.items():
        if len(paiwan) > 1:
            result = result.replace(paiwan, english)
    
    # 再處理單字符映射
    for paiwan, english in PAIWAN_TO_ENGLISH_APPROX.items():
        if len(paiwan) == 1:
            result = result.replace(paiwan, english)
    
    return result


# ============================================
# 測試用：對照表
# ============================================
TEST_WORDS = {
    # 數字
    "ita": None,        # 1
    "drusa": None,      # 2
    "tjelu": None,      # 3
    "sepatj": None,     # 4
    "lima": None,       # 5
    "unem": None,       # 6
    "pitju": None,      # 7
    "alu": None,        # 8
    "siva": None,       # 9
    "tapuluq": None,    # 10
    
    # 常用詞
    "masalu": None,     # 謝謝
    "qadaw": None,      # 太陽/日子
    "vatu": None,       # 狗
    "sapuy": None,      # 火
    "quma": None,       # 土地
    "cemdas": None,     # 美
    "ceqer": None,      # 壞
    "drail": None,      # 好吃
    
    # 句子
    "na tarivak sun": None,
    "tima su ngadan": None,
    "uri semainu sun": None,
    "maya kiljavaran": None,
}


if __name__ == "__main__":
    print("排灣語 → 英文近似音素 對照表")
    print("=" * 60)
    print(f"{'排灣語原文':<25} {'英文近似':<25}")
    print("-" * 60)
    
    # 測試詞彙
    all_words = list(TEST_WORDS.keys())
    # 加入所有語料庫詞彙
    import csv
    from pathlib import Path
    dataset_path = Path(__file__).parent / "dataset" / "metadata.csv"
    if dataset_path.exists():
        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row['text']
                if word not in all_words:
                    all_words.append(word)
    
    for word in all_words:
        converted = paiwan_to_english_approx(word)
        marker = " ✓" if converted != word else " ="
        print(f"{word:<25} {converted:<25} {marker}")
    
    print("\n" + "=" * 60)
    print("映射規則：")
    for k, v in PAIWAN_TO_ENGLISH_APPROX.items():
        print(f"  {k} → {v}")
