"""
crawl_klokah.py
從族語E樂園 (klokah.tw) 批量爬取排灣語詞彙
方言代碼: d=24 (排灣語)

輸出: data/klokah_paiwan_corpus.json
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import re
from pathlib import Path

BASE_URL = "https://web.klokah.tw/php/multiSearchResult.php"
DIALECT = 24  # 排灣語

# 搜索用的字元覆蓋（用每個排灣語常見起始字母）
SEARCH_TERMS = [
    # 單字母
    "a", "c", "d", "e", "i", "k", "l", "m", "n", "p", "q", "r", "s", "t", "u", "v", "z",
    # 雙字母（覆蓋更多）
    "ga", "ka", "la", "ma", "na", "pa", "qa", "sa", "ta", "va", "za",
    "ce", "ke", "le", "me", "ne", "pe", "se", "te",
    "dj", "tj", "lj", "nj",
    # 中文搜索
    "我", "你", "他", "我們", "你們", "他們",
    "是", "有", "在", "去", "來", "說", "做",
    "謝謝", "你好", "再見", "請",
]

def fetch_search(term: str) -> dict:
    """搜索一個詞，返回 XML 結果"""
    params = {"d": DIALECT, "txt": term, "f": "fuzzy"}
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  ❌ 搜索 '{term}' 失敗: {e}")
        return None

def parse_results(xml_text: str) -> list:
    """解析 XML 結果，提取詞彙對"""
    results = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results
    
    # 遍歷所有 section
    for section in root:
        for item in section.findall('.//item'):
            text_elem = item.find('text')
            chinese_elem = item.find('chinese')
            
            if text_elem is not None and chinese_elem is not None:
                paiwan = text_elem.text or ""
                chinese = chinese_elem.text or ""
                
                if paiwan.strip() and chinese.strip():
                    # 提取額外資訊
                    info = {
                        "paiwan": paiwan.strip(),
                        "chinese": chinese.strip(),
                    }
                    
                    # 句子資訊
                    sentence = item.find('sentence')
                    sentence_chinese = item.find('sentenceChinese')
                    if sentence is not None and sentence.text:
                        info["sentence_paiwan"] = sentence.text.strip()
                    if sentence_chinese is not None and sentence_chinese.text:
                        info["sentence_chinese"] = sentence_chinese.text.strip()
                    
                    # 難度/類型
                    difficulty = item.find('difficulty')
                    if difficulty is not None:
                        info["difficulty"] = difficulty.text
                    
                    type_elem = item.find('type')
                    if type_elem is not None:
                        info["type"] = type_elem.text
                    
                    # 教材來源
                    url_elem = item.find('url')
                    if url_elem is not None:
                        info["source_url"] = url_elem.text
                    
                    results.append(info)
    
    return results

def main():
    all_entries = {}  # 用 (paiwan, chinese) 做去重 key
    seen_ids = set()
    
    print(f"🏔️ 開始爬取族語E樂園排灣語詞彙")
    print(f"   方言代碼: {DIALECT}")
    print(f"   搜索詞數: {len(SEARCH_TERMS)}")
    print()
    
    for i, term in enumerate(SEARCH_TERMS):
        print(f"[{i+1}/{len(SEARCH_TERMS)}] 搜索: {term}")
        
        xml_text = fetch_search(term)
        if not xml_text:
            continue
        
        entries = parse_results(xml_text)
        new_count = 0
        
        for entry in entries:
            key = (entry["paiwan"], entry["chinese"])
            if key not in all_entries:
                all_entries[key] = entry
                new_count += 1
        
        print(f"  找到 {len(entries)} 條，新增 {new_count} 條（總計: {len(all_entries)}）")
        
        # 禮貌性延遲
        time.sleep(0.5)
    
    # 轉換為列表
    final_entries = list(all_entries.values())
    
    # 統計
    print()
    print(f"📊 爬取完成！")
    print(f"   總詞彙數: {len(final_entries)}")
    
    # 分類統計
    words = [e for e in final_entries if e.get("type") == "word" or (not e.get("sentence_paiwan") and len(e["paiwan"].split()) <= 3)]
    sentences = [e for e in final_entries if e.get("sentence_paiwan")]
    print(f"   詞彙: {len(words)}")
    print(f"   含例句: {len(sentences)}")
    
    # 存檔
    out_path = Path(__file__).parent / "data" / "klokah_paiwan_corpus.json"
    out_path.parent.mkdir(exist_ok=True)
    
    output = {
        "metadata": {
            "source": "族語E樂園 (klokah.tw)",
            "url": "https://web.klokah.tw/multiSearch/",
            "dialect": "排灣語 (Paiwan)",
            "dialect_code": 24,
            "total_entries": len(final_entries),
            "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "entries": final_entries,
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"   已存檔: {out_path}")
    print(f"   檔案大小: {out_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
