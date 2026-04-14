#!/usr/bin/env python3
"""
Klokah 族語資源爬蟲
爬取 https://web.klokah.tw 的排灣語單詞與音檔

作者: @Backend_Dev
日期: 2026-03-28
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import re
from urllib.parse import urljoin, urlparse
import time

# ============================================
# 配置
# ============================================

BASE_URL = "https://web.klokah.tw"
PRACTICE_URL = "https://web.klokah.tw/extension/rd_practice/index.php"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "database")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "klokah_dataset.json")

# 語言代碼：排灣語
LANGUAGE_ID = 15  # 排灣語

# ============================================
# 爬蟲類別
# ============================================

class KlokahCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        })
        self.dataset = {
            "metadata": {
                "source": "https://web.klokah.tw",
                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "language": "排灣語",
                "total_words": 0
            },
            "words": []
        }
    
    def fetch_page(self, url, params=None):
        """抓取頁面"""
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.encoding = 'utf-8'
            return response.text
        except Exception as e:
            print(f"[錯誤] 無法抓取 {url}: {e}")
            return None
    
    def parse_word_list(self, html):
        """解析單詞列表"""
        soup = BeautifulSoup(html, 'html.parser')
        words = []
        
        # 尋找單詞元素（需要根據實際頁面結構調整）
        # 常見的結構可能是 table, div, 或 li
        word_elements = soup.find_all(['tr', 'div'], class_=lambda x: x and 'word' in str(x).lower())
        
        if not word_elements:
            # 備用：尋找所有包含族語文字的元素
            word_elements = soup.find_all(['td', 'div', 'span'])
        
        for elem in word_elements:
            word_data = self.extract_word_data(elem)
            if word_data:
                words.append(word_data)
        
        return words
    
    def extract_word_data(self, elem):
        """從元素中提取單詞資料"""
        text = elem.get_text(strip=True)
        
        # 跳過空內容
        if not text or len(text) < 2:
            return None
        
        # 尋找音檔連結
        audio_url = None
        audio_elem = elem.find('audio')
        if audio_elem:
            source = audio_elem.find('source')
            if source and source.get('src'):
                audio_url = urljoin(BASE_URL, source['src'])
        
        # 尋找下載連結
        if not audio_url:
            for link in elem.find_all('a'):
                href = link.get('href', '')
                if any(ext in href.lower() for ext in ['.mp3', '.m4a', '.wav', '.ogg']):
                    audio_url = urljoin(BASE_URL, href)
                    break
        
        return {
            "text": text,
            "audio_url": audio_url
        }
    
    def crawl_practice_page(self, dialect_id=1):
        """爬取練習頁面"""
        print(f"[爬蟲] 開始爬取方言 {dialect_id}...")
        
        params = {
            'd': dialect_id,
            'l': LANGUAGE_ID,
            'view': 'article'
        }
        
        html = self.fetch_page(PRACTICE_URL, params)
        if not html:
            return []
        
        words = self.parse_word_list(html)
        print(f"[爬蟲] 找到 {len(words)} 個單詞")
        
        return words
    
    def crawl_all_dialects(self, max_dialects=10):
        """爬取所有方言"""
        all_words = []
        
        for d in range(1, max_dialects + 1):
            words = self.crawl_practice_page(d)
            all_words.extend(words)
            time.sleep(1)  # 避免過度請求
        
        return all_words
    
    def save_dataset(self):
        """儲存資料集"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        self.dataset["metadata"]["total_words"] = len(self.dataset["words"])
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        
        print(f"[爬蟲] 資料已儲存至 {OUTPUT_FILE}")
    
    def run(self):
        """執行爬蟲"""
        print("=" * 50)
        print("Klokah 族語資源爬蟲")
        print("=" * 50)
        
        # 爬取資料
        words = self.crawl_all_dialects(max_dialects=5)
        
        # 去重
        seen = set()
        unique_words = []
        for w in words:
            if w["text"] not in seen:
                seen.add(w["text"])
                unique_words.append(w)
        
        self.dataset["words"] = unique_words
        
        # 儲存
        self.save_dataset()
        
        print(f"\n[完成] 共爬取 {len(unique_words)} 個不重複單詞")
        print(f"其中有音檔的: {sum(1 for w in unique_words if w['audio_url'])}")


# ============================================
# 主程式
# ============================================

if __name__ == "__main__":
    crawler = KlokahCrawler()
    crawler.run()
