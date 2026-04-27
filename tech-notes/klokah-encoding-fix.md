# 技術筆記：klokah.tw 爬蟲編碼問題

## 問題

從族語E樂園 (klokah.tw) 爬取東排灣語語料時，中文回傳亂碼。

例如 `gacalju` 應翻譯為「起立！敬禮！(班長說)」，但實際得到：
```
èµ·ç«ï¼æ¬ç¦®ï¼(çé·èªª)
```

## 原因

`requests` 庫在收到回應時，會根據 HTTP header 猜測 encoding。klokah.tw 的 API 回應沒有聲明 `charset=utf-8`，所以 requests 預設用 `ISO-8859-1`（Latin-1）解碼。

實際內容是 UTF-8，但被當成 Latin-1 讀取後，中文字元被雙重編碼：
```
UTF-8 bytes → Latin-1 解碼（錯誤）→ 字串中每個 byte 變成一個 Unicode 字元
```

## 解法

**用 `r.content`（bytes）代替 `r.text`（str），直接讓 XML 解析器處理原始 bytes。**

```python
import requests
import xml.etree.ElementTree as ET

r = requests.get('https://web.klokah.tw/php/multiSearchResult.php',
                 params={'d': 23, 'txt': 'masalu'})

# ❌ 錯誤：r.text 已被 requests 用錯誤的 encoding 解碼
# root = ET.fromstring(r.text)

# ✅ 正確：r.content 是原始 bytes，XML parser 自己判斷 encoding
root = ET.fromstring(r.content)
```

## 驗證

```python
# 確認 encoding 問題
print(r.encoding)      # 'ISO-8859-1' ← 錯的
print(r.apparent_encoding)  # 'utf-8' ← 對的

# 手動修復也可以（但用 r.content 更乾淨）
r.encoding = 'utf-8'
text = r.text  # 這時就正確了
```

## 存檔注意

JSON 存檔時必須使用 `ensure_ascii=False` + `encoding='utf-8'`：

```python
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

否則中文會被存成 `\uXXXX` escape 序列。

## 日期
2026-04-27
