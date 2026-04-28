"""
feishu_bot/bitable_writer.py
語聲同行 2.0 — 飛書多維表格寫入服務

功能：將 ASR 轉寫結果自動寫入飛書多維表格
用途：語料收集、語言檔案化展示

表格欄位：
- 多行文本 (primary, 無法改名): ASR 轉寫文字
- 中文翻譯: RAG+LLM 翻譯結果
- 語料來源: 飛書對話/即時語音/教材
- 轉寫時間: 自動填入
- 置信度: 數字
- 是否正確: ✅正確/⚠️需修正/❌錯誤

作者: Backend_Dev
日期: 2026-04-28
"""

import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

APP_ID = os.getenv("FEISHU_APP_ID", "")
APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")

# Bitable 配置
BITABLE_APP_TOKEN = "KfUUbiz5taJhqJsFrDocjGP2nQu"
BITABLE_TABLE_ID = "tbl4WiIy8QIoa8pf"

# Token 快取
_token_cache = {"token": None, "expire_at": 0}


def get_token():
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expire_at"] - 300:
        return _token_cache["token"]
    
    resp = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": APP_ID, "app_secret": APP_SECRET},
        timeout=10,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Token 獲取失敗: {data}")
    
    _token_cache["token"] = data["tenant_access_token"]
    _token_cache["expire_at"] = now + data.get("expire", 7200)
    return _token_cache["token"]


def write_transcription(
    asr_text: str,
    corrected_text: str = "",
    chinese_translation: str = "",
    confidence: float = 0.0,
    source: str = "飛書對話",
) -> dict:
    """
    將一筆轉寫結果寫入飛書多維表格
    """
    token = get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    fields = {
        "多行文本": asr_text,
    }
    
    if chinese_translation:
        fields["中文翻譯"] = chinese_translation
    if source:
        fields["語料來源"] = source
    if confidence > 0:
        fields["置信度"] = round(confidence * 100, 1)
    
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables/{BITABLE_TABLE_ID}/records"
    
    resp = requests.post(url, headers=headers, json={"fields": fields}, timeout=15)
    return resp.json()


def batch_write(records: list) -> dict:
    """批次寫入"""
    token = get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    payload = []
    for r in records:
        fields = {"多行文本": r.get("asr_text", "")}
        if r.get("chinese_translation"):
            fields["中文翻譯"] = r["chinese_translation"]
        if r.get("source"):
            fields["語料來源"] = r["source"]
        payload.append({"fields": fields})
    
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables/{BITABLE_TABLE_ID}/records/batch_create"
    resp = requests.post(url, headers=headers, json={"records": payload}, timeout=30)
    return resp.json()


if __name__ == "__main__":
    result = write_transcription(
        asr_text="masalu",
        chinese_translation="謝謝",
        confidence=0.95,
        source="飛書對話",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
