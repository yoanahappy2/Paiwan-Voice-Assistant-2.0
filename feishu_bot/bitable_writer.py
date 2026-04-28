"""
feishu_bot/bitable_writer.py
語聲同行 2.0 — 飛書多維表格寫入服務

功能：將 ASR 轉寫結果自動寫入飛書多維表格
用途：語料收集、語言檔案化展示

前置條件：
- 飛書後台開通 bitable:app 權限
- 將 Bot 加為多維表格的協作者

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
BITABLE_TABLE_ID = "tbldwDfAJ3fxxhHt"

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
    source: str = "即時語音",
) -> dict:
    """
    將一筆轉寫結果寫入飛書多維表格
    
    Args:
        asr_text: ASR 原始轉寫文字
        corrected_text: 音韻校正後文字
        chinese_translation: 中文翻譯
        confidence: ASR 置信度 (0-1)
        source: 語料來源 (即時語音/教材/測驗)
    
    Returns:
        API 回應
    """
    token = get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    now_ms = int(time.time() * 1000)
    
    fields = {
        "文本": asr_text,  # 預設欄位名（rename 需 bitable:app:editable 權限）
        "单选": source,
    }
    
    if corrected_text:
        fields["文本"] = f"{asr_text} → {corrected_text}"
    if chinese_translation:
        fields["文本"] = f"{fields.get('文本', asr_text)} | 翻譯: {chinese_translation}"
    if confidence > 0:
        pass  # 置信度欄位待建立
    
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables/{BITABLE_TABLE_ID}/records"
    
    resp = requests.post(
        url,
        headers=headers,
        json={"fields": fields},
        timeout=15,
    )
    
    return resp.json()


def batch_write_transcriptions(records: list) -> dict:
    """
    批次寫入多筆轉寫結果
    
    Args:
        records: list of dicts with asr_text, corrected_text, etc.
    
    Returns:
        API 回應
    """
    token = get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    now_ms = int(time.time() * 1000)
    
    payload_records = []
    for r in records:
        fields = {
            "ASR 轉寫文字": r.get("asr_text", ""),
            "轉寫時間": now_ms,
            "語料來源": r.get("source", "即時語音"),
        }
        if r.get("corrected_text"):
            fields["音韻校正後"] = r["corrected_text"]
        if r.get("chinese_translation"):
            fields["中文翻譯"] = r["chinese_translation"]
        if r.get("confidence", 0) > 0:
            fields["置信度"] = round(r["confidence"] * 100, 1)
        payload_records.append({"fields": fields})
    
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables/{BITABLE_TABLE_ID}/records/batch_create"
    
    resp = requests.post(
        url,
        headers=headers,
        json={"records": payload_records},
        timeout=30,
    )
    
    return resp.json()


if __name__ == "__main__":
    # 測試寫入
    result = write_transcription(
        asr_text="masalu",
        corrected_text="masalu",
        chinese_translation="謝謝",
        confidence=0.95,
        source="即時語音",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
