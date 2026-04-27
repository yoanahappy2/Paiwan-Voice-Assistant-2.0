"""
feishu_bot/auth.py
語聲同行 2.0 — 飛書 Token 管理

負責 tenant_access_token 的獲取與自動刷新。
Token 有效期 2 小時，過期自動重新獲取。
"""

import os
import time
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_API_BASE = "https://open.feishu.cn/open-apis"

# Token 快取
_token_cache = {"token": None, "expire_at": 0}


def get_tenant_access_token(force_refresh: bool = False) -> str:
    """
    取得 tenant_access_token（自動刷新）
    
    Returns:
        str: 有效的 tenant_access_token
    """
    now = time.time()
    
    # Token 還有效（提前 5 分鐘刷新）
    if not force_refresh and _token_cache["token"] and now < _token_cache["expire_at"] - 300:
        return _token_cache["token"]
    
    # 請求新 token
    url = f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal"
    payload = {
        "app_id": FEISHU_APP_ID,
        "app_secret": FEISHU_APP_SECRET,
    }
    
    resp = requests.post(url, json=payload, timeout=10)
    data = resp.json()
    
    if data.get("code") != 0:
        raise RuntimeError(f"飛書 token 獲取失敗: {data.get('msg', 'unknown error')}")
    
    token = data["tenant_access_token"]
    expire = data.get("expire", 7200)
    
    _token_cache["token"] = token
    _token_cache["expire_at"] = now + expire
    
    print(f"[auth] Token 刷新成功，有效期 {expire}s")
    return token


def feishu_api_headers() -> dict:
    """返回帶 Authorization 的請求頭"""
    token = get_tenant_access_token()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }


if __name__ == "__main__":
    token = get_tenant_access_token()
    print(f"Token: {token[:20]}...{token[-5:]}")
    print("✅ 飛書認證成功！")
