"""
feishu_bot/media.py
語聲同行 2.0 — 飛書媒體檔案下載

用於下載用戶發送的語音訊息（ogg）並轉為 WAV 格式供 ASR 使用。
"""

import os
import requests
from pathlib import Path
from auth import get_tenant_access_token, FEISHU_API_BASE

# 臨時檔案目錄
TEMP_DIR = Path(__file__).parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)


def download_audio(message_id: str, file_key: str) -> str | None:
    """
    下載飛書語音訊息
    
    Args:
        message_id: 訊息 ID
        file_key: 檔案 key
        
    Returns:
        str: 下載後的 ogg 檔案路徑，失敗返回 None
    """
    token = get_tenant_access_token()
    url = f"{FEISHU_API_BASE}/im/v1/messages/{message_id}/resources/{file_key}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"type": "file"}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"[media] 下載失敗: HTTP {resp.status_code}")
            return None
        
        out_path = TEMP_DIR / f"{message_id}.ogg"
        with open(out_path, "wb") as f:
            f.write(resp.content)
        
        print(f"[media] 已下載: {out_path} ({len(resp.content)} bytes)")
        return str(out_path)
    except Exception as e:
        print(f"[media] 下載異常: {e}")
        return None


def ogg_to_wav(ogg_path: str) -> str | None:
    """
    將 ogg 轉為 wav 格式（供 ASR 使用）
    
    Args:
        ogg_path: ogg 檔案路徑
        
    Returns:
        str: wav 檔案路徑，失敗返回 None
    """
    wav_path = ogg_path.rsplit(".", 1)[0] + ".wav"
    
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"[media] ffmpeg 轉檔失敗: {result.stderr[:200]}")
            return None
        
        print(f"[media] 已轉檔: {wav_path}")
        return wav_path
    except FileNotFoundError:
        print("[media] ffmpeg 未安裝，嘗試直接使用 ogg")
        return ogg_path
    except Exception as e:
        print(f"[media] 轉檔異常: {e}")
        return ogg_path


if __name__ == "__main__":
    print("media.py — 語音下載工具（被 bot.py 調用）")
