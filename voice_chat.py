"""
voice_chat.py
語聲同行 競賽版 — Voice-to-Voice 全鏈路串接

流程: 錄音 → ASR (Whisper LoRA) → LLM (智譜 GLM-4-Flash) → 回覆

作者: Backend_Dev (OpenClaw Agent)
日期: 2026-04-15
用途: 飛書 AI 校園挑戰賽 — Phase 2 語音對話
"""

import os
import sys
import tempfile
import requests
from llm_service import VuvuService

# ============================================
# 配置
# ============================================

# 原專案 FastAPI 後端（本地運行中）
ASR_API_URL = "http://127.0.0.1:8000/api/audio/recognize"


# ============================================
# ASR 客戶端
# ============================================

def recognize_audio(audio_path: str) -> dict:
    """
    呼叫原專案的 ASR API 進行語音辨識

    Args:
        audio_path: 音檔路徑（支援 .m4a, .wav, .webm）

    Returns:
        {"text": "辨識文字", "confidence": 0.xx, ...}
    """
    if not os.path.exists(audio_path):
        return {"text": "", "confidence": 0.0, "error": f"檔案不存在: {audio_path}"}

    filename = os.path.basename(audio_path)
    # 根據副檔名決定 content type
    ext = os.path.splitext(filename)[1].lower()
    content_types = {
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
    }
    content_type = content_types.get(ext, "audio/wav")

    try:
        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, content_type)}
            response = requests.post(ASR_API_URL, files=files, timeout=30)

        if response.status_code == 200:
            result = response.json()
            return {
                "text": result.get("corrected", ""),
                "raw_text": result.get("debugInfo", {}).get("raw_text", ""),
                "confidence": result.get("confidence", 0.0),
                "logic": result.get("logic", ""),
            }
        else:
            return {"text": "", "confidence": 0.0, "error": f"API 錯誤: {response.status_code}"}

    except requests.exceptions.ConnectionError:
        return {"text": "", "confidence": 0.0, "error": "無法連接 ASR 服務，請確認 FastAPI 是否在 http://127.0.0.1:8000 運行"}
    except Exception as e:
        return {"text": "", "confidence": 0.0, "error": str(e)}


# ============================================
# 語音對話主流程
# ============================================

def voice_chat(audio_path: str, vuvu: VuvuService = None) -> dict:
    """
    完整語音對話流程: 音檔 → ASR → LLM → 回覆

    Args:
        audio_path: 用戶錄音檔路徑
        vuvu: VuvuService 實例（可選，不傳則新建）

    Returns:
        {
            "recognized_text": "ASR 辨識結果",
            "confidence": 0.xx,
            "vuvu_reply": "vuvu 的回覆",
            "asr_logic": "校正邏輯說明",
        }
    """
    if vuvu is None:
        vuvu = VuvuService()

    # Step 1: ASR 語音辨識
    print("🎤 辨識語音中...")
    asr_result = recognize_audio(audio_path)

    if asr_result.get("error"):
        return {
            "recognized_text": "",
            "confidence": 0.0,
            "vuvu_reply": f"[ASR 錯誤] {asr_result['error']}",
            "asr_logic": "",
        }

    recognized = asr_result.get("text", "")
    confidence = asr_result.get("confidence", 0.0)

    if not recognized:
        return {
            "recognized_text": "",
            "confidence": 0.0,
            "vuvu_reply": "[ASR] 辨識結果為空，請再試一次",
            "asr_logic": asr_result.get("logic", ""),
        }

    # Step 2: LLM 對話生成
    print(f"📝 辨識結果: {recognized} (信心度: {confidence:.2f})")
    print("👵 vuvu 思考中...")
    result = vuvu.chat_with_thinking(recognized)

    return {
        "recognized_text": recognized,
        "raw_text": asr_result.get("raw_text", ""),
        "confidence": confidence,
        "vuvu_reply": result["reply"],
        "vuvu_thinking": result["thinking"],
        "asr_logic": asr_result.get("logic", ""),
    }


# ============================================
# 終端機測試介面
# ============================================

def main():
    """終端機語音對話測試"""
    print("=" * 50)
    print("  語聲同行 競賽版 — Voice-to-Voice 測試")
    print("=" * 50)
    print()
    print("模式:")
    print("  1. 輸入音檔路徑（直接測試 .m4a / .wav）")
    print("  2. 輸入文字（跳過 ASR，直接跟 vuvu 對話）")
    print("  /quit — 離開")
    print("  /reset — 重置對話")
    print()

    # 檢查 ASR 服務是否可用
    try:
        requests.get("http://127.0.0.1:8000/health", timeout=3)
        print("✅ ASR 服務已連接")
    except:
        print("⚠️  ASR 服務未連接，只能使用文字模式")
        print("   啟動方式: cd paiwan_asr_project && python -m uvicorn api.server:app --reload")

    print("-" * 50)

    vuvu = VuvuService()

    while True:
        try:
            user_input = input("\n🧑 輸入音檔路徑或文字: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Pacunan!")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print("👋 Pacunan!")
            break
        if user_input == "/reset":
            vuvu.reset()
            print("✅ 對話已重置")
            continue

        # 判斷是音檔路徑還是文字
        if os.path.isfile(user_input):
            # 音檔模式
            result = voice_chat(user_input, vuvu)
            print(f"\n🎤 ASR: {result['recognized_text']}")
            if result.get('raw_text') and result['raw_text'] != result['recognized_text']:
                print(f"   原始: {result['raw_text']}")
            if result.get('asr_logic'):
                print(f"   校正: {result['asr_logic']}")
            print(f"   信心度: {result['confidence']:.2f}")
            print(f"\n👵 vuvu: {result['vuvu_reply']}")
        else:
            # 文字模式
            print("\n👵 vuvu: ", end="", flush=True)
            reply = vuvu.chat(user_input)
            print(reply)


if __name__ == "__main__":
    main()
