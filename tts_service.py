"""
tts_service.py
語聲同行 2.0 — TTS 語音合成服務

方案：
1. macOS say（Grandma 語音）— 本地離線，音質好，角色匹配
2. edge-tts（微軟線上）— 備選，需網路
3. gTTS（Google 線上）— 備選，需網路

輸出格式：WAV（Gradio 相容）

作者: Backend_Dev
日期: 2026-04-15
"""

import os
import subprocess
import tempfile
import platform
from pathlib import Path

# ============================================
# 配置
# ============================================

# macOS 語音選擇
MACOS_VOICE_TW = "Grandma (中文（台灣）)"  # 祖母語音，匹配 vuvu 角色
MACOS_VOICE_CN = "Grandma (中文（中國大陸）)"

# edge-tts 語音
EDGE_VOICE = "zh-TW-HsiaoChenNeural"  # 台灣女聲

# 輸出目錄
TTS_OUTPUT_DIR = Path(tempfile.gettempdir()) / "paiwan_tts"
TTS_OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================
# macOS say TTS（首選）
# ============================================

def tts_macos(text: str, output_path: str = None, voice: str = None) -> str:
    """
    macOS say 指令 TTS
    
    Args:
        text: 要合成的文字（中文部分）
        output_path: 輸出路徑（None 則自動生成）
        voice: 語音名稱（None 則用預設）
    
    Returns:
        輸出音檔路徑（WAV 格式）
    """
    if platform.system() != "Darwin":
        raise RuntimeError("macOS say 只能在 macOS 上使用")

    if not text.strip():
        return None

    # 輸出路徑
    if output_path is None:
        import time
        output_path = str(TTS_OUTPUT_DIR / f"tts_{int(time.time() * 1000)}.wav")

    voice = voice or MACOS_VOICE_TW

    # macOS say 輸出 AIFF，再用 afconvert 轉 WAV
    aiff_path = output_path.replace(".wav", ".aiff")

    try:
        # Step 1: say → AIFF
        subprocess.run(
            ["say", "-v", voice, "-o", aiff_path, text],
            capture_output=True, text=True, timeout=10
        )

        if not os.path.exists(aiff_path):
            return None

        # Step 2: AIFF → WAV（Gradio 需要 WAV/MP3）
        subprocess.run(
            ["afconvert", aiff_path, output_path, "-f", "WAVE", "-d", "LEI16@22050"],
            capture_output=True, text=True, timeout=10
        )

        # 清理 AIFF
        if os.path.exists(aiff_path) and aiff_path != output_path:
            os.remove(aiff_path)

        return output_path if os.path.exists(output_path) else None

    except Exception as e:
        print(f"TTS (macOS) error: {e}")
        return None


# ============================================
# edge-tts（備選）
# ============================================

async def tts_edge_async(text: str, output_path: str = None) -> str:
    """edge-tts 異步版本"""
    import edge_tts

    if not text.strip():
        return None

    if output_path is None:
        import time
        output_path = str(TTS_OUTPUT_DIR / f"tts_edge_{int(time.time() * 1000)}.mp3")

    try:
        communicate = edge_tts.Communicate(text, EDGE_VOICE)
        await communicate.save(output_path)
        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        print(f"TTS (edge) error: {e}")
        return None


def tts_edge(text: str, output_path: str = None) -> str:
    """edge-tts 同步包裝"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # 在已有 event loop 中（如 Gradio），用 nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        return loop.run_until_complete(tts_edge_async(text, output_path))
    else:
        return loop.run_until_complete(tts_edge_async(text, output_path))


# ============================================
# gTTS（備選）
# ============================================

def tts_gtts(text: str, output_path: str = None) -> str:
    """Google TTS"""
    from gtts import gTTS

    if not text.strip():
        return None

    if output_path is None:
        import time
        output_path = str(TTS_OUTPUT_DIR / f"tts_gtts_{int(time.time() * 1000)}.mp3")

    try:
        tts = gTTS(text=text, lang='zh-TW')
        tts.save(output_path)
        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        print(f"TTS (gTTS) error: {e}")
        return None


# ============================================
# 統一介面
# ============================================

def synthesize(text: str, output_path: str = None, engine: str = "auto") -> str:
    """
    統一 TTS 介面

    Args:
        text: 要合成的文字
        output_path: 輸出路徑（None 自動）
        engine: "auto" | "macos" | "edge" | "gtts"

    Returns:
        音檔路徑或 None
    """
    # 提取中文部分（TTS 只合成中文，排灣語部分無法 TTS）
    chinese_text = _extract_chinese(text)
    if not chinese_text:
        chinese_text = text  # fallback

    if engine == "auto":
        if platform.system() == "Darwin":
            result = tts_macos(chinese_text, output_path)
            if result:
                return result
        # macOS 失敗，嘗試 edge-tts
        result = tts_edge(chinese_text, output_path)
        if result:
            return result
        # edge 失敗，嘗試 gTTS
        return tts_gtts(chinese_text, output_path)

    elif engine == "macos":
        return tts_macos(chinese_text, output_path)
    elif engine == "edge":
        return tts_edge(chinese_text, output_path)
    elif engine == "gtts":
        return tts_gtts(chinese_text, output_path)
    else:
        raise ValueError(f"Unknown TTS engine: {engine}")


def _extract_chinese(text: str) -> str:
    """提取文字中的中文部分"""
    import re
    # 移除排灣語（主要是拉丁字母組成的詞）
    # 保留中文字、標點、數字
    chinese_chars = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u2000-\u206f0-9！？。，]+', text)
    return ''.join(chinese_chars)


# ============================================
# 測試
# ============================================

def main():
    print("=" * 50)
    print("  TTS 測試")
    print("=" * 50)

    test_texts = [
        "ai~~ 孫子啊，來跟 vuvu 說說話吧！",
        "masalu aravac！非常感謝你的愛。",
        "na tarivak aken！我也很好，謝謝你。",
        "vuvu 不太確定呢，你能再說一次嗎？",
    ]

    engine = "macos" if platform.system() == "Darwin" else "edge"
    print(f"使用引擎: {engine}")

    for text in test_texts:
        print(f"\n合成: {text}")
        result = synthesize(text, engine=engine)
        if result:
            print(f"✅ 輸出: {result} ({os.path.getsize(result)} bytes)")
        else:
            print("❌ 合成失敗")

    print(f"\n{'=' * 50}")


if __name__ == "__main__":
    main()
