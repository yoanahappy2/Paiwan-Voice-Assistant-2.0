"""
tts_service.py
語聲同行 2.0 — TTS 語音合成服務

方案：
1. 排灣語預合成音檔（來自 XTTS v2 微調）— 排灣語詞彙和句子
2. macOS say（Grandma 語音）— 中文部分，本地離線
3. edge-tts（微軟線上）— 備選，需網路

輸出格式：WAV（Gradio 相容）

作者: Backend_Dev
日期: 2026-04-16（更新 TTS 整合）
"""

import os
import subprocess
import tempfile
import platform
from pathlib import Path

# ============================================
# 配置
# ============================================

# 預合成排灣語音檔目錄
PAIWAN_TTS_DIR = Path(__file__).parent / "paiwan_tts"
PAIWAN_SENTENCES_DIR = PAIWAN_TTS_DIR / "sentences"

# macOS 語音選擇
MACOS_VOICE_TW = "Grandma (中文（台灣）)"
MACOS_VOICE_CN = "Grandma (中文（中國大陸）)"

# edge-tts 語音
EDGE_VOICE = "zh-TW-HsiaoChenNeural"

# 輸出目錄
TTS_OUTPUT_DIR = Path(tempfile.gettempdir()) / "paiwan_tts"
TTS_OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================
# 排灣語預合成音檔
# ============================================

def get_paiwan_audio(word: str) -> str | None:
    """
    取得排灣語詞彙的預合成音檔
    
    Args:
        word: 排灣語詞彙（如 "masalu"）
    
    Returns:
        音檔路徑或 None
    """
    if not word or not word.strip():
        return None
    
    word = word.strip().lower()
    
    # 先找句子
    sentence_path = PAIWAN_SENTENCES_DIR / f"{word}.wav"
    if sentence_path.exists():
        return str(sentence_path)
    
    # 再找詞彙
    word_path = PAIWAN_TTS_DIR / f"{word}.wav"
    if word_path.exists():
        return str(word_path)
    
    # 嘗試將空格替換為底線（句子格式）
    word_underscore = word.replace(" ", "_")
    sentence_path2 = PAIWAN_SENTENCES_DIR / f"{word_underscore}.wav"
    if sentence_path2.exists():
        return str(sentence_path2)
    
    return None


def get_paiwan_audio_for_text(text: str) -> str | None:
    """
    從一段文字中找出可以播放的排灣語音檔
    嘗試整句匹配，再嘗試逐詞匹配
    
    Args:
        text: 排灣語文字
    
    Returns:
        音檔路徑或 None
    """
    if not text:
        return None
    
    text = text.strip()
    
    # 1. 整句匹配（空格→底線）
    audio = get_paiwan_audio(text)
    if audio:
        return audio
    
    # 2. 去掉標點符號再試
    import re
    clean = re.sub(r'[。，！？、；：""''（）\[\]{}]', '', text)
    audio = get_paiwan_audio(clean)
    if audio:
        return audio
    
    # 3. 嘗試逐詞找第一個有音檔的詞
    words = clean.split()
    for w in words:
        audio = get_paiwan_audio(w)
        if audio:
            return audio
    
    return None


def list_available_paiwan_audio() -> dict:
    """列出所有可用的排灣語音檔"""
    result = {"words": [], "sentences": []}
    
    if PAIWAN_TTS_DIR.exists():
        for wav in sorted(PAIWAN_TTS_DIR.glob("*.wav")):
            result["words"].append(wav.stem)
    
    if PAIWAN_SENTENCES_DIR.exists():
        for wav in sorted(PAIWAN_SENTENCES_DIR.glob("*.wav")):
            result["sentences"].append(wav.stem)
    
    return result


# ============================================
# macOS say TTS（中文部分）
# ============================================

def tts_macos(text: str, output_path: str = None, voice: str = None) -> str | None:
    """macOS say 指令 TTS（中文）"""
    if platform.system() != "Darwin":
        raise RuntimeError("macOS say 只能在 macOS 上使用")

    if not text.strip():
        return None

    if output_path is None:
        import time
        output_path = str(TTS_OUTPUT_DIR / f"tts_{int(time.time() * 1000)}.wav")

    voice = voice or MACOS_VOICE_TW
    aiff_path = output_path.replace(".wav", ".aiff")

    try:
        subprocess.run(
            ["say", "-v", voice, "-o", aiff_path, text],
            capture_output=True, text=True, timeout=10
        )
        if not os.path.exists(aiff_path):
            return None

        subprocess.run(
            ["afconvert", aiff_path, output_path, "-f", "WAVE", "-d", "LEI16@22050"],
            capture_output=True, text=True, timeout=10
        )

        if os.path.exists(aiff_path) and aiff_path != output_path:
            os.remove(aiff_path)

        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        print(f"TTS (macOS) error: {e}")
        return None


# ============================================
# edge-tts（備選）
# ============================================

async def tts_edge_async(text: str, output_path: str = None) -> str | None:
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


def tts_edge(text: str, output_path: str = None) -> str | None:
    """edge-tts 同步包裝"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
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

def tts_gtts(text: str, output_path: str = None) -> str | None:
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

def synthesize_paiwan(text: str) -> str | None:
    """
    合成排灣語語音（優先用預合成音檔）
    
    Returns:
        音檔路徑或 None
    """
    return get_paiwan_audio_for_text(text)


def synthesize_chinese(text: str, output_path: str = None, engine: str = "auto") -> str | None:
    """
    合成中文語音
    
    Args:
        text: 中文文字
        engine: "auto" | "macos" | "edge" | "gtts"
    """
    import re
    chinese_chars = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u2000-\u206f0-9！？。，]+', text)
    chinese_text = ''.join(chinese_chars)
    if not chinese_text:
        chinese_text = text
    
    if engine == "auto":
        if platform.system() == "Darwin":
            result = tts_macos(chinese_text, output_path)
            if result:
                return result
        result = tts_edge(chinese_text, output_path)
        if result:
            return result
        return tts_gtts(chinese_text, output_path)
    elif engine == "macos":
        return tts_macos(chinese_text, output_path)
    elif engine == "edge":
        return tts_edge(chinese_text, output_path)
    elif engine == "gtts":
        return tts_gtts(chinese_text, output_path)
    else:
        raise ValueError(f"Unknown TTS engine: {engine}")


def synthesize(text: str, output_path: str = None, engine: str = "auto") -> str | None:
    """
    統一 TTS 介面（向後兼容）
    
    優先嘗試排灣語預合成，再合成中文部分
    """
    # 先試排灣語
    paiwan_audio = get_paiwan_audio_for_text(text)
    if paiwan_audio:
        return paiwan_audio
    
    # 再合成中文
    return synthesize_chinese(text, output_path, engine)


# ============================================
# 測試
# ============================================

def main():
    print("=" * 50)
    print("  TTS 測試")
    print("=" * 50)
    
    # 測試排灣語預合成
    print("\n--- 排灣語預合成 ---")
    available = list_available_paiwan_audio()
    print(f"詞彙: {len(available['words'])} 個")
    print(f"句子: {len(available['sentences'])} 個")
    
    test_words = ["masalu", "ita", "drusa", "lima", "na tarivak sun", "tima su ngadan"]
    for w in test_words:
        audio = get_paiwan_audio_for_text(w)
        status = f"✅ {audio}" if audio else "❌ 未找到"
        print(f"  {w}: {status}")
    
    # 測試中文合成
    print("\n--- 中文合成 ---")
    engine = "macos" if platform.system() == "Darwin" else "edge"
    test_texts = [
        "孫子啊，來跟 vuvu 說說話吧！",
        "非常感謝你的愛。",
        "我也很好，謝謝你。",
    ]
    for text in test_texts:
        print(f"\n合成: {text}")
        result = synthesize_chinese(text, engine=engine)
        if result:
            print(f"✅ 輸出: {result} ({os.path.getsize(result)} bytes)")
        else:
            print("❌ 合成失敗")


if __name__ == "__main__":
    main()
