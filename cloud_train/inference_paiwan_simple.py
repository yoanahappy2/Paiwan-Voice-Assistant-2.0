#!/usr/bin/env python3
"""
排灣語語音生成推理腳本 (簡化版)
使用 TTS API + 參考音訊生成排灣語語音
"""

import os
import sys

os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

def generate_speech(
    text: str,
    reference_audio: str,
    language: str = "pag",  # 排灣語 ISO 639-3 code
    output_file: str = "output_paiwan.wav"
):
    """
    使用 XTTS v2 生成排灣語語音

    Args:
        text: 要生成的排灣語文字
        reference_audio: 參考音訊檔案路徑（長老的聲音）
        language: 語言代碼
        output_file: 輸出檔案名稱
    """
    print(f"\n{'='*60}")
    print(f"排灣語語音生成 (XTTS v2)")
    print(f"{'='*60}")
    print(f"文字: {text}")
    print(f"參考音訊: {reference_audio}")
    print(f"語言: {language}")
    print(f"{'='*60}\n")

    # 檢查參考音訊
    if not os.path.exists(reference_audio):
        raise FileNotFoundError(f"找不到參考音訊: {reference_audio}")

    # 載入模型（使用 XTTS v2）
    print("📥 載入 XTTS v2 模型...")
    device = "cuda" if __import__("torch").cuda.is_available() else \
             "mps" if __import__("torch").backends.mps.is_available() else "cpu"

    # 使用 TTS API 載入模型
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1").to(device)
    print(f"✓ 模型載入完成 (設備: {device})")

    # 生成語音
    print("🔊 生成語音中...")
    tts.tts_to_file(
        text=text,
        speaker_wav=reference_audio,
        language=language,
        file_path=output_file
    )

    print(f"✓ 語音已保存: {output_file}")
    return output_file

def main():
    """主函數"""
    # 檢查命令列參數
    if len(sys.argv) < 3:
        print(__doc__)
        print("\n使用方法:")
        print("  python inference_paiwan_simple.py <文字> <參考音訊> [輸出檔案]")
        print("\n範例:")
        print('  python inference_paiwan_simple.py "kako a pacucuacay" dataset/wavs/s03-alu.wav')
        print('  python inference_paiwan_simple.py "kako a pacucuacay" dataset/wavs/s03-alu.wav output.wav')
        print("\n排灣語詞彙範例:")
        print("  alu, cemdas, ceqer, drusa, qadaw, qemudjalj, vuyu")
        sys.exit(1)

    text = sys.argv[1]
    reference_audio = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "output_paiwan.wav"

    try:
        generate_speech(
            text=text,
            reference_audio=reference_audio,
            output_file=output_file
        )

        print(f"\n{'='*60}")
        print("✓ 生成完成！")
        print(f"📁 輸出檔案: {output_file}")
        print(f"🎵 可用任何音訊播放器播放")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
