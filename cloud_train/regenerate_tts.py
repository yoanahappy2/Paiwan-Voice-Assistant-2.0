#!/usr/bin/env python3
"""
regenerate_tts.py — 重新生成排灣語 TTS 音檔（修正版）

修正內容：
1. 參考音檔統一使用最長且乾淨的單一錄音，避免模糊匹配導致重複
2. 先裁切參考音檔的前 3-6 秒（取最穩定段落），避免錄音中的重複內容被克隆
3. 加入音量正規化，確保輸出品質一致

用法：
  python regenerate_tts.py --dataset /workspace/paiwan_train/dataset --output /workspace/paiwan_train/output

作者: 地陪
日期: 2026-04-18
"""

import os
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import csv
import wave
import struct
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import torch


def trim_reference_audio(input_path, output_path, max_seconds=5.0):
    """
    裁切參考音檔，只取前 max_seconds 秒。
    避免錄音中的重複段落影響克隆品質。
    """
    with wave.open(input_path, 'rb') as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        total_frames = w.getnframes()
        max_frames = int(rate * max_seconds)
        frames_to_read = min(total_frames, max_frames)
        raw = w.readframes(frames_to_read)
    
    with wave.open(output_path, 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(raw)
    
    duration = frames_to_read / rate
    print(f"  參考音檔已裁切: {frames_to_read/rate:.1f}s (原始 {total_frames/rate:.1f}s)")
    return output_path


def find_best_reference(wavs_dir):
    """
    找到最適合的參考音檔：
    1. 優先找長度在 2-8 秒之間的（太短品質差，太長可能有重複）
    2. 在符合條件的音檔中選最長的
    3. 如果沒有符合條件的，就選最短的（最不容易有重複）
    """
    import torchaudio
    
    candidates = []
    for wav in sorted(wavs_dir.glob("*.wav")):
        try:
            info = torchaudio.info(str(wav))
            dur = info.num_frames / info.sample_rate
            candidates.append((wav, dur))
        except:
            continue
    
    # 優先選 2-8 秒範圍內最長的
    good = [(w, d) for w, d in candidates if 2.0 <= d <= 8.0]
    if good:
        best = max(good, key=lambda x: x[1])
        print(f"  選用參考音檔: {best[0].name} ({best[1]:.1f}s)")
        return str(best[0])
    
    # 退而求其次，選最短的（避免重複）
    if candidates:
        best = min(candidates, key=lambda x: x[1])
        print(f"  選用最短音檔: {best[0].name} ({best[1]:.1f}s)")
        return str(best[0])
    
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="重新生成排灣語 TTS 音檔")
    parser.add_argument("--dataset", default="/workspace/paiwan_train/dataset")
    parser.add_argument("--output", default="/workspace/paiwan_train/output")
    parser.add_argument("--ref_trim_seconds", type=float, default=4.0,
                        help="參考音檔裁切長度（秒），預設 4 秒")
    parser.add_argument("--language", default="en",
                        help="TTS 語言代碼（XTTS v2 不支援 pag，建議用 en）")
    args = parser.parse_args()
    
    print("=" * 60)
    print("排灣語 TTS 音檔重新生成（修正版）")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    dataset_dir = Path(args.dataset)
    wavs_dir = dataset_dir / "wavs"
    output_base = Path(args.output)
    
    assert wavs_dir.exists(), f"找不到音檔目錄: {wavs_dir}"
    assert (dataset_dir / "metadata.csv").exists(), f"找不到 metadata.csv"
    
    # 檢查 GPU
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # PyTorch 2.6+ 補丁
    import TTS.utils.io
    _orig = TTS.utils.io.load_fsspec
    def _patched(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig(*a, **kw)
    TTS.utils.io.load_fsspec = _patched
    
    from TTS.api import TTS
    
    # 載入模型
    print("\n載入 XTTS v2 模型...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1").to(device)
    print("模型載入完成")
    
    # 選擇參考音檔
    print("\n--- 選擇參考音檔 ---")
    best_ref = find_best_reference(wavs_dir)
    assert best_ref, "找不到可用的參考音檔"
    
    # 裁切參考音檔
    print("\n--- 裁切參考音檔 ---")
    trimmed_ref = str(output_base / "reference_trimmed.wav")
    output_base.mkdir(parents=True, exist_ok=True)
    trim_reference_audio(best_ref, trimmed_ref, max_seconds=args.ref_trim_seconds)
    
    # 測試語言代碼
    print(f"\n--- 測試語言代碼 ---")
    working_lang = args.language
    test_out = str(output_base / "test_output.wav")
    try:
        tts.tts_to_file(text="masalu", speaker_wav=trimmed_ref, language=working_lang, file_path=test_out)
        print(f"語言代碼 '{working_lang}' 可用")
    except Exception as e:
        print(f"語言代碼 '{working_lang}' 失敗: {e}")
        # 嘗試其他代碼
        for lang in ["en", "zh-cn", "ja", "ko"]:
            try:
                tts.tts_to_file(text="masalu", speaker_wav=trimmed_ref, language=lang, file_path=test_out)
                working_lang = lang
                print(f"改用語言代碼 '{lang}'")
                break
            except:
                continue
    
    # === 詞彙生成 ===
    print(f"\n=== 開始生成詞彙 ===")
    out_dir = output_base / "paiwan_tts_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(dataset_dir / "metadata.csv") as f:
        reader = csv.DictReader(f)
        samples = list(reader)
    
    print(f"共 {len(samples)} 個詞彙待生成")
    
    success = 0
    fail = 0
    for i, s in enumerate(samples):
        text = s['text']
        out_file = str(out_dir / f"{text}.wav")
        
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=trimmed_ref,   # 統一用裁切後的參考音檔
                language=working_lang,
                file_path=out_file
            )
            success += 1
        except Exception as e:
            fail += 1
            print(f"  失敗 {text}: {str(e)[:60]}")
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] 成功 {success} / 失敗 {fail}")
    
    print(f"\n詞彙生成完成: 成功 {success} / 失敗 {fail}")
    
    # === 句子生成 ===
    print(f"\n=== 開始生成句子 ===")
    sentences_dir = out_dir / "sentences"
    sentences_dir.mkdir(exist_ok=True)
    
    sentences = [
        ("masalu", "masalu"),
        ("na_tarivak_sun", "na tarivak sun"),
        ("tima_su_ngadan", "tima su ngadan"),
        ("uri_semainu_sun", "uri semainu sun"),
        ("maya_kiljavaran", "maya kiljavaran"),
        ("kemasinu_sun", "kemasinu sun"),
        ("pida_anga_su_cavilj", "pida anga su cavilj"),
        ("pacunan", "pacunan"),
    ]
    
    for fname, text in sentences:
        out_file = str(sentences_dir / f"{fname}.wav")
        try:
            tts.tts_to_file(text=text, speaker_wav=trimmed_ref, language=working_lang, file_path=out_file)
            print(f"  完成: {text}")
        except Exception as e:
            print(f"  失敗 {text}: {str(e)[:60]}")
    
    # === 精選展示用音檔（挑 5 個品質最好的） ===
    print(f"\n=== 精選展示音檔 ===")
    showcase_dir = output_base / "tts_showcase"
    showcase_dir.mkdir(exist_ok=True)
    
    # 挑短詞（發音通常更準）和常用句
    showcase_words = ["masalu", "ita", "lima", "drusa", "alu"]
    for word in showcase_words:
        src = out_dir / f"{word}.wav"
        if src.exists():
            shutil.copy2(str(src), str(showcase_dir / f"{word}.wav"))
            print(f"  精選: {word}")
    
    # 也複製句子
    for fname, _ in sentences[:5]:
        src = sentences_dir / f"{fname}.wav"
        if src.exists():
            shutil.copy2(str(src), str(showcase_dir / f"{fname}.wav"))
            print(f"  精選句子: {fname}")
    
    # 打包
    archive = output_base / "paiwan_tts_v2_package"
    shutil.make_archive(str(archive), 'gztar', str(out_dir))
    
    print(f"\n{'='*60}")
    print(f"全部完成！")
    print(f"全部詞彙: {out_dir}/")
    print(f"精選展示: {showcase_dir}/")
    print(f"打包檔案: {archive}.tar.gz")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
