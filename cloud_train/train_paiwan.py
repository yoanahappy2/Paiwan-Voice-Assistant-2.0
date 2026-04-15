#!/usr/bin/env python3
"""
train_paiwan.py — 排灣語 XTTS v2 微調訓練腳本（穩定版）

使用社群驗證的 XTTSv2-Finetuning-for-New-Languages 方案
如果克隆失敗，自動 fallback 到零樣本克隆

策略：
1. 嘗試克隆 XTTSv2-Finetuning repo 並用它訓練
2. 如果訓練失敗，直接用零樣本克隆生成所有詞彙
3. 最終都會產出可用的排灣語 TTS 音檔

作者: Backend_Dev / 地陪
日期: 2026-04-15
"""

import os
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import csv
import time
import torch
import subprocess
from pathlib import Path
from datetime import datetime


def check_gpu():
    """檢查 GPU 狀態"""
    print("=" * 60)
    print("🔍 環境檢查")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_mem / 1024**3:.1f} GB")
    return torch.cuda.is_available()


def prepare_dataset(dataset_dir):
    """驗證數據集"""
    dataset_dir = Path(dataset_dir)
    wavs_dir = dataset_dir / "wavs"
    metadata_path = dataset_dir / "metadata.csv"
    
    assert metadata_path.exists(), f"找不到 metadata: {metadata_path}"
    assert wavs_dir.exists(), f"找不到 wavs: {wavs_dir}"
    
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        samples = list(reader)
    
    wavs = list(wavs_dir.glob("*.wav"))
    print(f"\n📊 數據集: {len(samples)} 筆標註, {len(wavs)} 個音檔")
    
    # 驗證每筆都有對應音檔
    missing = 0
    for s in samples:
        if not (wavs_dir / s['audio_file']).exists():
            missing += 1
    if missing:
        print(f"⚠️ {missing} 筆缺少音檔")
    else:
        print("✓ 所有音檔完整")
    
    return samples


def method1_finetune_lib(dataset_dir, output_dir, num_epochs=50):
    """
    方法 1: 使用 XTTSv2-Finetuning-for-New-Languages
    社群最穩的 XTTS v2 微調方案
    """
    output_dir = Path(output_dir)
    repo_dir = output_dir / "XTTSv2-Finetuning"
    
    print("\n" + "=" * 60)
    print("📦 方法 1: XTTSv2-Finetuning-for-New-Languages")
    print("=" * 60)
    
    # 克隆 repo
    if not repo_dir.exists():
        print("📥 克隆 XTTSv2-Finetuning...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages.git", str(repo_dir)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"❌ 克隆失敗: {result.stderr}")
            return False
        print("✓ 克隆成功")
    
    # 準備訓練數據格式
    # 這個 repo 需要 LJSpeech 格式: audio_name|text|normalized_text
    dataset_dir = Path(dataset_dir)
    wavs_dir = dataset_dir / "wavs"
    
    ljspeech_path = dataset_dir / "metadata_ljspeech.txt"
    if not ljspeech_path.exists() or ljspeech_path.stat().st_size == 0:
        print("📝 生成 LJSpeech 格式...")
        import csv as csv_mod
        with open(dataset_dir / "metadata.csv") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        with open(ljspeech_path, 'w') as f:
            for r in rows:
                name = Path(r['audio_file']).stem  # s03-alu
                text = r['text']
                f.write(f"{name}|{text}|{text}\n")
        print(f"  ✓ {len(rows)} 筆")
    
    # 查看 repo 的 README 找正確的訓練命令
    # 這個 repo 通常用 run_training.py 或 train.py
    possible_scripts = [
        repo_dir / "xtts_finetune.py",
        repo_dir / "train.py",
        repo_dir / "run_training.py",
        repo_dir / "train_gpt_xtts.py",
    ]
    
    train_script = None
    for s in possible_scripts:
        if s.exists():
            train_script = s
            break
    
    if train_script is None:
        # 列出 repo 內的 Python 檔案
        py_files = list(repo_dir.glob("*.py"))
        print(f"  repo 內的 .py 檔案: {[f.name for f in py_files]}")
        print("⚠️ 找不到訓練腳本，fallback 到方法 2")
        return False
    
    print(f"🎯 訓練腳本: {train_script.name}")
    
    # 執行訓練（直接用 os.system 以便看到即時輸出）
    cmd = f"""cd {repo_dir} && python {train_script.name} \
        --dataset_path {dataset_dir} \
        --output_path {output_dir / 'checkpoints'} \
        --num_epochs {num_epochs} \
        --language pag"""
    
    print(f"\n🚀 執行訓練...")
    print(f"  {cmd}")
    print()
    
    ret = os.system(cmd)
    if ret == 0:
        print("\n✅ 訓練完成！")
        return True
    else:
        print(f"\n⚠️ 訓練返回碼: {ret}")
        return False


def method2_zero_shot(dataset_dir, output_dir):
    """
    方法 2: 零樣本語音克隆（最保險）
    用長老錄音做參考音色，直接合成排灣語
    不需要訓練，XTTS v2 原生支持
    """
    # PyTorch 2.6+ 補丁
    import TTS.utils.io
    _orig = TTS.utils.io.load_fsspec
    def _patched(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig(*a, **kw)
    TTS.utils.io.load_fsspec = _patched
    
    from TTS.api import TTS
    
    print("\n" + "=" * 60)
    print("🎤 方法 2: 零樣本語音克隆")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  設備: {device}")
    
    dataset_dir = Path(dataset_dir)
    wavs_dir = dataset_dir / "wavs"
    
    # 載入 XTTS v2
    print("📥 載入 XTTS v2...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1").to(device)
    print("✓ 模型就緒")
    
    # 選最長的音檔做參考（質量通常最好）
    import torchaudio
    best_ref = None
    best_dur = 0
    for wav in sorted(wavs_dir.glob("*.wav")):
        try:
            info = torchaudio.info(str(wav))
            dur = info.num_frames / info.sample_rate
            if dur > best_dur:
                best_dur = dur
                best_ref = str(wav)
        except:
            continue
    
    print(f"  參考音檔: {Path(best_ref).name} ({best_dur:.1f}s)")
    
    # 測試不同語言 code
    test_text = "masalu"
    test_langs = ["pag", "en", "zh-cn", "ja"]
    working_lang = None
    
    for lang in test_langs:
        try:
            test_out = str(Path(output_dir) / f"test_lang_{lang}.wav")
            tts.tts_to_file(text=test_text, speaker_wav=best_ref, language=lang, file_path=test_out)
            working_lang = lang
            print(f"  ✓ 語言 code '{lang}' 可用")
            break
        except Exception as e:
            print(f"  ✗ 語言 code '{lang}': {str(e)[:60]}")
    
    if not working_lang:
        print("❌ 所有語言 code 都失敗")
        return None
    
    print(f"\n  使用語言 code: {working_lang}")
    
    # 合成所有詞彙
    out_dir = Path(output_dir) / "paiwan_tts_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(dataset_dir / "metadata.csv") as f:
        reader = csv.DictReader(f)
        samples = list(reader)
    
    success = 0
    for i, s in enumerate(samples):
        text = s['text']
        out_file = str(out_dir / f"{text}.wav")
        
        # 嘗試用同詞的錄音做參考（效果更好）
        word_ref = None
        for wav in wavs_dir.glob("*.wav"):
            if text in wav.name:
                word_ref = str(wav)
                break
        if not word_ref:
            word_ref = best_ref
        
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=word_ref,
                language=working_lang,
                file_path=out_file
            )
            success += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(samples)}] 已完成 {success} 個")
        except Exception as e:
            print(f"  ❌ {text}: {str(e)[:80]}")
    
    # 合成句子（競賽語料庫）
    sentences = [
        ("na_tarivak_sun", "na tarivak sun"),
        ("masalu", "masalu"),
        ("tima_su_ngadan", "tima su ngadan"),
        ("uri_semainu_sun", "uri semainu sun tucu"),
        ("maya_kiljavaran", "maya kiljavaran"),
    ]
    
    sent_dir = out_dir / "sentences"
    sent_dir.mkdir(exist_ok=True)
    
    for fname, text in sentences:
        out_file = str(sent_dir / f"{fname}.wav")
        try:
            tts.tts_to_file(text=text, speaker_wav=best_ref, language=working_lang, file_path=out_file)
            print(f"  ✅ 句子: {text}")
        except Exception as e:
            print(f"  ❌ 句子 {text}: {str(e)[:60]}")
    
    print(f"\n✅ 零樣本克隆完成: {success}/{len(samples)} 詞彙")
    print(f"📁 輸出: {out_dir}/")
    
    # 打包
    import shutil
    archive = Path(output_dir) / "paiwan_tts_package.tar.gz"
    shutil.make_archive(str(archive.with_suffix("")), 'gztar', str(out_dir))
    print(f"📦 打包: {archive}")
    
    return out_dir


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/workspace/paiwan_train/dataset")
    parser.add_argument("--output", default="/workspace/paiwan_train/output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--method", default="auto", choices=["auto", "finetune", "zeroshot"])
    args = parser.parse_args()
    
    print("🌾" + "=" * 58)
    print("  排灣語 XTTS v2 訓練管線")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    has_gpu = check_gpu()
    samples = prepare_dataset(args.dataset)
    
    if args.method == "auto":
        # 先嘗試微調，失敗就零樣本
        if has_gpu:
            ok = method1_finetune_lib(args.dataset, args.output, args.epochs)
            if not ok:
                print("\n🔄 微調失敗，fallback 到零樣本克隆...")
                method2_zero_shot(args.dataset, args.output)
        else:
            print("\n⚠️ 沒有 GPU，直接用零樣本克隆")
            method2_zero_shot(args.dataset, args.output)
    
    elif args.method == "finetune":
        method1_finetune_lib(args.dataset, args.output, args.epochs)
    
    elif args.method == "zeroshot":
        method2_zero_shot(args.dataset, args.output)
    
    print(f"\n{'='*60}")
    print("🎉 完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
