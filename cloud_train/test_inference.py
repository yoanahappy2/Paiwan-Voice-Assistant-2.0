#!/usr/bin/env python3
"""
test_inference.py — 訓練後的推理測試

測試微調後的模型，生成排灣語語音樣本
"""

import os
os.environ["COQUI_TOS_AGREED"] = "1"

import sys
import csv
import torch
from pathlib import Path

# PyTorch 2.6+ 補丁
import TTS.utils.io
_orig = TTS.utils.io.load_fsspec
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
TTS.utils.io.load_fsspec = _patched

from TTS.api import TTS


def test_model(dataset_dir, output_dir, model_path=None):
    """測試 TTS 模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wavs_dir = Path(dataset_dir) / "wavs"
    
    # 載入模型
    print("📥 載入模型...")
    if model_path and Path(model_path).exists():
        print(f"  從 checkpoint 載入: {model_path}")
        # 如果有微調後的模型
        tts = TTS(model_path).to(device)
    else:
        print("  載入基礎 XTTS v2（零樣本模式）")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1").to(device)
    
    # 選參考音檔
    all_wavs = sorted(wavs_dir.glob("*.wav"))
    ref_wav = str(all_wavs[0])  # 用第一個做參考
    
    # 讀取測試詞
    metadata_path = Path(dataset_dir) / "metadata.csv"
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        samples = list(reader)
    
    # 測試輸出
    out_dir = Path(output_dir) / "test_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 測試詞（選 10 個代表性的）
    test_words = ["ita", "drusa", "tjelu", "lima", "masalu", 
                  "qadaw", "sapuy", "vavuy", "qemudjalj", "kalevelevan"]
    
    results = []
    for word in test_words:
        # 找到對應的參考音檔（如果有的話）
        word_ref = None
        for wav in all_wavs:
            if word in wav.name:
                word_ref = str(wav)
                break
        
        if word_ref is None:
            word_ref = ref_wav
        
        out_file = str(out_dir / f"test_{word}.wav")
        
        try:
            # 先嘗試排灣語 code
            tts.tts_to_file(
                text=word,
                speaker_wav=word_ref,
                language="pag",
                file_path=out_file
            )
            results.append((word, "✅", "pag"))
        except Exception as e1:
            try:
                # fallback 到 en
                tts.tts_to_file(
                    text=word,
                    speaker_wav=word_ref,
                    language="en",
                    file_path=out_file
                )
                results.append((word, "✅", "en (fallback)"))
            except Exception as e2:
                results.append((word, f"❌ {e2}", ""))
                continue
    
    # 報告
    print(f"\n{'='*60}")
    print("📊 推理測試結果")
    print(f"{'='*60}")
    for word, status, lang in results:
        print(f"  {status} {word} ({lang})")
    
    success = sum(1 for _, s, _ in results if "✅" in s)
    print(f"\n成功: {success}/{len(test_words)}")
    print(f"音檔位置: {out_dir}/")
    
    # 額外：測試句子合成（用競賽語料庫裡的句子）
    print(f"\n{'='*60}")
    print("📝 句子合成測試")
    print(f"{'='*60}")
    
    test_sentences = [
        ("masalu", "謝謝"),
        ("na tarivak sun", "你好嗎"),
        ("tima su ngadan", "你叫什麼名字"),
        ("uri semainu sun tucu", "你現在要去哪裡"),
    ]
    
    sent_dir = Path(output_dir) / "test_outputs" / "sentences"
    sent_dir.mkdir(parents=True, exist_ok=True)
    
    for paiwan, chinese in test_sentences:
        out_file = str(sent_dir / f"{paiwan[:20].replace(' ', '_')}.wav")
        try:
            tts.tts_to_file(
                text=paiwan,
                speaker_wav=ref_wav,
                language="en",
                file_path=out_file
            )
            print(f"  ✅ {paiwan}（{chinese}）")
        except Exception as e:
            print(f"  ❌ {paiwan}（{chinese}）: {e}")
    
    print(f"\n句子音檔: {sent_dir}/")
    
    # 打包所有測試結果
    import shutil
    result_archive = Path(output_dir) / "test_results.tar.gz"
    shutil.make_archive(str(result_archive.with_suffix("")), 'gztar', str(out_dir.parent))
    print(f"\n📦 測試結果已打包: {result_archive}")
    print("  下載後可用於競賽 Demo")


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/paiwan_train/dataset"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/paiwan_train/output"
    model_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    test_model(dataset_dir, output_dir, model_path)
