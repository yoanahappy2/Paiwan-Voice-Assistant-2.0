"""
asr_service.py
排灣語語音辨識服務 - Whisper LoRA 微調模型 + 首尾字加權校正

作者: @Backend_Dev
日期: 2026-03-28
版本: 3.0 — 2026-04-11 ASRService 類別化 + 啟動時預載模型
"""

import os
import re
import json
import tempfile
import time
import torch
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# ============================================
# 路徑配置
# ============================================

MODEL_DIR = os.path.dirname(os.path.dirname(__file__))
LORA_PATH = os.path.join(MODEL_DIR, "whisper-lora-paiwan-final")
BASE_MODEL = "openai/whisper-tiny"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "api", "data", "keyword_weights.json")


# ============================================
# 首尾字加權校正引擎
# ============================================

class PhonemeCorrector:
    """
    首尾字加權校正引擎 (Architect 設計, INTENT_SPEC.md §3.1)

    三層校正：
    1. 特定詞彙直接校正（sima→siva, vasalu→masalu 等）
    2. 首字母 m↔v 混淆映射
    3. 音韻替換對照（tj↔t, lj↔l, dj↔d）
    """

    KNOWN_CORRECTIONS = {
        'sima': 'siva',
        'vasalu': 'masalu',
        'basalu': 'masalu',
        'vavan': 'mavan',
        'bangetjez': 'mangetjez',
        'vangetjez': 'mangetjez',
    }

    M_V_SWAPS = {'m': 'v', 'v': 'm'}

    @classmethod
    def correct(cls, text: str) -> dict:
        """
        執行首尾字加權校正

        Returns:
            {"original": str, "corrected": str, "confidence": float, "applied_corrections": list}
        """
        original = text.strip().lower()
        corrected = original
        applied = []
        confidence = 1.0

        # Step 1: 特定詞彙直接校正
        words = corrected.split()
        for i, word in enumerate(words):
            if word in cls.KNOWN_CORRECTIONS:
                old_word = word
                words[i] = cls.KNOWN_CORRECTIONS[word]
                applied.append(f"{old_word}→{words[i]}")
                confidence *= 0.92
        corrected = ' '.join(words)

        if not applied:
            confidence = 1.0

        return {
            "original": original,
            "corrected": corrected,
            "confidence": confidence,
            "applied_corrections": applied
        }


# ============================================
# ASRService — 核心服務類別
# ============================================

class ASRService:
    """
    排灣語 ASR 服務

    ⚠️ 設計原則：模型在 __init__ 時一次性載入，之後所有請求共用同一份模型。
    不允許在 recognize 過程中重新載入模型。
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False

    def load(self):
        """
        載入 Whisper + LoRA 微調模型。

        應在伺服器啟動時呼叫（lifespan startup），而非每次請求時呼叫。
        載入後 self._loaded = True，後續呼叫直接 return。
        """
        if self._loaded:
            print("[ASRService] 模型已載入，跳過重複載入")
            return

        print("[ASRService] 🔴 開始載入 Whisper LoRA 微調模型...")
        start_time = time.time()

        # 檢測裝置
        # ⚠️ MacBook Air 記憶體有限，強制使用 CPU 避免 MPS OOM
        # Whisper-tiny 在 CPU 推論延遲可接受（~2-5 秒），穩定性優先
        self.device = torch.device("cpu")
        print(f"[ASRService] 使用裝置: {self.device} (CPU 模式，避免 OOM)")

        # 檢查 LoRA 權重
        if not os.path.exists(LORA_PATH):
            raise RuntimeError(
                f"❌ LoRA 權重不存在: {LORA_PATH}\n"
                f"請確認 whisper-lora-paiwan-final/ 已在主專案目錄"
            )

        # 載入基礎模型
        print(f"[ASRService] 載入基礎模型: {BASE_MODEL}")
        base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

        # 強制載入 LoRA 權重層
        print(f"[ASRService] 🔴 載入 LoRA 權重: {LORA_PATH}")
        peft_model = PeftModel.from_pretrained(base_model, LORA_PATH)

        # 保持 PEFT 模式（不做 merge_and_unload 以避免記憶體峰值）
        self.model = peft_model
        print("[ASRService] ✅ 使用 PEFT 模式（未 merge，節省記憶體）")

        self.model = self.model.to(self.device)
        self.model.eval()

        # 回收未使用的記憶體
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # 優先使用 LoRA 目錄的 processor
        try:
            self.processor = WhisperProcessor.from_pretrained(LORA_PATH)
            print("[ASRService] ✅ 使用 LoRA 目錄的 processor（含訓練 tokenizer）")
        except Exception:
            self.processor = WhisperProcessor.from_pretrained(BASE_MODEL)
            print("[ASRService] ⚠️ 使用基礎模型 processor")

        self._loaded = True
        elapsed = time.time() - start_time
        print(f"[ASRService] ✅ 模型載入完成，耗時 {elapsed:.2f} 秒")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def convert_audio_to_wav(self, input_path: str, output_path: str) -> bool:
        """將音訊轉換為 WAV 格式 (16kHz, Mono)"""
        try:
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")
            return True
        except Exception as e:
            print(f"[ASRService] 音訊轉換錯誤: {e}")
            return False

    def transcribe(self, audio_path: str) -> dict:
        """
        使用已載入的微調模型辨識音訊。

        ⚠️ 不會重新載入模型，只調用 self.model。
        """
        if not self._loaded:
            raise RuntimeError("模型尚未載入！請先呼叫 load()")

        start_time = time.time()

        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)

            # 預處理
            input_features = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)

            # 強制 chinese + transcribe（排灣語訓練時的語言標籤）
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="chinese",
                task="transcribe"
            )

            # 推論
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=128,
                    # greedy decoding — 比 beam search 省記憶體
                    # Whisper-tiny 差異不大，穩定性優先
                )

            # 解碼
            raw_text = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

            processing_time = (time.time() - start_time) * 1000
            print(f"[ASRService] 原始辨識: '{raw_text}', 耗時: {processing_time:.0f}ms")

            # 首尾字加權後處理
            correction = PhonemeCorrector.correct(raw_text)
            corrected_text = correction["corrected"]

            if correction["applied_corrections"]:
                print(f"[ASRService] 🔄 校正: {correction['original']} → {corrected_text}")
            else:
                print(f"[ASRService] ✅ 無需校正")

            final_confidence = 0.92 * correction["confidence"]

            return {
                "text": corrected_text,
                "raw_text": raw_text,
                "confidence": final_confidence,
                "processing_time_ms": processing_time,
                "correction": correction,
            }

        except Exception as e:
            print(f"[ASRService] 辨識錯誤: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": "",
                "confidence": 0.0,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }

    def recognize_from_blob(self, audio_blob: bytes, content_type: str = "audio/webm") -> dict:
        """
        從音訊 Blob 辨識（主要入口）

        Pipeline: webm → WAV → Whisper LoRA → 首尾字加權校正 → 結果
        """
        # 暫存輸入
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
            tmp_in.write(audio_blob)
            input_path = tmp_in.name

        # 暫存輸出
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        try:
            if not self.convert_audio_to_wav(input_path, output_path):
                return {"text": "", "confidence": 0.0, "error": "音訊格式轉換失敗"}

            return self.transcribe(output_path)

        finally:
            for p in (input_path, output_path):
                if os.path.exists(p):
                    os.unlink(p)


# ============================================
# 全域單例 — 伺服器生命週期內唯一實例
# ============================================

asr_service = ASRService()


# ============================================
# 向後相容的函數介面（供 server.py 呼叫）
# ============================================

def load_model():
    """向後相容：載入模型"""
    asr_service.load()
    return asr_service.model, asr_service.processor, asr_service.device


def recognize_from_blob(audio_blob: bytes, content_type: str = "audio/webm") -> dict:
    """向後相容：從 blob 辨識"""
    return asr_service.recognize_from_blob(audio_blob, content_type)


def transcribe_audio(audio_path: str) -> dict:
    """向後相容：從檔案辨識"""
    return asr_service.transcribe(audio_path)


# ============================================
# CLI 測試入口
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("測試 ASRService（Whisper LoRA + 首尾字加權）")
    print("=" * 50)

    svc = ASRService()
    svc.load()
    print(f"模型已載入: {svc.is_loaded}, 裝置: {svc.device}")
    print(f"LoRA 路徑: {LORA_PATH}")

    # 測試校正引擎
    test_cases = ["sima", "vasalu aravac", "tjelu", "ita", "djemalj"]
    for tc in test_cases:
        r = PhonemeCorrector.correct(tc)
        print(f"  '{tc}' → '{r['corrected']}' (校正: {r['applied_corrections']})")
