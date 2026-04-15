"""
modules/asr_evaluator.py
排灣語發音評測引擎

核心功能：
1. 語音辨識（呼叫原專案 ASR API）
2. 與標準答案對比（音韻規則 + Levenshtein）
3. 綴詞結構分析
4. 生成結構化評測報告

作者: Backend_Dev
日期: 2026-04-15
"""

import requests
import json
import re
from dataclasses import dataclass
from typing import Optional

# ============================================
# 音韻規則引擎（從原專案遷移）
# ============================================

class PhonemeEngine:
    """
    排灣語音韻規則引擎
    
    處理排灣語特有的音變現象：
    - tj/t, lj/l, dj/d 舌尖音變化
    - m/v 混淆（老年發音者特徵）
    - 特定詞彙校正
    """
    
    KNOWN_CORRECTIONS = {
        'sima': 'siva',
        'vasalu': 'masalu',
        'basalu': 'masalu',
        'vavan': 'mavan',
        'bangetjez': 'mangetjez',
        'vangetjez': 'mangetjez',
    }
    
    PHONETIC_RULES = {
        'tj→t': ('tj', 't'),
        'lj→l': ('lj', 'l'),
        'dj→d': ('dj', 'd'),
    }
    
    M_V_SWAPS = {'m': 'v', 'v': 'm'}
    
    @classmethod
    def correct(cls, text: str) -> dict:
        """三層校正：特定詞 → m/v 互換 → 音韻替換"""
        original = text.strip().lower()
        corrected = original
        corrections = []
        
        # Layer 1: 特定詞彙
        words = corrected.split()
        for i, w in enumerate(words):
            if w in cls.KNOWN_CORRECTIONS:
                corrections.append(f"{w} → {cls.KNOWN_CORRECTIONS[w]}")
                words[i] = cls.KNOWN_CORRECTIONS[w]
        corrected = ' '.join(words)
        
        # Layer 2: 音韻替換
        for rule_name, (pattern, replacement) in cls.PHONETIC_RULES.items():
            if pattern in corrected:
                corrected = corrected.replace(pattern, replacement)
                corrections.append(rule_name)
        
        return {
            "original": original,
            "corrected": corrected,
            "corrections": corrections,
            "was_corrected": original != corrected,
        }
    
    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return PhonemeEngine.levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]
    
    @classmethod
    def compare(cls, recognized: str, target: str) -> dict:
        """
        比對辨識結果與標準答案
        返回分數、音韻分析、詳細比對
        """
        # 正規化
        norm = lambda t: re.sub(r'[^\w\s]', '', t.lower().strip())
        r_norm = norm(recognized)
        t_norm = norm(target)
        
        # 校正
        r_corrected = cls.correct(r_norm)
        
        # 計算距離
        exact = r_corrected["corrected"] == t_norm
        dist = cls.levenshtein(r_corrected["corrected"], t_norm)
        max_len = max(len(r_corrected["corrected"]), len(t_norm), 1)
        similarity = (max_len - dist) / max_len
        
        # 評分
        score = round(similarity * 100)
        if exact:
            score = 100
        
        return {
            "score": score,
            "exact_match": exact,
            "similarity": round(similarity, 3),
            "levenshtein_distance": dist,
            "recognized_raw": recognized,
            "recognized_normalized": r_norm,
            "recognized_corrected": r_corrected["corrected"],
            "target": target,
            "corrections_applied": r_corrected["corrections"],
            "grade": _grade(score),
        }


def _grade(score: int) -> str:
    if score >= 95: return "Perfect"
    if score >= 85: return "Excellent"
    if score >= 70: return "Good"
    if score >= 50: return "Fair"
    return "Try Again"


# ============================================
# 綴詞分析器
# ============================================

class AffixAnalyzer:
    """
    排灣語綴詞結構分析器
    
    排灣語是黏著語（agglutinative），通過前綴、中綴、後綴變化表達語法功能。
    """
    
    PREFIXES = {
        'na': {'type': 'tense', 'label': '過去/完成', 'description': '表示動作已完成或狀態已改變'},
        'uri': {'type': 'tense', 'label': '未來/將要', 'description': '表示即將發生的動作'},
        'maya': {'type': 'negation', 'label': '禁止', 'description': '否定命令，意為「不要」'},
        'ini': {'type': 'negation', 'label': '陳述否定', 'description': '否定陳述，意為「不」'},
        'inika': {'type': 'negation', 'label': '屬性否定', 'description': '否定屬性，意為「不是」'},
        'ma-': {'type': 'voice', 'label': '狀態', 'description': '表示狀態或自動動作'},
        'pa-': {'type': 'voice', 'label': '致使', 'description': '使某事發生'},
        'si-': {'type': 'voice', 'label': '相互', 'description': '共同動作'},
        'mu-': {'type': 'voice', 'label': '移動', 'description': '表示移動動作'},
        'kem-': {'type': 'voice', 'label': '動作', 'description': '一般動作標記'},
        'tja-': {'type': 'voice', 'label': '集合', 'description': '集合或共同所有'},
    }
    
    @classmethod
    def analyze(cls, text: str) -> dict:
        """分析句子中的綴詞結構"""
        text_lower = text.lower().strip()
        found = []
        
        for prefix, info in cls.PREFIXES.items():
            clean = prefix.rstrip('-')
            if text_lower.startswith(clean + ' ') or text_lower.startswith(clean):
                found.append({
                    "prefix": clean,
                    "type": info['type'],
                    "label": info['label'],
                    "description": info['description'],
                    "rest": text_lower[len(clean):].strip(),
                })
        
        return {
            "text": text,
            "affixes_found": found,
            "count": len(found),
            "structure_description": cls._describe_structure(found, text),
        }
    
    @classmethod
    def _describe_structure(cls, affixes: list, text: str) -> str:
        if not affixes:
            return f"「{text}」為基礎句型，無明顯綴詞標記。"
        
        parts = []
        for a in affixes:
            parts.append(f"前綴「{a['prefix']}」({a['label']}：{a['description']})")
        return "此句包含：" + "；".join(parts)


# ============================================
# 意圖分類器（基於規則）
# ============================================

class IntentClassifier:
    """排灣語意圖分類（不使用 LLM，純規則）"""
    
    KEYWORDS = {
        'greeting': ['tarivak', 'masalu', 'nanguaq', 'pacunan', 'vaikanga'],
        'identity_query': ['tima'],
        'quantity_query': ['pida', 'mapida'],
        'location_query': ['inu', 'inuan', 'ainu', 'kemasinu', 'semainu', 'kasinu'],
        'object_query': ['anema'],
        'negation': ['ini', 'inika'],
        'prohibition': ['maya'],
        'question': ['tjengelay', 'padjalim', 'rutjemalupung'],
        'culture': ['paiwan', 'sekacalisian', 'kacalisian'],
        'poetry': ['vangaw', 'qemudalj', 'sasiq'],
    }
    
    @classmethod
    def classify(cls, text: str) -> dict:
        text_lower = text.lower()
        scores = {}
        for intent, keywords in cls.KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return {"intent": "unknown", "confidence": 0.0, "all_scores": {}}
        
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best] / total
        
        return {
            "intent": best,
            "confidence": round(confidence, 2),
            "keyword_hits": scores,
        }


# ============================================
# ASR 服務（呼叫原專案 API）
# ============================================

ASR_API = "http://127.0.0.1:8000/api/audio/recognize"

def call_asr(audio_path: str) -> dict:
    """呼叫原專案 ASR API 進行語音辨識"""
    try:
        with open(audio_path, 'rb') as f:
            resp = requests.post(
                ASR_API,
                files={"file": ("audio.wav", f, "audio/wav")},
                timeout=15,
            )
        resp.raise_for_status()
        data = resp.json()
        return {
            "text": data.get("corrected", ""),
            "confidence": data.get("confidence", 0.0),
            "logic": data.get("logic", ""),
            "raw": data.get("debugInfo", {}).get("rawASR", ""),
            "corrections": data.get("debugInfo", {}).get("correction", {}),
            "processing_ms": data.get("processingTimeMs", 0),
        }
    except requests.exceptions.ConnectionError:
        return {"error": "ASR 服務未啟動。請先啟動原專案後端（localhost:8000）"}
    except Exception as e:
        return {"error": str(e)}


# ============================================
# 整合評測
# ============================================

@dataclass
class EvalResult:
    """完整評測結果"""
    target_paiwan: str
    target_chinese: str
    recognized_text: str = ""
    score: int = 0
    grade: str = ""
    exact_match: bool = False
    similarity: float = 0.0
    corrections: list = None
    affix_analysis: dict = None
    intent: dict = None
    asr_detail: dict = None
    error: str = ""
    
    def __post_init__(self):
        if self.corrections is None:
            self.corrections = []


def evaluate_pronunciation(audio_path: str, target_paiwan: str, target_chinese: str) -> EvalResult:
    """
    完整的發音評測流程：
    1. ASR 辨識
    2. 音韻校正比對
    3. 綴詞分析
    4. 意圖分類
    """
    # Step 1: ASR
    asr_result = call_asr(audio_path)
    if asr_result.get("error"):
        return EvalResult(
            target_paiwan=target_paiwan,
            target_chinese=target_chinese,
            error=asr_result["error"],
        )
    
    recognized = asr_result["text"]
    
    # Step 2: 音韻比對
    comparison = PhonemeEngine.compare(recognized, target_paiwan)
    
    # Step 3: 綴詞分析
    affix = AffixAnalyzer.analyze(target_paiwan)
    
    # Step 4: 意圖
    intent = IntentClassifier.classify(target_paiwan)
    
    return EvalResult(
        target_paiwan=target_paiwan,
        target_chinese=target_chinese,
        recognized_text=recognized,
        score=comparison["score"],
        grade=comparison["grade"],
        exact_match=comparison["exact_match"],
        similarity=comparison["similarity"],
        corrections=comparison["corrections_applied"],
        affix_analysis=affix,
        intent=intent,
        asr_detail=asr_result,
    )


def evaluate_text(user_text: str, target_paiwan: str, target_chinese: str) -> EvalResult:
    """文字輸入的評測（不需要 ASR）"""
    comparison = PhonemeEngine.compare(user_text, target_paiwan)
    affix = AffixAnalyzer.analyze(target_paiwan)
    intent = IntentClassifier.classify(target_paiwan)
    
    return EvalResult(
        target_paiwan=target_paiwan,
        target_chinese=target_chinese,
        recognized_text=user_text,
        score=comparison["score"],
        grade=comparison["grade"],
        exact_match=comparison["exact_match"],
        similarity=comparison["similarity"],
        corrections=comparison["corrections_applied"],
        affix_analysis=affix,
        intent=intent,
    )
