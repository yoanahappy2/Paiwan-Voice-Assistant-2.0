"""
語聲同行 - 排灣族語學習平台後端 API
FastAPI 伺服器主程式

作者: @Backend_Dev
日期: 2026-03-28
版本: 1.1
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import json
import uvicorn
import os
import time

# ASR 服務（全域單例）
from api.asr_service import asr_service


# ============================================
# Lifespan — 伺服器啟動時預載模型
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時載入 ASR 模型，關閉時清理"""
    print("[Server] 🚀 伺服器啟動中，預載 ASR 模型...")
    try:
        asr_service.load()
        print(f"[Server] ✅ ASR 模型就緒 (device={asr_service.device})")
    except Exception as e:
        print(f"[Server] ⚠️ ASR 模型預載失敗（將在首次請求時重試）: {e}")
    yield
    print("[Server] 🔴 伺服器關閉")


# ============================================
# FastAPI 應用程式初始化
# ============================================

app = FastAPI(
    title="語聲同行 API",
    description="排灣族語學習平台後端服務",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境請改為具體域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 靜態檔案服務 - 音檔
# ============================================

AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audio")
print(f"[DEBUG] AUDIO_DIR: {AUDIO_DIR}, exists: {os.path.exists(AUDIO_DIR)}")

# ============================================
# 路由
# ============================================

from fastapi.responses import FileResponse

@app.get("/audio/{filename:path}")
async def get_audio(filename: str):
    """取得音檔"""
    # 嘗試直接匹配
    file_path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mp4")
    
    # 嘗試用單詞匹配（對應 sXX-單詞.m4a 格式）
    # 從 filename 提取單詞（如 "ita一.m4a" -> "ita"）
    word_part = filename.replace('.m4a', '').replace('.mp3', '').replace('一', '').replace('二', '').replace('三', '').strip()
    
    # 搜尋匹配的音檔
    if os.path.exists(AUDIO_DIR):
        for f in os.listdir(AUDIO_DIR):
            if f.endswith('.m4a') and word_part in f:
                return FileResponse(os.path.join(AUDIO_DIR, f), media_type="audio/mp4")
    
    raise HTTPException(status_code=404, detail=f"音檔 {filename} 不存在")

# ============================================
# 資料模型定義
# ============================================

# 請求模型
class GameMatchRequest(BaseModel):
    """遊戲匹配請求"""
    wordId: str
    targetWord: str
    recognizedText: str
    audioBlob: Optional[str] = None  # Base64 編碼的音訊（可選）


class ChatIntentRequest(BaseModel):
    """對話意圖識別請求"""
    text: str
    sessionId: Optional[str] = None
    context: Optional[dict] = None


# 回應模型
class MatchDetails(BaseModel):
    """匹配細節"""
    exactMatch: bool
    phoneticSimilarity: float
    appliedRules: List[str]


class GameMatchResponse(BaseModel):
    """遊戲匹配回應"""
    wordId: str
    recognizedText: str
    pronunciationScore: int  # 0-100
    uiStateTrigger: str  # 'SUCCESS' | 'FAILED'
    matchDetails: MatchDetails
    isCorrect: bool
    pointsEarned: int
    successLevel: Optional[str] = None  # 'Perfect' | 'Excellent' | 'Good'
    feedbackMessage: str
    debug: Optional[dict] = None


class AffixAnalysis(BaseModel):
    """綴詞分析結果"""
    prefix: Optional[str] = None
    type: Optional[str] = None
    matchedWord: Optional[str] = None


class ChatIntentResponse(BaseModel):
    """對話意圖識別回應"""
    recognizedText: str
    intent: str
    confidence: float
    affixAnalysis: AffixAnalysis
    llmResponse: str
    suggestedReply: Optional[str] = None
    debugInfo: Optional[dict] = None


# ============================================
# 音韻匹配引擎
# ============================================

class PronunciationMatcher:
    """發音匹配引擎"""
    
    # 音韻規則定義
    PHONETIC_RULES = {
        'tj→t': {'pattern': 'tj', 'replacement': 't'},
        'lj→l': {'pattern': 'lj', 'replacement': 'l'},
        'dj→d': {'pattern': 'dj', 'replacement': 'd'},
    }
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """計算 Levenshtein 距離"""
        if len(s1) < len(s2):
            return PronunciationMatcher.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @classmethod
    def normalize_text(cls, text: str) -> str:
        """文字正則化"""
        normalized = text.lower().strip()
        # 移除標點符號
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        return normalized
    
    @classmethod
    def apply_phonetic_rules(cls, text: str) -> tuple:
        """應用音韻規則，返回 (轉換後文字, 應用的規則列表)"""
        applied_rules = []
        result = text
        
        for rule_name, rule in cls.PHONETIC_RULES.items():
            if rule['pattern'] in result:
                result = result.replace(rule['pattern'], rule['replacement'])
                applied_rules.append(rule_name)
        
        return result, applied_rules
    
    @classmethod
    def calculate_score(cls, recognized: str, target: str) -> dict:
        """計算發音分數"""
        import time
        start_time = time.time()
        
        # Step 1: 基本正則化
        normalized_recognized = cls.normalize_text(recognized)
        normalized_target = cls.normalize_text(target)
        
        # Step 2: 檢查完全匹配
        exact_match = normalized_recognized == normalized_target
        
        # Step 3: 應用音韻規則
        phonetic_recognized, rules_recognized = cls.apply_phonetic_rules(normalized_recognized)
        phonetic_target, rules_target = cls.apply_phonetic_rules(normalized_target)
        
        all_applied_rules = list(set(rules_recognized + rules_target))
        
        # Step 4: 計算相似度
        if exact_match:
            similarity = 1.0
        else:
            distance = cls.levenshtein_distance(phonetic_recognized, phonetic_target)
            max_length = max(len(phonetic_recognized), len(phonetic_target))
            similarity = (max_length - distance) / max_length if max_length > 0 else 0
        
        score = round(similarity * 100)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'score': score,
            'exact_match': exact_match,
            'phonetic_similarity': similarity,
            'applied_rules': all_applied_rules,
            'normalized_recognized': phonetic_recognized,
            'normalized_target': phonetic_target,
            'levenshtein_distance': cls.levenshtein_distance(phonetic_recognized, phonetic_target),
            'processing_time_ms': processing_time
        }


# ============================================
# 遊戲服務
# ============================================

class GameService:
    """遊戲服務類別"""
    
    def __init__(self):
        self.vocabulary = self._load_vocabulary()
    
    def _load_vocabulary(self) -> dict:
        """載入單詞庫"""
        try:
            with open('api/data/game_vocabulary.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'game_levels': {}}
    
    def get_word_by_id(self, word_id: str) -> Optional[dict]:
        """根據 ID 取得單詞"""
        for level_id, level in self.vocabulary.get('game_levels', {}).items():
            for word in level.get('words', []):
                if word['id'] == word_id:
                    return word
        return None
    
    @staticmethod
    def calculate_points(score: int) -> int:
        """計算獲得分數"""
        if score >= 95:
            return 150  # Perfect
        elif score >= 85:
            return 130  # Excellent
        elif score >= 70:
            return 100  # Good
        else:
            return 0    # Failed
    
    @staticmethod
    def get_success_level(score: int) -> Optional[str]:
        """取得成功等級"""
        if score >= 95:
            return 'Perfect'
        elif score >= 85:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        return None
    
    @staticmethod
    def get_feedback_message(score: int, success_level: Optional[str]) -> str:
        """取得反饋訊息"""
        if success_level == 'Perfect':
            return '🌟 Perfect! 完美發音！'
        elif success_level == 'Excellent':
            return '⭐ Excellent! 很棒！'
        elif success_level == 'Good':
            return '✅ Good! 不錯！'
        else:
            return '再試一次！'


# ============================================
# 意圖識別服務
# ============================================

class IntentService:
    """意圖識別服務"""
    
    def __init__(self):
        self.training_data = self._load_training_data()
        self.affix_weights = self._load_affix_weights()
    
    def _load_training_data(self) -> list:
        """載入訓練資料"""
        try:
            data = []
            with open('api/data/intent_training.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        except FileNotFoundError:
            return []
    
    def _load_affix_weights(self) -> dict:
        """載入綴詞權重"""
        # 預設權重
        return {
            'na': 0.35,      # 過去時
            'uri': 0.35,     # 未來時
            'maya': 0.30,    # 否定命令
            'inika': 0.30,   # 屬性否定
            'ini': 0.30,     # 陳述否定
            'ma-': 0.20,     # 狀態
            'pa-': 0.20,     # 致使
            'si-': 0.20,     # 相互
            'mu-': 0.20,     # 移動
            'kem-': 0.15,    # 動作
            'ki-': 0.15,     # 獲取
            'tja-': 0.15,    # 集合
        }
    
    def analyze_affix(self, text: str) -> AffixAnalysis:
        """分析綴詞"""
        text_lower = text.lower()
        
        # 時態綴詞
        if text_lower.startswith('na '):
            return AffixAnalysis(prefix='na', type='past_tense', matchedWord='na')
        if text_lower.startswith('uri '):
            return AffixAnalysis(prefix='uri', type='future_tense', matchedWord='uri')
        
        # 否定綴詞
        if text_lower.startswith('maya'):
            return AffixAnalysis(prefix='maya', type='prohibition', matchedWord='maya')
        if 'inika' in text_lower:
            return AffixAnalysis(prefix='inika', type='attribute_negation', matchedWord='inika')
        if text_lower.startswith('ini'):
            return AffixAnalysis(prefix='ini', type='statement_negation', matchedWord='ini')
        
        # 動詞綴詞
        if text_lower.startswith('ma'):
            return AffixAnalysis(prefix='ma-', type='stative', matchedWord=text_lower[:5])
        if text_lower.startswith('pa'):
            return AffixAnalysis(prefix='pa-', type='causative', matchedWord=text_lower[:5])
        if text_lower.startswith('si'):
            return AffixAnalysis(prefix='si-', type='reciprocal', matchedWord=text_lower[:5])
        
        return AffixAnalysis()
    
    def classify_intent(self, text: str) -> dict:
        """分類意圖"""
        text_lower = text.lower()
        
        # 關鍵詞匹配
        if any(kw in text_lower for kw in ['tima', 'ngadan']):
            return {'intent': 'identity_query', 'confidence': 0.91}
        if any(kw in text_lower for kw in ['pida', 'mapida']):
            return {'intent': 'quantity_query', 'confidence': 0.89}
        if any(kw in text_lower for kw in ['inu', 'inuan', 'ainu']):
            return {'intent': 'location_query', 'confidence': 0.88}
        if any(kw in text_lower for kw in ['anema']):
            return {'intent': 'object_query', 'confidence': 0.87}
        if any(kw in text_lower for kw in ['tarivak', 'masalu']):
            return {'intent': 'greeting', 'confidence': 0.93}
        if any(kw in text_lower for kw in ['maya', 'ini', 'inika']):
            return {'intent': 'negation', 'confidence': 0.92}
        
        return {'intent': 'statement', 'confidence': 0.75}
    
    def generate_response(self, intent: str, text: str) -> str:
        """生成 LLM 回應"""
        responses = {
            'greeting': '你好！我是排灣族語智慧助教，有什麼可以幫你的？',
            'identity_query': '關於身份的問題，讓我幫你解答。',
            'quantity_query': '關於數量的問題，讓我想想...',
            'location_query': '關於位置的問題，我可以幫你。',
            'object_query': '這是什麼東西呢？讓我看看...',
            'negation': '了解，這是一個否定句。',
            'statement': '謝謝你的分享！'
        }
        return responses.get(intent, '我聽到了，請繼續說。')


# 初始化服務
game_service = GameService()
intent_service = IntentService()


# ============================================
# API 端點 - 遊戲軌道
# ============================================

@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "語聲同行 API",
        "version": "1.0.0",
        "endpoints": {
            "game": "/api/game/match",
            "chat": "/api/chat/intent"
        }
    }


@app.get("/api/game/levels")
async def get_game_levels():
    """取得遊戲關卡列表"""
    levels = []
    for level_id, level in game_service.vocabulary.get('game_levels', {}).items():
        levels.append({
            "levelId": level_id,
            "name": level['name'],
            "icon": level['icon'],
            "wordCount": len(level['words']),
            "unlockCondition": level['unlock_condition']
        })
    return {"levels": levels}


@app.get("/api/game/level/{level_id}/words")
async def get_level_words(level_id: str):
    """取得關卡單詞"""
    level = game_service.vocabulary.get('game_levels', {}).get(level_id)
    if not level:
        raise HTTPException(status_code=404, detail="關卡不存在")
    return {"levelId": level_id, "words": level['words']}


@app.post("/api/game/match", response_model=GameMatchResponse)
async def match_pronunciation(request: GameMatchRequest):
    """
    遊戲發音匹配端點
    
    接收 ASR 辨識結果，計算發音分數並返回 UI 狀態觸發器
    """
    # Step 1: 驗證單詞存在
    word = game_service.get_word_by_id(request.wordId)
    if not word:
        raise HTTPException(status_code=404, detail=f"單詞 {request.wordId} 不存在")
    
    # Step 2: 計算發音分數
    match_result = PronunciationMatcher.calculate_score(
        request.recognizedText,
        request.targetWord
    )
    
    # Step 3: 判定結果
    score = match_result['score']
    is_correct = score >= 70
    ui_state_trigger = 'SUCCESS' if is_correct else 'FAILED'
    success_level = game_service.get_success_level(score) if is_correct else None
    points_earned = game_service.calculate_points(score)
    feedback_message = game_service.get_feedback_message(score, success_level)
    
    # Step 4: 構建回應
    return GameMatchResponse(
        wordId=request.wordId,
        recognizedText=request.recognizedText,
        pronunciationScore=score,
        uiStateTrigger=ui_state_trigger,
        matchDetails=MatchDetails(
            exactMatch=match_result['exact_match'],
            phoneticSimilarity=match_result['phonetic_similarity'],
            appliedRules=match_result['applied_rules']
        ),
        isCorrect=is_correct,
        pointsEarned=points_earned,
        successLevel=success_level,
        feedbackMessage=feedback_message,
        debug={
            "normalizedTarget": match_result['normalized_target'],
            "normalizedRecognized": match_result['normalized_recognized'],
            "levenshteinDistance": match_result['levenshtein_distance'],
            "processingTimeMs": round(match_result['processing_time_ms'], 2)
        }
    )


# ============================================
# API 端點 - 對話軌道
# ============================================

@app.post("/api/chat/intent", response_model=ChatIntentResponse)
async def classify_intent(request: ChatIntentRequest):
    """
    對話意圖識別端點
    
    分析綴詞、分類意圖、生成 LLM 回應
    """
    # Step 1: 綴詞分析
    affix_analysis = intent_service.analyze_affix(request.text)
    
    # Step 2: 意圖分類
    intent_result = intent_service.classify_intent(request.text)
    
    # Step 3: LLM 回應生成
    llm_response = intent_service.generate_response(intent_result['intent'], request.text)
    
    # Step 4: 構建回應
    return ChatIntentResponse(
        recognizedText=request.text,
        intent=intent_result['intent'],
        confidence=intent_result['confidence'],
        affixAnalysis=affix_analysis,
        llmResponse=llm_response,
        suggestedReply=f"你可以試著說：{request.text}",
        debugInfo={
            "context": request.context,
            "sessionId": request.sessionId
        }
    )


@app.get("/api/chat/intents")
async def get_intent_types():
    """取得所有意圖類型"""
    return {
        "intents": [
            {"id": "greeting", "label": "問候", "color": "#3B82F6"},
            {"id": "identity_query", "label": "身份詢問", "color": "#8B5CF6"},
            {"id": "quantity_query", "label": "數量詢問", "color": "#10B981"},
            {"id": "location_query", "label": "位置詢問", "color": "#F59E0B"},
            {"id": "object_query", "label": "物品詢問", "color": "#EC4899"},
            {"id": "time_query", "label": "時間詢問", "color": "#6366F1"},
            {"id": "negation", "label": "否定", "color": "#EF4444"},
            {"id": "statement", "label": "陳述", "color": "#6B7280"},
            {"id": "thanks", "label": "感謝", "color": "#F97316"},
            {"id": "farewell", "label": "道別", "color": "#14B8A6"},
            {"id": "daily_action", "label": "日常動作", "color": "#A855F7"}
        ]
    }


# ============================================
# 音訊辨識端點 (MediaRecorder + 後端模型)
# ============================================

from fastapi import UploadFile, File

from api.asr_service import asr_service

class RecognizeResponse(BaseModel):
    """音訊辨識回應（與前端 useASR.ts 對齊）"""
    corrected: str
    confidence: float
    logic: str
    processingTimeMs: Optional[float] = None
    debugInfo: Optional[dict] = None


@app.post("/api/audio/recognize", response_model=RecognizeResponse)
async def recognize_audio(file: UploadFile = File(...)):
    """
    音訊辨識端點
    
    接收前端錄製的音訊 Blob，使用微調語音模型辨識
    """
    start_time = time.time()
    
    try:
        # 讀取音訊資料
        audio_data = await file.read()
        audio_size = len(audio_data)
        content_type = file.content_type or "audio/webm"
        
        print(f"[ASR] 收到音訊: {audio_size} bytes, 類型: {content_type}")
        
        # 檢查音訊大小（太小可能是空錄音）
        if audio_size < 1000:
            return RecognizeResponse(
                corrected="",
                confidence=0.0,
                logic="音訊太短或為空",
                processingTimeMs=(time.time() - start_time) * 1000,
                debugInfo={"error": "音訊太短或為空"}
            )
        
        # 使用微調模型辨識
        result = asr_service.recognize_from_blob(audio_data, content_type)
        
        processing_time = (time.time() - start_time) * 1000
        
        # 建構校正邏輯說明
        recognized = result.get("text", "")
        raw_text = result.get("raw_text", recognized)
        correction = result.get("correction", {})
        
        # 建構邏輯說明：包含原始辨識 + 校正過程
        logic_parts = []
        if recognized:
            logic_parts.append(f"Whisper LoRA 辨識: {raw_text}")
            if correction.get("applied_corrections"):
                logic_parts.append(f"校正: {' → '.join(correction['applied_corrections'])}")
                logic_parts.append(f"校正後: {recognized}")
            else:
                logic_parts.append("無需校正")
        logic_desc = " | ".join(logic_parts) if logic_parts else "模型未辨識到語音"
        
        return RecognizeResponse(
            corrected=recognized,
            confidence=result.get("confidence", 0.0),
            logic=logic_desc,
            processingTimeMs=processing_time,
            debugInfo={
                "audioSize": audio_size,
                "audioType": content_type,
                "rawASR": raw_text,
                "correction": correction,
                "modelResult": result
            }
        )
        
    except Exception as e:
        print(f"[ASR] 錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return RecognizeResponse(
            corrected="",
            confidence=0.0,
            logic=f"辨識錯誤: {str(e)}",
            processingTimeMs=(time.time() - start_time) * 1000,
            debugInfo={"error": str(e)}
        )


# ============================================
# 健康檢查
# ============================================

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "service": "語聲同行 API",
        "version": "1.1.0"
    }


# ============================================
# 主程式入口
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
