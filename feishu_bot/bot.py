"""
feishu_bot/bot.py
語聲同行 2.0 — 飛書機器人（完整版）

功能：
- 文字對話（RAG + LLM，vuvu Maliq 角色）
- /translate 指令（雙向翻譯）
- /learn 指令（教學模式）
- /quiz 指令（互動測驗）
- /help 指令
- 語音訊息接收（下載→ASR→處理）

啟動：python bot.py
外部暴露：ngrok http 8080

作者: 地陪
日期: 2026-04-27
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from flask import Flask, request, jsonify

# 將專案根目錄加入 path，以便 import 上層服務
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from auth import get_tenant_access_token, feishu_api_headers, FEISHU_API_BASE
from media import download_audio, ogg_to_wav

# ============================================
# 載入核心服務
# ============================================

# 延遲載入，避免 import 時就噴錯
_llm_service = None
_rag_service = None
_translate_service = None
_corpus_data = None


def get_llm_service():
    global _llm_service
    if _llm_service is None:
        from llm_service import VuvuService
        _llm_service = VuvuService(use_rag=True)
        print("[bot] LLM 服務已載入")
    return _llm_service


def get_translate_service():
    global _translate_service
    if _translate_service is None:
        from translate_service import PaiwanTranslator
        _translate_service = PaiwanTranslator()
        print("[bot] 翻譯服務已載入")
    return _translate_service


def get_corpus_data():
    """載入語料庫（用於 /learn 和 /quiz）"""
    global _corpus_data
    if _corpus_data is None:
        corpus_path = PROJECT_ROOT / "data" / "expanded_corpus.json"
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, dict) and "entries" in raw:
                    _corpus_data = raw["entries"]
                elif isinstance(raw, list):
                    _corpus_data = raw
                else:
                    _corpus_data = []
            print(f"[bot] 語料庫已載入: {len(_corpus_data)} 條")
        else:
            _corpus_data = []
            print("[bot] ⚠️ 語料庫未找到")
    return _corpus_data


# ============================================
# 對話歷史管理（簡單記憶體版）
# ============================================

# { open_id: [ {role, content}, ... ] }
_chat_history = {}
MAX_HISTORY = 10  # 每個用戶保留最近 10 輪


def add_to_history(open_id: str, role: str, content: str):
    if open_id not in _chat_history:
        _chat_history[open_id] = []
    _chat_history[open_id].append({"role": role, "content": content})
    # 保留最近 MAX_HISTORY 條
    if len(_chat_history[open_id]) > MAX_HISTORY * 2:
        _chat_history[open_id] = _chat_history[open_id][-MAX_HISTORY * 2:]


def get_history(open_id: str) -> list:
    return _chat_history.get(open_id, [])


# ============================================
# Quiz 狀態管理
# ============================================

# { open_id: { "question": ..., "answer_paiwan": ..., "answer_chinese": ... } }
_quiz_state = {}


# ============================================
# 飛書訊息 API
# ============================================

def reply_text(message_id: str, text: str) -> bool:
    """回覆文字訊息"""
    headers = feishu_api_headers()
    url = f"{FEISHU_API_BASE}/im/v1/messages/{message_id}/reply"
    payload = {
        "msg_type": "text",
        "content": json.dumps({"text": text})
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        data = resp.json()
        if data.get("code") != 0:
            print(f"[bot] 回覆失敗: {data}")
            return False
        return True
    except Exception as e:
        print(f"[bot] 回覆異常: {e}")
        return False


# 需要 requests
import requests


# ============================================
# 指令處理
# ============================================

def handle_help(open_id: str) -> str:
    return (
        "🏔️ 語聲同行 2.0 — 排灣語 AI 助手\n"
        "我是 vuvu Maliq，你的排灣族祖母語伴！\n\n"
        "📋 指令列表：\n"
        "• /help — 查看幫助\n"
        "• /translate <文字> — 排灣語⇄中文翻譯\n"
        "• /learn — 學一個新詞\n"
        "• /quiz — 排灣語小測驗\n"
        "• /daily — 每日一句排灣語\n\n"
        "💡 也可以直接跟我聊天，我會用排灣語+中文回覆你！"
    )


def handle_translate(text: str, open_id: str) -> str:
    """處理翻譯指令"""
    query = text.strip()
    if not query:
        return "請輸入要翻譯的內容，例如：\n/translate masalu\n/translate 謝謝"
    
    try:
        ts = get_translate_service()
        result = ts.translate(query)
        if isinstance(result, dict):
            translation = result.get('translation', str(result))
        else:
            translation = str(result)
        add_to_history(open_id, "user", f"[翻譯] {query}")
        add_to_history(open_id, "assistant", translation)
        return f"🔄 翻譯結果：\n\n{query}\n  ↓\n{translation}"
    except Exception as e:
        return f"翻譯服務暫時不可用: {e}"


def handle_learn(open_id: str) -> str:
    """隨機教一個排灣語詞彙"""
    corpus = get_corpus_data()
    if not corpus:
        return "語料庫暫時無法使用"
    
    entry = random.choice(corpus)
    paiwan = entry.get("paiwan", entry.get("排灣語", ""))
    chinese = entry.get("chinese", entry.get("中文", ""))
    
    if not paiwan or not chinese:
        return "語料格式異常，請稍後再試"
    
    add_to_history(open_id, "assistant", f"[教學] {paiwan} = {chinese}")
    
    msg = f"📚 今日排灣語：\n\n"
    msg += f"🟢 {paiwan}\n"
    msg += f"🔵 {chinese}\n\n"
    msg += f"試著跟我說說看！💬"
    return msg


def handle_quiz(open_id: str, message_id: str) -> str:
    """啟動或回答測驗"""
    corpus = get_corpus_data()
    if not corpus:
        return "語料庫暫時無法使用"
    
    # 如果有正在進行的測驗，先清除
    entry = random.choice(corpus)
    chinese = entry.get("chinese", entry.get("中文", ""))
    paiwan = entry.get("paiwan", entry.get("排灣語", ""))
    
    if not paiwan or not chinese:
        return "語料格式異常，請稍後再試"
    
    # 儲存測驗狀態
    _quiz_state[open_id] = {
        "answer_paiwan": paiwan,
        "answer_chinese": chinese,
    }
    
    msg = f"🎯 排灣語小測驗！\n\n"
    msg += f"請問「{chinese}」的排灣語怎麼說？\n\n"
    msg += f"（直接輸入你的答案，或輸入「不知道」跳過）"
    return msg


def handle_daily(open_id: str) -> str:
    """每日一句"""
    corpus = get_corpus_data()
    if not corpus:
        return "語料庫暫時無法使用"
    
    # 用日期作為隨機種子，每天固定一句
    today_seed = int(time.strftime("%Y%m%d"))
    random.seed(today_seed)
    entry = random.choice(corpus)
    random.seed()  # 恢復隨機
    
    paiwan = entry.get("paiwan", entry.get("排灣語", ""))
    chinese = entry.get("chinese", entry.get("中文", ""))
    
    msg = f"☀️ 每日排灣語 ({time.strftime('%m/%d')})\n\n"
    msg += f"🟢 {paiwan}\n"
    msg += f"🔵 {chinese}\n\n"
    msg += f"— vuvu Maliq 💛"
    return msg


# ============================================
# 對話處理（核心）
# ============================================

def handle_chat(text: str, open_id: str) -> str:
    """一般對話處理"""
    # 先檢查是否在 quiz 模式中
    if open_id in _quiz_state:
        quiz = _quiz_state[open_id]
        answer = quiz["answer_paiwan"]
        chinese = quiz["answer_chinese"]
        del _quiz_state[open_id]
        
        if text.strip() in ["不知道", "跳過", "skip", "pass"]:
            return f"沒關係！答案是：\n\n🟢 {answer}\n🔵 {chinese}\n\n再接再厲！💪"
        
        # 簡單匹配：去掉空格和標點後比較
        user_answer = text.strip().lower().replace(" ", "")
        correct = answer.strip().lower().replace(" ", "")
        
        if user_answer == correct:
            return f"🎉 完全正確！太棒了！\n\n🟢 {answer} = {chinese}\n\n要再試一題嗎？輸入 /quiz"
        elif user_answer in correct or correct in user_answer:
            return f"👍 很接近了！正確答案是：\n\n🟢 {answer}\n🔵 {chinese}\n\n輸入 /quiz 再挑戰！"
        else:
            return f"答案不一樣哦～正確的是：\n\n🟢 {answer}\n🔵 {chinese}\n\n輸入 /quiz 再試一題！"
    
    # 一般對話 → LLM + RAG
    try:
        vuvu = get_llm_service()
        result = vuvu.chat_with_thinking(text)
        reply = result.get("reply", "vuvu 暫時無法回應")
        
        add_to_history(open_id, "user", text)
        add_to_history(open_id, "assistant", reply)
        
        return reply
    except Exception as e:
        print(f"[bot] 對話異常: {e}")
        return f"ai~~ vuvu 現在有点忙，稍等一下再跟你聊天～\n（錯誤: {e}）"


# ============================================
# Flask 路由
# ============================================

app = Flask(__name__)

# 飛書事件訂閱的驗證 token（從飛書後台獲取）
VERIFICATION_TOKEN = os.getenv("FEISHU_VERIFICATION_TOKEN", "")
ENCRYPT_KEY = os.getenv("FEISHU_ENCRYPT_KEY", "")


@app.route("/feishu/webhook", methods=["POST"])
def handle_webhook():
    """飛書事件回調入口"""
    data = request.json
    print(f"[webhook] 收到事件: type={data.get('type')}, schema={data.get('schema')}")
    
    # 1. URL 驗證
    if data.get("type") == "url_verification":
        challenge = data.get("challenge", "")
        print(f"[webhook] URL 驗證: challenge={challenge[:20]}...")
        return jsonify({"challenge": challenge})
    
    # 2. 驗證 token（如果有設置）
    if VERIFICATION_TOKEN and data.get("token") != VERIFICATION_TOKEN:
        print(f"[webhook] Token 驗證失敗")
        return jsonify({"code": -1})
    
    # 3. 處理事件
    event = data.get("event", {})
    event_type = event.get("type", "")
    
    # 訊息事件
    if event_type == "message":
        # 取得訊息資訊
        message = event.get("message", {})
        msg_type = message.get("message_type", "")
        message_id = message.get("message_id", "")
        sender = event.get("sender", {})
        open_id = sender.get("open_id", "unknown")
        
        print(f"[webhook] 訊息: type={msg_type}, from={open_id[:15]}...")
        
        # 文字訊息
        if msg_type == "text":
            content = json.loads(message.get("content", "{}"))
            text = content.get("text", "").strip()
            
            if not text:
                return jsonify({"code": 0})
            
            # 去掉 @機器人 的 mention
            text = text.replace("@_user_1", "").strip()
            
            print(f"[webhook] 文字內容: {text[:50]}...")
            
            # 路由指令
            reply = route_message(text, open_id, message_id)
            if reply:
                reply_text(message_id, reply)
        
        # 語音訊息
        elif msg_type == "audio":
            content = json.loads(message.get("content", "{}"))
            file_key = content.get("file_key", "")
            
            print(f"[webhook] 語音訊息: file_key={file_key[:20]}...")
            
            # 下載語音
            ogg_path = download_audio(message_id, file_key)
            if ogg_path:
                wav_path = ogg_to_wav(ogg_path)
                if wav_path:
                    # 呼叫 ASR API
                    asr_text = call_asr(wav_path)
                    if asr_text:
                        reply_text(message_id, f"🎤 你說的排灣語：{asr_text}\n\n讓 vuvu 來回覆你...")
                        reply = handle_chat(asr_text, open_id)
                        reply_text(message_id, reply)
                    else:
                        reply_text(message_id, "語音辨識失敗，請再試一次或改用文字輸入 😅")
                else:
                    reply_text(message_id, "語音格式轉換失敗 😅")
            else:
                reply_text(message_id, "語音下載失敗，請再試一次 😅")
        
        else:
            reply_text(message_id, f"目前只支援文字和語音訊息哦～（收到類型: {msg_type}）")
    
    return jsonify({"code": 0})


def route_message(text: str, open_id: str, message_id: str) -> str | None:
    """路由訊息到對應處理器"""
    # 指令路由
    if text.startswith("/help") or text.startswith("/幫助"):
        return handle_help(open_id)
    elif text.startswith("/translate ") or text.startswith("/翻譯 "):
        query = text.split(" ", 1)[1] if " " in text else ""
        return handle_translate(query, open_id)
    elif text.startswith("/learn") or text.startswith("/學習"):
        return handle_learn(open_id)
    elif text.startswith("/quiz") or text.startswith("/測驗"):
        return handle_quiz(open_id, message_id)
    elif text.startswith("/daily") or text.startswith("/每日"):
        return handle_daily(open_id)
    else:
        # 一般對話
        return handle_chat(text, open_id)


def call_asr(audio_path: str) -> str | None:
    """呼叫本地 ASR API"""
    try:
        import requests as req
        with open(audio_path, "rb") as f:
            resp = req.post(
                "http://127.0.0.1:8000/asr",
                files={"audio": f},
                timeout=30
            )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("text", "")
        else:
            print(f"[bot] ASR 失敗: HTTP {resp.status_code}")
            return None
    except Exception as e:
        print(f"[bot] ASR 異常: {e}")
        return None


@app.route("/health", methods=["GET"])
def health():
    """健康檢查"""
    return jsonify({
        "status": "ok",
        "service": "語聲同行 2.0 飛書 Bot",
        "version": "2.0.0",
        "timestamp": int(time.time()),
    })


# ============================================
# 啟動
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("  🏔️ 語聲同行 2.0 — 飛書機器人")
    print("  Bot 後端啟動中...")
    print("=" * 50)
    
    # 預檢：確認憑證
    app_id = os.getenv("FEISHU_APP_ID", "")
    app_secret = os.getenv("FEISHU_APP_SECRET", "")
    if not app_id or not app_secret:
        print("❌ 請在 .env 中設定 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
        sys.exit(1)
    
    # 預檢：測試 token
    try:
        token = get_tenant_access_token()
        print(f"✅ 飛書認證成功 (token: {token[:15]}...)")
    except Exception as e:
        print(f"❌ 飛書認證失敗: {e}")
        print("請檢查 FEISHU_APP_ID 和 FEISHU_APP_SECRET 是否正確")
        sys.exit(1)
    
    print()
    print("📋 Webhook URL: http://localhost:8080/feishu/webhook")
    print("🏥 Health:      http://localhost:8080/health")
    print()
    print("💡 下一步:")
    print("   1. 開 ngrok: ngrok http 8080")
    print("   2. 把 ngrok URL + /feishu/webhook 填到飛書後台事件訂閱")
    print("   3. 在飛書私聊 Bot 測試")
    print()
    
    app.run(host="0.0.0.0", port=8080, debug=True)
