"""
feishu_bot/bot_ws.py
語聲同行 2.0 — 飛書機器人（WebSocket 長連接模式）

使用飛書官方 SDK 的 WebSocket 長連接，無需 ngrok 或公網 IP。
飛書主動連到本地 Bot，適合開發和現場演示。

啟動：python3 bot_ws.py

作者: 地陪
日期: 2026-04-27
"""

import os
import sys
import json
import time
import random
from pathlib import Path

# 路徑設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import lark_oapi as lark
from lark_oapi.ws.client import Client as WsClient
from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody

# ============================================
# 配置
# ============================================

FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")

if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
    print("❌ 請在 .env 中設定 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
    sys.exit(1)

# ============================================
# 延遲載入服務
# ============================================

_llm_service = None
_translate_service = None
_corpus_data = None


def get_llm_service():
    global _llm_service
    if _llm_service is None:
        from llm_service import VuvuService
        _llm_service = VuvuService(use_rag=True)
        print("[bot] ✅ LLM 服務已載入")
    return _llm_service


def get_translate_service():
    global _translate_service
    if _translate_service is None:
        from translate_service import PaiwanTranslator
        _translate_service = PaiwanTranslator()
        print("[bot] ✅ 翻譯服務已載入")
    return _translate_service


def get_corpus_data():
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
            print(f"[bot] ✅ 語料庫已載入: {len(_corpus_data)} 條")
        else:
            _corpus_data = []
            print("[bot] ⚠️ 語料庫未找到")
    return _corpus_data


# ============================================
# 訊息去重 + 對話歷史管理
# ============================================

_processed_messages = set()  # 已處理的 message_id，防止重複回覆
_chat_history = {}
MAX_HISTORY = 10


def add_to_history(open_id: str, role: str, content: str):
    if open_id not in _chat_history:
        _chat_history[open_id] = []
    _chat_history[open_id].append({"role": role, "content": content})
    if len(_chat_history[open_id]) > MAX_HISTORY * 2:
        _chat_history[open_id] = _chat_history[open_id][-MAX_HISTORY * 2:]


# ============================================
# Quiz 狀態
# ============================================

_quiz_state = {}


# ============================================
# 指令處理器
# ============================================

def handle_help() -> str:
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
    query = text.strip()
    if not query:
        return "請輸入要翻譯的內容，例如：\n/translate masalu\n/translate 謝謝"
    try:
        ts = get_translate_service()
        result = ts.translate(query)
        if isinstance(result, dict):
            translation = result.get("translation", str(result))
        else:
            translation = str(result)
        add_to_history(open_id, "user", f"[翻譯] {query}")
        add_to_history(open_id, "assistant", translation)
        return f"🔄 翻譯結果：\n\n{query}\n  ↓\n{translation}"
    except Exception as e:
        return f"翻譯服務暫時不可用: {e}"


def handle_learn(open_id: str) -> str:
    corpus = get_corpus_data()
    if not corpus:
        return "語料庫暫時無法使用"
    entry = random.choice(corpus)
    paiwan = entry.get("paiwan", entry.get("排灣語", ""))
    chinese = entry.get("chinese", entry.get("中文", ""))
    if not paiwan or not chinese:
        return "語料格式異常，請稍後再試"
    add_to_history(open_id, "assistant", f"[教學] {paiwan} = {chinese}")
    msg = f"📚 今日排灣語：\n\n🟢 {paiwan}\n🔵 {chinese}\n\n試著跟我說說看！💬"
    return msg


def handle_quiz(open_id: str) -> str:
    corpus = get_corpus_data()
    if not corpus:
        return "語料庫暫時無法使用"
    entry = random.choice(corpus)
    chinese = entry.get("chinese", entry.get("中文", ""))
    paiwan = entry.get("paiwan", entry.get("排灣語", ""))
    if not paiwan or not chinese:
        return "語料格式異常，請稍後再試"
    _quiz_state[open_id] = {"answer_paiwan": paiwan, "answer_chinese": chinese}
    msg = f"🎯 排灣語小測驗！\n\n請問「{chinese}」的排灣語怎麼說？\n\n（直接輸入你的答案，或輸入「不知道」跳過）"
    return msg


def handle_daily() -> str:
    corpus = get_corpus_data()
    if not corpus:
        return "語料庫暫時無法使用"
    today_seed = int(time.strftime("%Y%m%d"))
    random.seed(today_seed)
    entry = random.choice(corpus)
    random.seed()
    paiwan = entry.get("paiwan", entry.get("排灣語", ""))
    chinese = entry.get("chinese", entry.get("中文", ""))
    msg = f"☀️ 每日排灣語 ({time.strftime('%m/%d')})\n\n🟢 {paiwan}\n🔵 {chinese}\n\n— vuvu Maliq 💛"
    return msg


# ============================================
# 對話核心
# ============================================

def handle_chat(text: str, open_id: str) -> str:
    # Quiz 模式
    if open_id in _quiz_state:
        quiz = _quiz_state[open_id]
        answer = quiz["answer_paiwan"]
        chinese = quiz["answer_chinese"]
        del _quiz_state[open_id]

        if text.strip() in ["不知道", "跳過", "skip", "pass"]:
            return f"沒關係！答案是：\n\n🟢 {answer}\n🔵 {chinese}\n\n再接再厲！💪"

        user_answer = text.strip().lower().replace(" ", "")
        correct = answer.strip().lower().replace(" ", "")

        if user_answer == correct:
            return f"🎉 完全正確！太棒了！\n\n🟢 {answer} = {chinese}\n\n要再試一題嗎？輸入 /quiz"
        elif user_answer in correct or correct in user_answer:
            return f"👍 很接近了！正確答案是：\n\n🟢 {answer}\n🔵 {chinese}\n\n輸入 /quiz 再挑戰！"
        else:
            return f"答案不一樣哦～正確的是：\n\n🟢 {answer}\n🔵 {chinese}\n\n輸入 /quiz 再試一題！"

    # 一般對話
    try:
        vuvu = get_llm_service()
        result = vuvu.chat_with_thinking(text)
        reply = result.get("reply", "vuvu 暫時無法回應")
        add_to_history(open_id, "user", text)
        add_to_history(open_id, "assistant", reply)
        return reply
    except Exception as e:
        print(f"[bot] 對話異常: {e}")
        return f"ai~~ vuvu 現在有點忙，稍等一下再跟你聊天～\n（錯誤: {e}）"


# ============================================
# 訊息路由
# ============================================

def route_message(text: str, open_id: str) -> str | None:
    if text.startswith("/help") or text.startswith("/幫助"):
        return handle_help()
    elif text.startswith("/translate ") or text.startswith("/翻譯 "):
        query = text.split(" ", 1)[1] if " " in text else ""
        return handle_translate(query, open_id)
    elif text.startswith("/learn") or text.startswith("/學習"):
        return handle_learn(open_id)
    elif text.startswith("/quiz") or text.startswith("/測驗"):
        return handle_quiz(open_id)
    elif text.startswith("/daily") or text.startswith("/每日"):
        return handle_daily()
    else:
        return handle_chat(text, open_id)


# ============================================
# 飛書 SDK 事件處理
# ============================================

def handle_im_message(data) -> None:
    """處理收到的飛書訊息事件"""
    try:
        event_data = data.event
        if not event_data:
            print("[警告] event_data 為空")
            return

        message = event_data.message
        sender = event_data.sender

        msg_type = message.message_type if message else ""
        message_id = message.message_id if message else ""
        open_id = sender.sender_id.open_id if sender and sender.sender_id else "unknown"

        # 去重：同一條訊息不重複處理
        if message_id in _processed_messages:
            return
        _processed_messages.add(message_id)
        # 防止記憶體洩漏，只保留最近 500 條
        if len(_processed_messages) > 500:
            _processed_messages.clear()

        print(f"\n[收到訊息] type={msg_type}, from={open_id[:15]}...")

        if msg_type == "text":
            content_str = message.content if message else "{}"
            content = json.loads(content_str) if isinstance(content_str, str) else content_str
            text = content.get("text", "").strip()
            # 去掉 @機器人 mention
            text = text.replace("@_user_1", "").strip()

            if not text:
                return

            print(f"  內容: {text[:80]}")

            reply = route_message(text, open_id)
            if reply:
                send_reply(message_id, reply)
            
            # 寫入飛書多維表格（語料收集）
            try:
                from bitable_writer import write_transcription
                # 只寫入翻譯結果，不寫完整回覆
                translation = ""
                if text.startswith("/translate") or text.startswith("/翻譯"):
                    # 翻譯指令：提取 ↓ 後面的翻譯結果
                    if "↓" in (reply or ""):
                        translation = reply.split("↓")[-1].strip()
                    else:
                        translation = reply or ""
                else:
                    # 一般對話：不寫翻譯欄位，只記錄用戶輸入
                    translation = ""
                write_transcription(
                    asr_text=text,
                    chinese_translation=translation,
                    source="飛書對話",
                )
                print(f"  ✅ 已寫入 Bitable")
            except Exception as e:
                print(f"  ⚠️ Bitable 寫入失敗: {e}")

        elif msg_type == "audio":
            content_str = message.content if message else "{}"
            content = json.loads(content_str) if isinstance(content_str, str) else content_str
            file_key = content.get("file_key", "")
            print(f"  語音: file_key={file_key[:20]}...")
            send_reply(message_id, "🎤 語音功能即將上線！目前請先用文字跟我聊天～")

        else:
            send_reply(message_id, f"目前只支援文字訊息哦～（收到: {msg_type}）")

    except Exception as e:
        import traceback
        print(f"[處理異常] {e}")
        traceback.print_exc()


def send_reply(message_id: str, text: str):
    """透過飛書 API 回覆訊息"""
    try:
        client = create_client()
        request = ReplyMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(ReplyMessageRequestBody.builder()
                          .msg_type("text")
                          .content(json.dumps({"text": text}))
                          .build()) \
            .build()

        response = client.im.v1.message.reply(request)
        if not response.success():
            print(f"[回覆失敗] code={response.code}, msg={response.msg}")
        else:
            print(f"  ✅ 已回覆 ({len(text)} chars)")
    except Exception as e:
        print(f"[回覆異常] {e}")


def create_client() -> lark.Client:
    """建立飛書 Client"""
    return lark.Client.builder() \
        .app_id(FEISHU_APP_ID) \
        .app_secret(FEISHU_APP_SECRET) \
        .log_level(lark.LogLevel.DEBUG) \
        .build()


# ============================================
# 啟動
# ============================================

def main():
    print("=" * 50)
    print("  🏔️ 語聲同行 2.0 — 飛書機器人 (WebSocket)")
    print("=" * 50)

    # 建立 Client
    client = create_client()

    # 註冊事件處理器
    event_handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(handle_im_message)
        .build()
    )

    # 建立 WebSocket 連接
    ws_client = WsClient(
        app_id=FEISHU_APP_ID,
        app_secret=FEISHU_APP_SECRET,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )

    print()
    print("📡 正在連接飛書 WebSocket...")
    print("💡 無需 ngrok，飛書主動連到本地")
    print()

    # 啟動（阻塞）
    ws_client.start()


if __name__ == "__main__":
    main()
