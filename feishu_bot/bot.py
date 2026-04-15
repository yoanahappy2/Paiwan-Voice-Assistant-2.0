"""
feishu_bot/bot.py
語聲同行 2.0 — 飛書機器人原型

架構設計：接收飛書訊息 → 轉發 RAG LLM → 回覆
此為概念原型，展示飛書整合的技術可行性。

作者: Architect
日期: 2026-04-15
"""

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# ============================================
# 飛書事件訂閱處理
# ============================================

@app.route("/feishu/webhook", methods=["POST"])
def handle_event():
    """處理飛書事件回調"""
    data = request.json

    # 1. URL 驗證（首次配置時）
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})

    # 2. 處理訊息事件
    event = data.get("event", {})
    if event.get("type") == "message":
        return handle_message(event)

    return jsonify({"code": 0})


def handle_message(event: dict) -> dict:
    """
    處理收到的飛書訊息
    
    流程:
    1. 解析用戶訊息文字
    2. 呼叫 RAG + LLM 取得回覆
    3. 透過飛書 API 回覆
    """
    message = event.get("message", {})
    content = json.loads(message.get("content", "{}"))
    user_text = content.get("text", "").strip()

    if not user_text:
        return jsonify({"code": 0})

    # 呼叫 vuvu 服務（RAG 模式）
    try:
        from llm_service import VuvuService
        vuvu = VuvuService(use_rag=True)
        result = vuvu.chat_with_thinking(user_text)
        reply = result["reply"]
    except Exception as e:
        reply = f"vuvu 暫時無法回應: {e}"

    # TODO: 透過飛書 API 回覆訊息
    # reply_to_feishu(message["message_id"], reply)

    return jsonify({"code": 0})


# ============================================
# 飛書 API 整合（示意）
# ============================================

FEISHU_API_BASE = "https://open.feishu.cn/open-apis"
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")


def get_tenant_access_token() -> str:
    """取得飛書 tenant_access_token"""
    # 實作: POST /auth/v3/tenant_access_token/internal
    pass


def reply_to_feishu(message_id: str, text: str):
    """回覆飛書訊息"""
    # 實作: POST /im/v1/messages/{message_id}/reply
    # headers = {"Authorization": f"Bearer {token}"}
    # body = {"content": json.dumps({"text": text}), "msg_type": "text"}
    pass


def send_feishu_message(open_id: str, text: str):
    """主動發送飛書訊息"""
    # 實作: POST /im/v1/messages
    pass


# ============================================
# 啟動
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("  語聲同行 2.0 — 飛書機器人原型")
    print("  注意：此為概念原型，需配置飛書 App 後才能上線")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8080, debug=True)
