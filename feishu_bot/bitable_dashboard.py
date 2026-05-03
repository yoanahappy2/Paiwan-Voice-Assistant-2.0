"""
feishu_bot/bitable_dashboard.py — 飛書多維表格儀表板引擎

核心價值：
不只是「記錄對話」，而是把 Bitable 變成活的學習平台：
1. 學習熱力圖 — 每個用戶每天的學習軌跡，視覺化掌握度
2. 社群語料協作 — 用戶提交新詞 → 母語者審核 → 回流語料庫
3. 詞彙掌握追蹤 — 每個詞被多少人學過、正確率多少

這才是飛書多維表格應該發揮的價值：
協作 + 結構化數據 + 可視化儀表板 = 瀕危語言的社群保存平台

作者: 地陪
日期: 2026-05-02
"""

import os
import time
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

APP_ID = os.getenv("FEISHU_APP_ID", "")
APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")

# Bitable 配置（排灣族語研究室 App）
BITABLE_APP_TOKEN = "DBe0bLbJRaWlRQsSZgJciXrLnIW"

# === 新增表格 ID（啟動時自動建立） ===
TABLE_HEATMAP = None       # 學習熱力圖
TABLE_COMMUNITY = None     # 社群語料審核
TABLE_WORD_STATS = None    # 詞彙掌握統計

# Token 快取
_token_cache = {"token": None, "expire_at": 0}
MAX_RETRIES = 3
RETRY_DELAY = 1.0


# ============================================
# Token & HTTP（複用 bitable_writer 的邏輯）
# ============================================

def get_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expire_at"] - 300:
        return _token_cache["token"]

    resp = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": APP_ID, "app_secret": APP_SECRET},
        timeout=10,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Token 獲取失敗: {data}")

    _token_cache["token"] = data["tenant_access_token"]
    _token_cache["expire_at"] = now + data.get("expire", 7200)
    return _token_cache["token"]


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }


def _retry_request(method: str, url: str, payload: dict = None, timeout: int = 15) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if method == "POST":
                resp = requests.post(url, headers=_headers(), json=payload or {}, timeout=timeout)
            elif method == "GET":
                resp = requests.get(url, headers=_headers(), timeout=timeout)
            elif method == "PATCH":
                resp = requests.patch(url, headers=_headers(), json=payload or {}, timeout=timeout)
            else:
                raise ValueError(f"不支援的 HTTP method: {method}")

            data = resp.json()
            if data.get("code") == 99991663:
                _token_cache["token"] = None
                _token_cache["expire_at"] = 0
                continue
            return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            last_err = e
            logger.warning(f"請求失敗 (第 {attempt} 次): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"請求失敗 ({MAX_RETRIES} 次重試後): {last_err}")


# ============================================
# 表格初始化
# ============================================

def _find_or_create_table(name: str, description: str = "") -> str:
    """查找或建立表格，返回 table_id"""
    base_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables"

    # 查找
    data = _retry_request("GET", base_url)
    for table in data.get("data", {}).get("items", []):
        if table.get("name") == name:
            return table["table_id"]

    # 建立（帶 default_view_name 時必須帶 fields）
    create_body = {
        "table": {
            "name": name,
            "default_view_name": description or name,
            "fields": [
                {"field_name": "索引", "type": 1},  # 預設索引欄位
            ]
        }
    }
    data = _retry_request("POST", base_url, create_body)
    table_id = data["data"]["table_id"]
    logger.info(f"已建立表格: {name} → {table_id}")
    return table_id


def _create_field(table_id: str, field_name: str, field_type: int, property: dict = None):
    """建立欄位（忽略已存在錯誤）"""
    url = (
        f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}"
        f"/tables/{table_id}/fields"
    )
    body = {"field_name": field_name, "type": field_type}
    if property:
        body["property"] = property
    try:
        _retry_request("POST", url, body)
    except Exception as e:
        if "99991402" not in str(e):  # 欄位已存在
            logger.warning(f"建立欄位 {field_name} 失敗: {e}")


def init_all_tables():
    """初始化所有儀表板表格"""
    global TABLE_HEATMAP, TABLE_COMMUNITY, TABLE_WORD_STATS

    # === 1. 學習熱力圖 ===
    TABLE_HEATMAP = _find_or_create_table("學習熱力圖", "每日學習軌跡")
    _create_field(TABLE_HEATMAP, "用戶", 1)          # Text (primary)
    _create_field(TABLE_HEATMAP, "日期", 5)          # DateTime
    _create_field(TABLE_HEATMAP, "學習詞數", 2)       # Number
    _create_field(TABLE_HEATMAP, "翻譯次數", 2)
    _create_field(TABLE_HEATMAP, "測驗正確", 2)
    _create_field(TABLE_HEATMAP, "測驗錯誤", 2)
    _create_field(TABLE_HEATMAP, "口說練習", 2)
    _create_field(TABLE_HEATMAP, "學過的詞", 1)       # Text（逗號分隔）
    _create_field(TABLE_HEATMAP, "活躍度", 3, property={  # SingleSelect
        "options": [
            {"name": "🔥 高度活躍", "color": 0},
            {"name": "☀️ 活躍", "color": 1},
            {"name": "☁️ 普通", "color": 2},
            {"name": "❄️ 低活躍", "color": 3},
        ]
    })

    # === 2. 社群語料審核 ===
    TABLE_COMMUNITY = _find_or_create_table("社群語料審核", "用戶提交新詞")
    _create_field(TABLE_COMMUNITY, "排灣語", 1)       # Text (primary)
    _create_field(TABLE_COMMUNITY, "中文釋義", 1)
    _create_field(TABLE_COMMUNITY, "提交者", 1)
    _create_field(TABLE_COMMUNITY, "提交時間", 5)     # DateTime
    _create_field(TABLE_COMMUNITY, "來源", 1)         # Bot對話 / 手動提交 / 語音辨識
    _create_field(TABLE_COMMUNITY, "審核狀態", 3, property={  # SingleSelect
        "options": [
            {"name": "⏳ 待審核", "color": 2},
            {"name": "✅ 已通過", "color": 0},
            {"name": "❌ 已駁回", "color": 3},
            {"name": "🔄 需修正", "color": 1},
        ]
    })
    _create_field(TABLE_COMMUNITY, "審核者", 1)
    _create_field(TABLE_COMMUNITY, "備註", 1)

    # === 3. 詞彙掌握統計 ===
    TABLE_WORD_STATS = _find_or_create_table("詞彙掌握統計", "每個詞的學習數據")
    _create_field(TABLE_WORD_STATS, "排灣語", 1)      # Text (primary)
    _create_field(TABLE_WORD_STATS, "中文", 1)
    _create_field(TABLE_WORD_STATS, "學習人數", 2)     # Number
    _create_field(TABLE_WORD_STATS, "正確次數", 2)
    _create_field(TABLE_WORD_STATS, "錯誤次數", 2)
    _create_field(TABLE_WORD_STATS, "正確率", 2)       # Number (%)
    _create_field(TABLE_WORD_STATS, "掌握度", 3, property={  # SingleSelect
        "options": [
            {"name": "🟢 已掌握", "color": 0},
            {"name": "🟡 學習中", "color": 1},
            {"name": "🔴 困難詞", "color": 3},
            {"name": "⚪ 新詞", "color": 2},
        ]
    })
    _create_field(TABLE_WORD_STATS, "詞彙主題", 1)
    _create_field(TABLE_WORD_STATS, "真人錄音", 3, property={
        "options": [
            {"name": "✅ 有", "color": 0},
            {"name": "❌ 無", "color": 3},
        ]
    })

    logger.info("✅ 儀表板表格初始化完成")
    return {
        "heatmap": TABLE_HEATMAP,
        "community": TABLE_COMMUNITY,
        "word_stats": TABLE_WORD_STATS,
    }


def get_table_ids() -> dict:
    """取得表格 ID（如未初始化則初始化）"""
    if not TABLE_HEATMAP:
        return init_all_tables()
    return {
        "heatmap": TABLE_HEATMAP,
        "community": TABLE_COMMUNITY,
        "word_stats": TABLE_WORD_STATS,
    }


# ============================================
# 學習熱力圖
# ============================================

# 記憶體中的每日計數器（避免頻繁寫 Bitable）
_daily_tracker = {}  # {(open_id, date_str): {metrics}}


def _get_tracker_key(open_id: str) -> tuple:
    return (open_id, date.today().isoformat())


def _get_or_create_tracker(open_id: str) -> dict:
    key = _get_tracker_key(open_id)
    if key not in _daily_tracker:
        _daily_tracker[key] = {
            "words_learned": [],
            "translations": 0,
            "quiz_correct": 0,
            "quiz_wrong": 0,
            "speak_practice": 0,
        }
    return _daily_tracker[key]


def record_heatmap_event(
    open_id: str,
    event_type: str,  # "learn" | "translate" | "quiz_correct" | "quiz_wrong" | "speak"
    word: str = "",
):
    """記錄一次學習事件到熱力圖追蹤器"""
    tracker = _get_or_create_tracker(open_id)

    if event_type == "learn" and word:
        if word not in tracker["words_learned"]:
            tracker["words_learned"].append(word)
    elif event_type == "translate":
        tracker["translations"] += 1
        if word and word not in tracker["words_learned"]:
            tracker["words_learned"].append(word)
    elif event_type == "quiz_correct":
        tracker["quiz_correct"] += 1
        if word and word not in tracker["words_learned"]:
            tracker["words_learned"].append(word)
    elif event_type == "quiz_wrong":
        tracker["quiz_wrong"] += 1
        if word and word not in tracker["words_learned"]:
            tracker["words_learned"].append(word)
    elif event_type == "speak":
        tracker["speak_practice"] += 1
        if word and word not in tracker["words_learned"]:
            tracker["words_learned"].append(word)


def flush_heatmap_to_bitable(open_id: str, user_display_name: str = ""):
    """將今天的學習數據寫入 Bitable 熱力圖表"""
    key = _get_tracker_key(open_id)
    tracker = _daily_tracker.get(key)
    if not tracker:
        return

    total_actions = (
        len(tracker["words_learned"])
        + tracker["translations"]
        + tracker["quiz_correct"]
        + tracker["quiz_wrong"]
        + tracker["speak_practice"]
    )
    if total_actions == 0:
        return

    tables = get_table_ids()
    url = (
        f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}"
        f"/tables/{tables['heatmap']}/records"
    )

    # 活躍度分級
    if total_actions >= 15:
        activity = "🔥 高度活躍"
    elif total_actions >= 8:
        activity = "☀️ 活躍"
    elif total_actions >= 3:
        activity = "☁️ 普通"
    else:
        activity = "❄️ 低活躍"

    today = date.today()
    timestamp_ms = int(datetime(today.year, today.month, today.day).timestamp() * 1000)

    fields = {
        "用戶": user_display_name or open_id[:8],
        "日期": timestamp_ms,
        "學習詞數": len(tracker["words_learned"]),
        "翻譯次數": tracker["translations"],
        "測驗正確": tracker["quiz_correct"],
        "測驗錯誤": tracker["quiz_wrong"],
        "口說練習": tracker["speak_practice"],
        "學過的詞": ", ".join(tracker["words_learned"][:30]),  # 限制長度
        "活躍度": activity,
    }

    # 嘗試更新今天的記錄（如果已存在）
    try:
        filter_url = url + f"?filter={{\"conjunction\":\"and\",\"conditions\":[{{\"field_name\":\"用戶\",\"operator\":\"is\",\"value\":[\"{fields['用戶']}\"]}},{{\"field_name\":\"日期\",\"operator\":\"is\",\"value\":[{timestamp_ms}]}}]}}&page_size=1"
        existing = _retry_request("GET", filter_url)
        items = existing.get("data", {}).get("items", [])
        if items:
            # 更新現有記錄
            record_id = items[0]["record_id"]
            update_url = f"{url}/{record_id}"
            _retry_request("PATCH", update_url, {"fields": fields})
            logger.info(f"熱力圖已更新: {fields['用戶']} {today.isoformat()}")
            return
    except Exception as e:
        logger.warning(f"查詢熱力圖記錄失敗（將新增）: {e}")

    # 新增記錄
    try:
        _retry_request("POST", url, {"fields": fields})
        logger.info(f"熱力圖已寫入: {fields['用戶']} {today.isoformat()}")
    except Exception as e:
        logger.error(f"熱力圖寫入失敗: {e}")


# ============================================
# 社群語料協作
# ============================================

def submit_community_word(
    paiwan: str,
    chinese: str,
    submitter: str = "飛書用戶",
    source: str = "Bot提交",
) -> dict:
    """
    用戶提交新詞彙到社群語料審核表

    Returns:
        {"success": bool, "message": str}
    """
    tables = get_table_ids()
    url = (
        f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}"
        f"/tables/{tables['community']}/records"
    )

    # 去重：檢查是否已提交
    try:
        filter_url = url + f"?filter={{\"conjunction\":\"and\",\"conditions\":[{{\"field_name\":\"排灣語\",\"operator\":\"is\",\"value\":[\"{paiwan.strip()}\"]}},{{\"field_name\":\"中文釋義\",\"operator\":\"is\",\"value\":[\"{chinese.strip()}\"]}}]}}&page_size=1"
        existing = _retry_request("GET", filter_url)
        items = existing.get("data", {}).get("items", [])
        if items:
            return {"success": False, "message": f"「{paiwan}」（{chinese}）已經有人提交過了！"}
    except Exception:
        pass

    timestamp_ms = int(time.time() * 1000)

    fields = {
        "排灣語": paiwan.strip(),
        "中文釋義": chinese.strip(),
        "提交者": submitter,
        "提交時間": timestamp_ms,
        "來源": source,
        "審核狀態": "⏳ 待審核",
    }

    try:
        _retry_request("POST", url, {"fields": fields})
        logger.info(f"社群語料已提交: {paiwan} = {chinese}")
        return {"success": True, "message": f"✅ 已提交「{paiwan}」（{chinese}）到社群語料庫！\n母語者審核通過後會自動加入詞彙表。💛"}
    except Exception as e:
        return {"success": False, "message": f"提交失敗: {e}"}


def get_pending_reviews(limit: int = 20) -> list:
    """獲取待審核的社群語料"""
    tables = get_table_ids()
    url = (
        f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}"
        f"/tables/{tables['community']}/records"
        f"?filter={{\"conjunction\":\"and\",\"conditions\":[{{\"field_name\":\"審核狀態\",\"operator\":\"is\",\"value\":[\"⏳ 待審核\"]}}]}}"
        f"&page_size={limit}"
    )
    data = _retry_request("GET", url)
    return data.get("data", {}).get("items", [])


def review_community_word(record_id: str, status: str, reviewer: str = "", note: str = "") -> bool:
    """
    審核社群語料

    Args:
        record_id: Bitable 記錄 ID
        status: "✅ 已通過" / "❌ 已駁回" / "🔄 需修正"
        reviewer: 審核者
        note: 備註
    """
    tables = get_table_ids()
    url = (
        f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}"
        f"/tables/{tables['community']}/records/{record_id}"
    )

    fields = {"審核狀態": status}
    if reviewer:
        fields["審核者"] = reviewer
    if note:
        fields["備註"] = note

    try:
        _retry_request("PATCH", url, {"fields": fields})
        return True
    except Exception as e:
        logger.error(f"審核更新失敗: {e}")
        return False


# ============================================
# 詞彙掌握統計
# ============================================

def update_word_stats(word: str, chinese: str, is_correct: bool, theme: str = "", has_audio: bool = False):
    """更新詞彙掌握統計"""
    tables = get_table_ids()
    url = (
        f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}"
        f"/tables/{tables['word_stats']}/records"
    )

    word_lower = word.strip().lower()

    # 查找現有記錄
    try:
        filter_url = url + f"?filter={{\"conjunction\":\"and\",\"conditions\":[{{\"field_name\":\"排灣語\",\"operator\":\"is\",\"value\":[\"{word_lower}\"]}}]}}&page_size=1"
        existing = _retry_request("GET", filter_url)
        items = existing.get("data", {}).get("items", [])

        if items:
            # 更新
            rec = items[0]
            record_id = rec["record_id"]
            old_fields = rec.get("fields", {})
            learners = old_fields.get("學習人數", 0) or 0
            correct = old_fields.get("正確次數", 0) or 0
            wrong = old_fields.get("錯誤次數", 0) or 0

            if is_correct:
                correct += 1
            else:
                wrong += 1

            total = correct + wrong
            accuracy = round(correct / total * 100, 1) if total > 0 else 0

            # 掌握度
            if accuracy >= 80:
                mastery = "🟢 已掌握"
            elif accuracy >= 50:
                mastery = "🟡 學習中"
            else:
                mastery = "🔴 困難詞"

            update_fields = {
                "正確次數": correct,
                "錯誤次數": wrong,
                "正確率": accuracy,
                "掌握度": mastery,
            }
            update_url = f"{url}/{record_id}"
            _retry_request("PATCH", update_url, {"fields": update_fields})
            return
    except Exception as e:
        logger.warning(f"查詢詞彙統計失敗（將新增）: {e}")

    # 新增
    fields = {
        "排灣語": word_lower,
        "中文": chinese.strip(),
        "學習人數": 1,
        "正確次數": 1 if is_correct else 0,
        "錯誤次數": 0 if is_correct else 1,
        "正確率": 100.0 if is_correct else 0.0,
        "掌握度": "⚪ 新詞",
        "詞彙主題": theme,
        "真人錄音": "✅ 有" if has_audio else "❌ 無",
    }

    try:
        _retry_request("POST", url, {"fields": fields})
    except Exception as e:
        logger.error(f"詞彙統計寫入失敗: {e}")


def increment_word_learners(word: str):
    """增加詞彙的學習人數"""
    tables = get_table_ids()
    url = (
        f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}"
        f"/tables/{tables['word_stats']}/records"
    )

    word_lower = word.strip().lower()

    try:
        filter_url = url + f"?filter={{\"conjunction\":\"and\",\"conditions\":[{{\"field_name\":\"排灣語\",\"operator\":\"is\",\"value\":[\"{word_lower}\"]}}]}}&page_size=1"
        existing = _retry_request("GET", filter_url)
        items = existing.get("data", {}).get("items", [])

        if items:
            rec = items[0]
            record_id = rec["record_id"]
            old_learners = rec.get("fields", {}).get("學習人數", 0) or 0
            update_url = f"{url}/{record_id}"
            _retry_request("PATCH", update_url, {"fields": {"學習人數": old_learners + 1}})
    except Exception:
        pass


# ============================================
# /stats 指令：個人學習熱力圖（文字版）
# ============================================

def format_stats_reply(open_id: str, profile_data: dict) -> str:
    """
    生成 /stats 指令的文字版熱力圖

    Args:
        open_id: 用戶 ID
        profile_data: UserProfile.get_stats() 的返回值
    """
    stats = profile_data
    lines = ["📊 你的排灣語學習報告", ""]

    # 學習天數
    first_seen = stats.get("first_seen", "")
    last_seen = stats.get("last_seen", "")
    if first_seen:
        lines.append(f"📅 學習天數：從 {first_seen} 開始")
    if last_seen:
        lines.append(f"🕐 最後學習：{last_seen}")

    # 連續天數
    streak = stats.get("streak", 0)
    streak_display = "🔥 " * min(streak, 7) if streak else ""
    lines.append(f"\n{streak_display} 連續學習：{streak} 天")

    # 詞彙量
    total = stats.get("total_learned", 0)
    correct = stats.get("total_correct", 0)
    wrong = stats.get("total_wrong", 0)
    accuracy = stats.get("accuracy", 0)

    lines.append(f"\n📚 詞彙量：{total} 詞")
    lines.append(f"✅ 答對：{correct} 次")
    lines.append(f"❌ 答錯：{wrong} 次")
    lines.append(f"🎯 正確率：{accuracy}%")

    # 正確率條
    bar_len = 20
    filled = int(accuracy / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    lines.append(f"   [{bar}] {accuracy}%")

    # 熱力圖（最近 7 天）
    lines.append("\n📆 最近 7 天學習熱力圖：")
    lines.append("  一  二  三  四  五  六  日")

    # 從 tracker 中取最近 7 天數據
    today = date.today()
    week_data = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        key = (open_id, d.isoformat())
        tracker = _daily_tracker.get(key)
        if tracker:
            total_actions = (
                len(tracker["words_learned"])
                + tracker["translations"]
                + tracker["quiz_correct"]
                + tracker["quiz_wrong"]
                + tracker["speak_practice"]
            )
        else:
            total_actions = 0
        week_data.append(total_actions)

    # 找到今天是星期幾
    weekday_today = today.weekday()  # 0=Mon
    # 對齊：把 week_data 映射到正確的星期位置
    heat_chars = ["  "] * 7
    for i, count in enumerate(week_data):
        d = today - timedelta(days=6 - i)
        wd = d.weekday()
        if count >= 10:
            heat_chars[wd] = "🔴"
        elif count >= 5:
            heat_chars[wd] = "🟠"
        elif count >= 2:
            heat_chars[wd] = "🟡"
        elif count >= 1:
            heat_chars[wd] = "🟢"
        else:
            heat_chars[wd] = "⚪"

    lines.append("  " + "  ".join(heat_chars))

    # 圖例
    lines.append("\n  🔴≥10  🟠≥5  🟡≥2  🟢≥1  ⚪=0")

    # 排名/鼓勵
    if total >= 50:
        lines.append("\n🏆 排灣語高手！你已經學了 50+ 詞！")
    elif total >= 20:
        lines.append("\n💪 很棒！繼續保持！")
    elif total >= 5:
        lines.append("\n🌱 好的開始！每天學一點，積少成多！")
    else:
        lines.append("\n✨ 開始你的排灣語之旅吧！輸入 /learn 或 /daily")

    lines.append("\n🔗 完整數據儀表板：")
    lines.append("https://lcn1rq9sb717.feishu.cn/base/DBe0bLbJRaWlRQsSZgJciXrLnIW")

    return "\n".join(lines)


# ============================================
# 對話驅動推薦（語料農場）
# ============================================

def get_related_recommendation(learned_word: str, learned_set: set) -> str | None:
    """
    學了一個詞後，推薦相關詞（引導用戶繼續探索）

    Returns:
        推薦訊息，或 None
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from knowledge_graph import WORD_GROUPS

    word_lower = learned_word.strip().lower()

    for group in WORD_GROUPS:
        for pw, cn in group["words"]:
            if pw.lower() == word_lower:
                # 找同群組中還沒學過的
                unlearned = [
                    (w, c) for w, c in group["words"]
                    if w.lower() != word_lower and w.lower() not in learned_set
                ]
                if unlearned:
                    import random
                    rec = random.choice(unlearned)
                    return (
                        f"💡 你剛學了「{learned_word}」！\n"
                        f"你知道同主題的「{rec[0]}」嗎？它也是「{group['theme']}」相關的詞！\n"
                        f"意思是：{rec[1]}\n\n"
                        f"輸入 /learn 繼續探索！"
                    )
    return None


# ============================================
# 定期 flush（Bot 主循環呼叫）
# ============================================

_last_flush = {"date": None, "open_ids": set()}


def periodic_flush(open_id: str, display_name: str = ""):
    """
    每天至少 flush 一次熱力圖數據到 Bitable
    Bot 可以在每次互動後呼叫
    """
    today = date.today().isoformat()
    if _last_flush["date"] != today:
        _last_flush["date"] = today
        _last_flush["open_ids"] = set()

    if open_id not in _last_flush["open_ids"]:
        _last_flush["open_ids"].add(open_id)
        flush_heatmap_to_bitable(open_id, display_name)


# ============================================
# CLI 測試
# ============================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 儀表板初始化測試 ===\n")
    tables = init_all_tables()
    print(f"學習熱力圖: {tables['heatmap']}")
    print(f"社群語料審核: {tables['community']}")
    print(f"詞彙掌握統計: {tables['word_stats']}")

    # 測試社群提交
    print("\n--- 社群語料提交測試 ---")
    result = submit_community_word("tjengelay", "愛", "測試用戶", "Bot提交")
    print(result["message"])

    # 測試熱力圖
    print("\n--- 熱力圖測試 ---")
    record_heatmap_event("test_user_001", "learn", "masalu")
    record_heatmap_event("test_user_001", "translate", "kipusalu")
    record_heatmap_event("test_user_001", "quiz_correct", "masalu")
    record_heatmap_event("test_user_001", "speak", "nanguangua")

    tracker = _daily_tracker.get(("test_user_001", date.today().isoformat()), {})
    print(f"今日追蹤: {tracker}")

    flush_heatmap_to_bitable("test_user_001", "測試用戶")
    print("✅ 熱力圖已寫入")

    # 測試 /stats
    print("\n--- /stats 測試 ---")
    stats_reply = format_stats_reply("test_user_001", {
        "total_learned": 3,
        "total_correct": 1,
        "total_wrong": 0,
        "accuracy": 100,
        "streak": 1,
        "last_seen": date.today().isoformat(),
        "first_seen": date.today().isoformat(),
    })
    print(stats_reply)
