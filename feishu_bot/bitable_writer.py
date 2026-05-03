"""
feishu_bot/bitable_writer.py
语声同行 2.0_yu — 飛書多維表格寫入服務

功能：將 ASR 轉寫結果自動寫入飛書多維表格
用途：語料收集、語言檔案化展示、學習統計

表格：
- 語料收集表 (tbl4WiIy8QIoa8pf)
- 詞彙關聯表 (tblKXZLHvRAXio6Y)
- 綴詞規則表 (tbl0b9SwBIwHaEnH)
- 學習統計表 (自動建立)

作者: Backend_Dev
日期: 2026-04-30
"""

import os
import time
import json
import logging
from datetime import datetime, date
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
TABLE_CORPUS = "tblxxEnL08nPkmOk"
TABLE_VOCAB = "tbljOpiUGUG8Iim6"
TABLE_AFFIX = "tblb1pWYHFyevyav"

# 學習統計表 ID（啟動時自動查找/建立）
TABLE_STATS: Optional[str] = None
STATS_TABLE_NAME = "學習統計"

# Token 快取
_token_cache = {"token": None, "expire_at": 0}

# Retry 設定
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # 秒


# ============================================
# Token 管理
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
        "Content-Type": "application/json",
    }


# ============================================
# Retry 裝飾器
# ============================================

def _retry_request(method: str, url: str, payload: dict, timeout: int = 15) -> dict:
    """帶重試的 HTTP 請求"""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if method == "POST":
                resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
            elif method == "GET":
                resp = requests.get(url, headers=_headers(), timeout=timeout)
            elif method == "PUT":
                resp = requests.put(url, headers=_headers(), json=payload, timeout=timeout)
            else:
                raise ValueError(f"不支援的 HTTP method: {method}")

            data = resp.json()

            # Token 過期 → 強制刷新重試
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
# 學習統計表管理
# ============================================

def _ensure_stats_table() -> str:
    """確保學習統計表存在，返回 table_id"""
    global TABLE_STATS
    if TABLE_STATS:
        return TABLE_STATS

    base_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables"

    # 查找現有表
    try:
        data = _retry_request("GET", base_url, {})
        for table in data.get("data", {}).get("items", []):
            if table.get("name") == STATS_TABLE_NAME:
                TABLE_STATS = table["table_id"]
                logger.info(f"學習統計表已存在: {TABLE_STATS}")
                return TABLE_STATS
    except Exception as e:
        logger.warning(f"查詢表格列表失敗: {e}")

    # 建立新表
    create_body = {
        "table": {
            "name": STATS_TABLE_NAME,
            "default_view_name": "統計總覽",
        }
    }
    data = _retry_request("POST", base_url, create_body)
    TABLE_STATS = data["data"]["table_id"]
    logger.info(f"已建立學習統計表: {TABLE_STATS}")

    # 建立欄位
    fields_to_create = [
        {"field_name": "日期", "field_type": 5},          # DateTime
        {"field_name": "總對話數", "field_type": 2},       # Number
        {"field_name": "翻譯次數", "field_type": 2},
        {"field_name": "高置信度數", "field_type": 2},
        {"field_name": "中置信度數", "field_type": 2},
        {"field_name": "低置信度數", "field_type": 2},
        {"field_name": "新增詞彙數", "field_type": 2},
    ]
    field_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables/{TABLE_STATS}/fields"
    for f in fields_to_create:
        try:
            _retry_request("POST", field_url, f)
        except Exception as e:
            logger.warning(f"建立欄位 {f['field_name']} 失敗（可能已存在）: {e}")

    return TABLE_STATS


# ============================================
# 對話統計追蹤器（記憶體中累計，定期寫入 Bitable）
# ============================================

class StatsTracker:
    """追蹤對話統計，每天寫入一筆記錄"""

    def __init__(self):
        self._counts = {}  # {date_str: {metric: count}}
        self._today_key = date.today().isoformat()
        self._init_day(self._today_key)

    def _init_day(self, day_key: str):
        if day_key not in self._counts:
            self._counts[day_key] = {
                "total_chats": 0,
                "translations": 0,
                "high_conf": 0,
                "medium_conf": 0,
                "low_conf": 0,
                "new_vocab": 0,
            }

    def record(
        self,
        is_translation: bool = False,
        confidence_level: str = "",
        new_vocab: int = 0,
    ):
        """記錄一次互動"""
        day_key = date.today().isoformat()
        if day_key != self._today_key:
            # 新的一天，先 flush 舊的
            self.flush_to_bitable(self._today_key)
            self._today_key = day_key
        self._init_day(day_key)

        c = self._counts[day_key]
        c["total_chats"] += 1
        if is_translation:
            c["translations"] += 1
        if confidence_level == "high":
            c["high_conf"] += 1
        elif confidence_level == "medium":
            c["medium_conf"] += 1
        elif confidence_level == "low":
            c["low_conf"] += 1
        c["new_vocab"] += new_vocab

    def flush_to_bitable(self, day_key: Optional[str] = None):
        """將統計寫入 Bitable"""
        if not day_key:
            day_key = self._today_key

        c = self._counts.get(day_key)
        if not c or c["total_chats"] == 0:
            return

        table_id = _ensure_stats_table()
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables/{table_id}/records"

        # 轉換日期為 timestamp ms
        dt = datetime.strptime(day_key, "%Y-%m-%d")
        timestamp_ms = int(dt.timestamp() * 1000)

        fields = {
            "日期": timestamp_ms,
            "總對話數": c["total_chats"],
            "翻譯次數": c["translations"],
            "高置信度數": c["high_conf"],
            "中置信度數": c["medium_conf"],
            "低置信度數": c["low_conf"],
            "新增詞彙數": c["new_vocab"],
        }

        try:
            _retry_request("POST", url, {"fields": fields})
            logger.info(f"統計已寫入 Bitable: {day_key} → {c}")
        except Exception as e:
            logger.error(f"統計寫入 Bitable 失敗: {e}")

    def get_snapshot(self) -> dict:
        """返回今天的統計快照"""
        day_key = date.today().isoformat()
        self._init_day(day_key)
        return dict(self._counts.get(day_key, {}))


# 全域追蹤器
stats_tracker = StatsTracker()


# ============================================
# 詞彙自動回流（高置信度 → 詞彙關聯表）
# ============================================

def add_to_vocab_table(paiwan_word: str, chinese_meaning: str, word_type: str = "") -> bool:
    """
    將高置信度詞彙寫入詞彙關聯表（去重）。
    處理網路錯誤，不影響主流程。
    """
    try:
        url = (
            f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}"
            f"/tables/{TABLE_VOCAB}/records"
        )

        # 查詢是否已存在
        filter_payload = {
            "filter": {
                "conjunction": "and",
                "conditions": [
                    {
                        "field_name": "多行文本",
                        "operator": "is",
                        "value": [paiwan_word.strip()]
                    }
                ]
            }
        }
        query_url = url + "?page_size=1"
        try:
            data = _retry_request("GET", query_url, {})
            # Check if any existing record matches
            for rec in data.get("data", {}).get("items", []):
                text_val = rec.get("fields", {}).get("多行文本", "")
                if isinstance(text_val, list):
                    text_val = "\n".join(str(t) for t in text_val)
                if text_val.strip() == paiwan_word.strip():
                    logger.info(f"詞彙已存在於關聯表，跳過: {paiwan_word}")
                    return False
        except Exception as e:
            logger.warning(f"查詢詞彙關聯表失敗（繼續新增）: {e}")

        # 新增記錄
        fields = {"多行文本": paiwan_word.strip()}
        if chinese_meaning:
            fields["中文釋義"] = chinese_meaning
        if word_type:
            fields["詞類"] = word_type

        _retry_request("POST", url, {"fields": fields})
        logger.info(f"詞彙已寫入關聯表: {paiwan_word} = {chinese_meaning}")
        stats_tracker.record(new_vocab=1)
        return True
    except Exception as e:
        logger.warning(f"詞彙回流失敗（不影響主流程）: {e}")
        return False


# ============================================
# 語料收集表寫入
# ============================================

def write_transcription(
    asr_text: str,
    corrected_text: str = "",
    chinese_translation: str = "",
    confidence: float = 0.0,
    source: str = "飛書對話",
    confidence_level: str = "",
    method: str = "",
    needs_confirmation: bool = False,
) -> dict:
    """
    將一筆轉寫結果寫入飛書多維表格（帶重試）
    """
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables/{TABLE_CORPUS}/records"

    fields = {"文本": asr_text}

    if chinese_translation:
        fields["中文翻譯"] = chinese_translation
    if source:
        fields["語料來源"] = source
    if confidence > 0:
        fields["置信度"] = round(confidence * 100, 1)

    if method:
        fields["翻譯方法"] = method
    if confidence_level:
        fields["置信度等級"] = {
            "high": "🟢高", "medium": "🟡中", "low": "🔴低"
        }.get(confidence_level, confidence_level)
    if needs_confirmation:
        fields["是否正確"] = "⚠️待確認"
    elif confidence >= 0.85:
        fields["是否正確"] = "✅自動接受"

    result = _retry_request("POST", url, {"fields": fields})

    # 高置信度詞彙自動回流到詞彙關聯表
    if confidence_level == "high" and chinese_translation and corrected_text:
        try:
            add_to_vocab_table(corrected_text, chinese_translation)
        except Exception:
            pass  # 不影響主流程

    # 記錄統計
    is_translation = method in ("rag_llm", "exact", "error") or bool(chinese_translation)
    stats_tracker.record(
        is_translation=is_translation,
        confidence_level=confidence_level,
    )

    return result


def batch_write(records: list) -> dict:
    """批次寫入（帶重試）"""
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables/{TABLE_CORPUS}/records/batch_create"

    payload = []
    for r in records:
        fields = {"文本": r.get("asr_text", "")}
        if r.get("chinese_translation"):
            fields["中文翻譯"] = r["chinese_translation"]
        if r.get("source"):
            fields["語料來源"] = r["source"]
        payload.append({"fields": fields})

    return _retry_request("POST", url, {"records": payload}, timeout=30)


# ============================================
# 儀表板：讀取統計
# ============================================

def get_dashboard_stats() -> dict:
    """從 Bitable 讀取語料收集表的統計摘要"""
    url = (
        f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}"
        f"/tables/{TABLE_CORPUS}/records?page_size=500"
    )

    all_records = []
    page_token = None

    while True:
        req_url = url
        if page_token:
            req_url += f"&page_token={page_token}"
        data = _retry_request("GET", req_url, {})
        items = data.get("data", {}).get("items", [])
        all_records.extend(items)
        page_token = data.get("data", {}).get("page_token")
        if not page_token or not data.get("data", {}).get("has_more"):
            break

    # 統計分析
    total = len(all_records)
    conf_dist = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    source_dist = {}
    daily_counts = {}

    for rec in all_records:
        fields = rec.get("fields", {})

        # 置信度分布
        level = fields.get("置信度等級", "")
        if "高" in level:
            conf_dist["high"] += 1
        elif "中" in level:
            conf_dist["medium"] += 1
        elif "低" in level:
            conf_dist["low"] += 1
        else:
            conf_dist["unknown"] += 1

        # 來源分布
        src = fields.get("語料來源", "未知")
        source_dist[src] = source_dist.get(src, 0) + 1

        # 日期趨勢
        ts = fields.get("轉寫時間", 0)
        if isinstance(ts, (int, float)) and ts > 0:
            day = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        else:
            day = "unknown"
        daily_counts[day] = daily_counts.get(day, 0) + 1

    # 最近 7 天趨勢
    today = date.today()
    last_7_days = []
    for i in range(6, -1, -1):
        d = today - __import__("datetime").timedelta(days=i)
        key = d.isoformat()
        last_7_days.append({"date": key, "count": daily_counts.get(key, 0)})

    return {
        "total": total,
        "confidence_distribution": conf_dist,
        "source_distribution": source_dist,
        "last_7_days": last_7_days,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = write_transcription(
        asr_text="masalu",
        chinese_translation="謝謝",
        confidence=0.95,
        source="飛書對話",
        confidence_level="high",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n--- 儀表板統計 ---")
    stats = get_dashboard_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
