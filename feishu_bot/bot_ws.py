"""
feishu_bot/bot_ws.py
语声同行 2.0_yu — 飛書機器人（WebSocket 長連接模式）

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
import tempfile
import subprocess
from pathlib import Path

import requests as http_requests

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

# 排灣語音檔目錄
PAIWAN_TTS_DIR = PROJECT_ROOT / "paiwan_tts"
PAIWAN_SENTENCES_DIR = PAIWAN_TTS_DIR / "sentences"
# 真人母語者錄音（優先使用）
REAL_AUDIO_DIR = Path("/Users/sbb-mei/paiwan_asr_project/audio")

# 建立真人音檔索引 {排灣語詞彙: 檔案路徑}，s03 優先
_real_audio_index = None


def _build_real_audio_index() -> dict:
    """掃描真人錄音目錄，建立詞彙→檔案索引，s03 優先"""
    global _real_audio_index
    if _real_audio_index is not None:
        return _real_audio_index
    _real_audio_index = {}
    if not REAL_AUDIO_DIR.exists():
        return _real_audio_index
    import glob
    # 先掃 s01/s02（低優先），再掃 s03（覆蓋，高優先）
    for pattern in ["s01-*.m4a", "s02-*.m4a", "s03-*.m4a"]:
        for f in sorted(REAL_AUDIO_DIR.glob(pattern)):
            base = f.stem  # e.g. s03-kina
            # 去掉 s01-/s02-/s03- 前綴
            word = base.split("-", 1)[1] if "-" in base else base
            _real_audio_index[word.lower()] = str(f)
    print(f"[音檔] 真人錄音索引: {len(_real_audio_index)} 個詞彙")
    return _real_audio_index

# ASR 服務
ASR_API_URL = "http://127.0.0.1:8000/api/audio/recognize"

# ============================================
# 延遲載入服務
# ============================================

_llm_service = None
_translate_service = None
_corpus_data = None
_active_learner = None
_confirmation_state = {}  # {open_id: {input, translation, level, entry}}
_user_profile_instance = None


def get_user_profile():
    global _user_profile_instance
    if _user_profile_instance is None:
        from user_profile import UserProfile
        _user_profile_instance = UserProfile()
        print("[bot] ✅ 用戶學習檔案已載入")
    return _user_profile_instance


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


def get_active_learner():
    global _active_learner
    if _active_learner is None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from active_learner import ActiveLearner
        _active_learner = ActiveLearner()
        print("[bot] ✅ 主動學習引擎已載入")
    return _active_learner


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
# Quiz + Speak 狀態
# ============================================

_quiz_state = {}
_speak_state = {}  # {open_id: {"answer", "answer_chinese", "mode", "attempts", "audio_path"}}


# ============================================
# 語音輔助函數
# ============================================

def _get_paiwan_audio_path(paiwan_text: str) -> str | None:
    """根據排灣語文字找到對應音檔路徑（優先真人錄音）"""
    if not paiwan_text:
        return None
    word = paiwan_text.strip().lower()
    
    # 1. 優先：真人母語者錄音
    idx = _build_real_audio_index()
    if word in idx:
        return idx[word]
    # 嘗試空格替換
    word_underscore = word.replace(" ", " ")
    if word_underscore in idx:
        return idx[word_underscore]
    
    # 2. 次要：XTTS 合成音檔
    fname = word.replace(" ", "_") + ".wav"
    word_path = PAIWAN_TTS_DIR / fname
    if word_path.exists():
        return str(word_path)
    sent_path = PAIWAN_SENTENCES_DIR / fname
    if sent_path.exists():
        return str(sent_path)
    
    return None


def _get_tenant_token() -> str:
    """取得飛書 tenant_access_token"""
    resp = http_requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={
            "app_id": FEISHU_APP_ID,
            "app_secret": FEISHU_APP_SECRET,
        },
        timeout=10,
    )
    data = resp.json()
    return data.get("tenant_access_token", "")


def download_audio_message(message_id: str, file_key: str) -> str | None:
    """
    下載飛書語音訊息的音檔

    Returns:
        本地音檔路徑，或 None
    """
    try:
        token = _get_tenant_token()
        if not token:
            print("[語音下載] 無法取得 tenant_access_token")
            return None

        url = (
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}"
            f"/resources/{file_key}?type=file"
        )
        resp = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"[語音下載] HTTP {resp.status_code}")
            return None

        # 飛書語音通常是 opus 格式，存為臨時檔
        tmp_dir = tempfile.mkdtemp(prefix="paiwan_voice_")
        # 嘗試從 header 取得副檔名，預設 .opus
        content_type = resp.headers.get("Content-Type", "")
        ext = ".opus"
        if "mp4" in content_type:
            ext = ".m4a"
        elif "wav" in content_type:
            ext = ".wav"
        elif "ogg" in content_type:
            ext = ".ogg"

        tmp_path = os.path.join(tmp_dir, f"user_audio{ext}")
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        print(f"[語音下載] 已儲存: {tmp_path} ({len(resp.content)} bytes, type={content_type})")
        
        # 轉成 wav 給 ASR 用（飛書 opus 可能不相容 librosa）
        if not tmp_path.endswith(".wav"):
            wav_path = os.path.join(tmp_dir, "user_audio.wav")
            try:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and os.path.exists(wav_path):
                    print(f"[語音下載] ✅ 已轉 wav: {wav_path}")
                    return wav_path
                else:
                    print(f"[語音下載] ⚠️ 轉 wav 失敗，用原始檔: {result.stderr[-100:]}")
            except Exception as e:
                print(f"[語音下載] ⚠️ 轉 wav 異常: {e}")
        
        return tmp_path
    except Exception as e:
        print(f"[語音下載] 失敗: {e}")
        return None


def _convert_to_opus(input_path: str) -> str | None:
    """將音檔轉為飛書語音訊息需要的 opus 格式（支援 wav/m4a/mp3 輸入）"""
    try:
        if input_path.endswith(".opus"):
            return input_path
        opus_path = input_path.rsplit(".", 1)[0] + ".opus"
        # 快取：已轉過且比原檔新就跳過
        if os.path.exists(opus_path) and os.path.getmtime(opus_path) >= os.path.getmtime(input_path):
            return opus_path
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-c:a", "libopus", "-b:a", "32k", "-ar", "16000", "-ac", "1", "-vn", opus_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and os.path.exists(opus_path):
            print(f"[ffmpeg] ✅ 轉檔成功: {os.path.basename(input_path)} → opus ({os.path.getsize(opus_path)} bytes)")
            return opus_path
        print(f"[ffmpeg] 轉檔失敗: {result.stderr[-200:]}")
        return None
    except Exception as e:
        print(f"[ffmpeg] 異常: {e}")
        return None


def upload_audio_to_feishu(audio_path: str) -> str | None:
    """
    上傳音檔到飛書，返回 file_key
    飛書語音訊息只接受 opus 格式，file_type 必須是 "opus"

    Returns:
        file_key 字串，或 None
    """
    try:
        token = _get_tenant_token()
        if not token:
            return None

        # 音檔 → opus 轉檔
        opus_path = _convert_to_opus(audio_path)
        if not opus_path:
            print("[音檔上傳] 音檔→opus 轉檔失敗")
            return None

        filename = os.path.basename(opus_path)
        with open(opus_path, "rb") as f:
            resp = http_requests.post(
                "https://open.feishu.cn/open-apis/im/v1/files",
                headers={"Authorization": f"Bearer {token}"},
                data={"file_type": "opus", "file_name": filename},
                files={"file": (filename, f, "audio/opus")},
                timeout=30,
            )
        data = resp.json()
        if data.get("code") == 0:
            print(f"[音檔上傳] ✅ 成功: {data['data']['file_key']}")
            return data["data"]["file_key"]
        else:
            print(f"[音檔上傳] 失敗: code={data.get('code')} msg={data.get('msg', 'unknown')}")
            return None
    except Exception as e:
        print(f"[音檔上傳] 異常: {e}")
        return None


def send_audio_reply(message_id: str, audio_path: str):
    """回覆語音訊息"""
    try:
        file_key = upload_audio_to_feishu(audio_path)
        if not file_key:
            print("[音檔回覆] 上傳失敗，跳過語音回覆")
            return

        client = create_client()
        request = ReplyMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(ReplyMessageRequestBody.builder()
                          .msg_type("audio")
                          .content(json.dumps({"file_key": file_key}))
                          .build()) \
            .build()
        response = client.im.v1.message.reply(request)
        if response.success():
            print(f"  ✅ 已回覆語音 ({os.path.basename(audio_path)})")
        else:
            print(f"[音檔回覆] 失敗: code={response.code}")
    except Exception as e:
        print(f"[音檔回覆] 異常: {e}")


def recognize_user_audio(audio_path: str) -> dict:
    """
    呼叫 ASR API 辨識用戶語音

    Returns:
        {"text": "...", "confidence": 0.xx} 或 {"text": "", "error": "..."}
    """
    try:
        filename = os.path.basename(audio_path)
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            ".m4a": "audio/mp4", ".wav": "audio/wav", ".webm": "audio/webm",
            ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".opus": "audio/ogg",
        }
        content_type = content_types.get(ext, "audio/wav")

        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, content_type)}
            resp = http_requests.post(ASR_API_URL, files=files, timeout=10)

        if resp.status_code == 200:
            result = resp.json()
            return {
                "text": result.get("corrected", result.get("text", "")),
                "raw_text": result.get("debugInfo", {}).get("raw_text", ""),
                "confidence": result.get("confidence", 0.0),
            }
        else:
            return {"text": "", "error": f"ASR API 回應 {resp.status_code}"}
    except http_requests.exceptions.ConnectionError:
        return {"text": "", "error": "語音辨識服務暫時不可用，請稍後再試"}
    except http_requests.exceptions.Timeout:
        return {"text": "", "error": "語音辨識超時，請再試一次"}
    except Exception as e:
        return {"text": "", "error": f"語音辨識異常: {e}"}


# 高準確率詞彙（ASR 實測通過）
_RELIABLE_WORDS = None

def _get_reliable_words() -> set:
    """載入 ASR 實測高準確率詞彙"""
    global _RELIABLE_WORDS
    if _RELIABLE_WORDS is not None:
        return _RELIABLE_WORDS
    path = PROJECT_ROOT / "data" / "reliable_speak_words.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            _RELIABLE_WORDS = set(w.lower().strip() for w in json.load(f))
        print(f"[口語練習] 高準確率詞彙: {len(_RELIABLE_WORDS)} 個")
    else:
        _RELIABLE_WORDS = set()  # fallback: 不篩選
    return _RELIABLE_WORDS


def _get_speakable_entries() -> list:
    """取得 ASR 高準確率 + 有音檔的語料條目"""
    corpus = get_corpus_data()
    if not corpus:
        return []
    reliable = _get_reliable_words()
    result = []
    for entry in corpus:
        paiwan = entry.get("paiwan", entry.get("排灣語", "")).strip().lower()
        if not paiwan:
            continue
        # 必須有音檔
        if not _get_paiwan_audio_path(paiwan):
            continue
        # 如果有篩選詞庫，必須在其中
        if reliable and paiwan not in reliable:
            continue
        result.append(entry)
    return result


def _get_sentence_entries() -> list:
    """取得句子級音檔的語料條目（跟讀模式用）"""
    reliable = _get_reliable_words()
    entries = []
    # 從真人錄音找句子級的（多詞組合）
    idx = _build_real_audio_index()
    for word, path in idx.items():
        # 句子 = 包含空格的詞組
        if " " not in word:
            continue
        if reliable and word.lower() not in reliable:
            continue
        entries.append({"paiwan": word, "chinese": "跟讀練習"})
    # fallback
    if not entries:
        for entry in _get_speakable_entries():
            paiwan = entry.get("paiwan", entry.get("排灣語", ""))
            if " " in paiwan:
                entries.append(entry)
    return entries


# ============================================
# 口語練習指令處理器
# ============================================

def handle_speak(open_id: str, message_id: str = "") -> str:
    """🎤 口說測驗模式"""
    entries = _get_speakable_entries()
    if not entries:
        return "抱歉，目前沒有可用的口說練習題目 😅"

    entry = random.choice(entries)
    paiwan = entry.get("paiwan", entry.get("排灣語", ""))
    chinese = entry.get("chinese", entry.get("中文", ""))
    audio_path = _get_paiwan_audio_path(paiwan)

    _speak_state[open_id] = {
        "answer": paiwan,
        "answer_chinese": chinese,
        "mode": "speak",
        "attempts": 0,
        "audio_path": audio_path,
    }

    # 先回覆文字提示
    text = (
        f"🎤 口說練習！\n\n"
        f"題目：請用排灣語說出「{chinese}」\n\n"
        f"🔊 先聽聽正確發音 👇"
    )
    if not audio_path:
        text += "\n⚠️ 此詞暫無預錄音檔，請參考拼讀嘗試"
    text += "\n\n準備好了就用語音回覆吧！"

    # 再發送正確發音語音（文字在上，語音在下）
    if audio_path and message_id:
        send_audio_reply(message_id, audio_path)

    return text


def handle_shadow(open_id: str, message_id: str = "") -> str:
    """👂 跟讀模式"""
    entries = _get_sentence_entries()
    if not entries:
        # fallback 到一般詞彙
        entries = _get_speakable_entries()
    if not entries:
        return "抱歉，目前沒有可用的跟讀練習題目 😅"

    entry = random.choice(entries)
    paiwan = entry.get("paiwan", entry.get("排灣語", ""))
    chinese = entry.get("chinese", entry.get("中文", ""))
    audio_path = _get_paiwan_audio_path(paiwan)

    _speak_state[open_id] = {
        "answer": paiwan,
        "answer_chinese": chinese,
        "mode": "shadow",
        "attempts": 0,
        "audio_path": audio_path,
    }

    # 發送範例語音
    if audio_path and message_id:
        send_audio_reply(message_id, audio_path)

    return (
        f"👂 跟讀練習！\n\n"
        f"先仔細聽這句話...\n\n"
        f"🔊 (點擊上方語音仔細聽)\n\n"
        f"「{chinese}」→ {paiwan}\n\n"
        f"然後跟著念一遍，用語音回覆！"
    )


def _handle_speak_audio(open_id: str, message_id: str, file_key: str) -> None:
    """處理 speak/shadow 模式中的語音訊息"""
    state = _speak_state.get(open_id)
    if not state:
        send_reply(message_id, "請先輸入 /speak 或 /shadow 開始練習！")
        return

    answer = state["answer"]
    answer_chinese = state["answer_chinese"]
    mode = state["mode"]
    audio_path = state.get("audio_path")

    # Step 1: 下載用戶語音
    send_reply(message_id, "⏳ 辨識中...")
    user_audio = download_audio_message(message_id, file_key)
    if not user_audio:
        send_reply(message_id, "抱歉，無法下載你的語音，請再試一次 🙏")
        return

    try:
        # Step 2: ASR 辨識
        asr_result = recognize_user_audio(user_audio)

        if asr_result.get("error"):
            send_reply(message_id, f"⚠️ {asr_result['error']}")
            return

        recognized = asr_result.get("text", "").strip().lower()
        if not recognized:
            send_reply(message_id, "聽不清楚，請再錄一次！🎤")
            return

        print(f"[口語練習] ASR結果: '{recognized}', 正確答案: '{answer}'")
        raw_text = asr_result.get('raw_text', '')
        asr_conf = asr_result.get('confidence', 0)
        print(f"[口語練習] ASR原始: raw='{raw_text}', confidence={asr_conf}")

        # Step 3: 比對（模糊匹配）
        correct_norm = answer.strip().lower()
        state["attempts"] += 1

        # 計算相似度
        def _similarity(a: str, b: str) -> float:
            """計算兩個字串的相似度（0~1）"""
            if not a or not b:
                return 0.0
            if a == b:
                return 1.0
            # 包含關係
            if a in b or b in a:
                longer, shorter = max(len(a), len(b)), min(len(a), len(b))
                return shorter / longer
            # 詞級比對
            a_words = set(a.split())
            b_words = set(b.split())
            if a_words and b_words:
                overlap = len(a_words & b_words)
                total = len(a_words | b_words)
                return overlap / total if total > 0 else 0.0
            # 字元級比對
            from difflib import SequenceMatcher
            return SequenceMatcher(None, a, b).ratio()

        sim = _similarity(recognized, correct_norm)
        print(f"[口語練習] 相似度: {sim:.2f} (ASR='{recognized}', 答案='{correct_norm}')")

        if mode == "shadow":
            # 跟讀模式：寬鬆，40% 以上算通過
            is_correct = sim >= 0.4
            is_partial = sim >= 0.2 and not is_correct
        else:
            # 口說模式：55% 以上算正確，30% 以上算接近
            # ASR 是 Whisper-tiny + LoRA，短詞辨識有限，用寬鬆門檻
            is_correct = sim >= 0.55
            is_partial = sim >= 0.25 and not is_correct

        # Step 4: 給評價（顯示 ASR 辨識結果，展示技術）
        asr_display = f"\n🎤 AI 聽到：{recognized}（相似度 {int(sim*100)}%）\n" if recognized else "\n"

        if is_correct:
            del _speak_state[open_id]
            send_reply(
                message_id,
                f"🎉 太棒了！發音非常標準！{asr_display}\n"
                f"🟢 {answer}\n"
                f"🔵 {answer_chinese}\n\n"
                f"要再練習嗎？輸入 /speak 或 /shadow"
            )
        elif mode == "speak" and not is_correct and is_partial:
            send_reply(
                message_id,
                f"👍 很接近了！{asr_display}\n"
                f"正確答案是：{answer}\n"
                f"再聽一次正確發音～"
            )
            if audio_path:
                send_audio_reply(message_id, audio_path)
        else:
            attempts = state["attempts"]
            if attempts >= 3:
                # 三次都沒過，給答案
                del _speak_state[open_id]
                send_reply(
                    message_id,
                    f"💪 沒關係！{asr_display}\n"
                    f"正確答案是：\n\n"
                    f"🟢 {answer}\n"
                    f"🔵 {answer_chinese}\n\n"
                    f"多聽幾次就會了！輸入 /speak 或 /shadow 再練習"
                )
                if audio_path:
                    send_audio_reply(message_id, audio_path)
            else:
                hint = f"（第 {attempts}/3 次嘗試）" if mode == "speak" else ""
                send_reply(
                    message_id,
                    f"💪 再試一次吧！{hint}\n\n"
                    f"提示：正確答案包含「{answer_chinese}」的意思"
                )
                if audio_path:
                    send_audio_reply(message_id, audio_path)

    finally:
        # 清理臨時檔案
        try:
            if user_audio and os.path.exists(user_audio):
                os.remove(user_audio)
                # 嘗試移除臨時目錄
                tmp_dir = os.path.dirname(user_audio)
                if os.path.isdir(tmp_dir):
                    os.rmdir(tmp_dir)
        except Exception:
            pass


# ============================================
# 指令處理器
# ============================================

def handle_help() -> str:
    return (
        "🏔️ 语声同行 2.0_yu — 排灣語 AI 助手\n"
        "我是 vuvu Maliq，你的排灣族祖母語伴！\n\n"
        "📋 指令列表：\n"
        "• /help — 查看幫助\n"
        "• /translate <文字> — 排灣語⇄中文翻譯\n"
        "• /lookup <詞> — 詞彙解析（相關詞+綴詞分析）\n"
        "• /learn — 學一個新詞（智能推薦）\n"
        "• /quiz — 排灣語小測驗\n"
        "• /notebook — 個人單字本\n"
        "• /daily — 每日一句排灣語\n"
        "• /speak — 🎤 口說練習（用語音回覆！）\n"
        "• /shadow — 👂 跟讀練習（模仿發音）\n"
        "• /submit <排灣語> <中文> — 提交新詞彙到社群語料庫\n"
        "• /stats — 📊 學習報告 + 熱力圖\n\n"
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

        # === 主動學習：評估置信度 ===
        learner = get_active_learner()
        conf_eval = learner.evaluate_confidence(result)
        
        # 記錄到學習日誌
        learner.log_interaction(
            user_input=query,
            translation_result=result,
            confidence_eval=conf_eval,
            source="飛書對話",
        )

        # 寫入 Bitable（帶置信度等級）
        try:
            from bitable_writer import write_transcription
            write_transcription(
                asr_text=query,
                chinese_translation=translation,
                confidence=conf_eval["confidence"],
                source="飛書對話",
                confidence_level=conf_eval["level"],
                method=result.get("method", "unknown"),
                needs_confirmation=conf_eval["needs_confirmation"],
            )
        except Exception as e:
            print(f"  ⚠️ Bitable 寫入失敗: {e}")

        add_to_history(open_id, "user", f"[翻譯] {query}")
        add_to_history(open_id, "assistant", translation)

        # 如果需要確認，保存狀態並提示用戶
        base_reply = f"🔄 翻譯結果：\n\n{query}\n  ↓\n{translation}"
        
        if conf_eval["needs_confirmation"]:
            confirm_prompt = learner.generate_confirmation_prompt(
                query, translation, conf_eval["level"]
            )
            _confirmation_state[open_id] = {
                "input": query,
                "translation": translation,
                "level": conf_eval["level"],
                "method": result.get("method", "unknown"),
            }
            base_reply += f"\n\n---\n{confirm_prompt}"
        else:
            # 高置信度，加個小標記
            base_reply += "\n\n🟢 (高置信度翻譯)"

        # === 熱力圖追蹤 ===
        try:
            from bitable_dashboard import record_heatmap_event, periodic_flush
            record_heatmap_event(open_id, "translate", query)
            periodic_flush(open_id)
        except Exception:
            pass

        return base_reply
    except Exception as e:
        return f"翻譯服務暫時不可用: {e}"


def handle_lookup(text: str) -> str:
    """詞彙解析：相關詞 + 綴詞分析 + 親屬關係"""
    word = text.strip()
    if not word:
        return "請輸入要查詢的詞，例如：\n/lookup masalu\n/lookup 母親"
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from knowledge_graph import PaiwanKnowledgeGraph
        kg = PaiwanKnowledgeGraph()
        result = kg.lookup(word)
        reply = kg.format_lookup_reply(result)
        if reply:
            return reply
        else:
            return f"找不到「{word}」的相關資訊。\n試試 /translate {word} 來翻譯看看！"
    except Exception as e:
        return f"詞彙查詢服務暫時不可用: {e}"


def handle_learn(open_id: str) -> str:
    profile = get_user_profile()
    corpus = get_corpus_data()

    # 優先使用智能推薦（基於已學詞彙推薦相關詞）
    suggestion = profile.suggest_next(open_id, corpus)
    if suggestion:
        paiwan = suggestion["paiwan"]
        chinese = suggestion["chinese"]
    elif corpus:
        entry = random.choice(corpus)
        paiwan = entry.get("paiwan", entry.get("排灣語", ""))
        chinese = entry.get("chinese", entry.get("中文", ""))
    else:
        return "語料庫暫時無法使用"

    if not paiwan or not chinese:
        return "語料格式異常，請稍後再試"

    profile.record_learn(open_id, paiwan)
    add_to_history(open_id, "assistant", f"[教學] {paiwan} = {chinese}")

    # === 熱力圖追蹤 ===
    try:
        from bitable_dashboard import record_heatmap_event, get_related_recommendation, update_word_stats, increment_word_learners
        record_heatmap_event(open_id, "learn", paiwan)
        update_word_stats(paiwan, chinese, True, suggestion.get("theme", "") if suggestion else "")
        increment_word_learners(paiwan)
    except Exception:
        pass

    msg = f"📚 學新詞：\n\n🟢 {paiwan}\n🔵 {chinese}\n\n試著跟我說說看！💬"

    # === 對話驅動推薦 ===
    try:
        learned_set = set(profile.get_learned_words(open_id))
        rec_msg = get_related_recommendation(paiwan, learned_set)
        if rec_msg:
            msg += f"\n\n{rec_msg}"
    except Exception:
        pass

    return msg


def handle_notebook(open_id: str) -> str:
    """個人單字本"""
    profile = get_user_profile()
    corpus = get_corpus_data()
    return profile.format_notebook(open_id, corpus)


def handle_submit(text: str, open_id: str) -> str:
    """提交新詞彙到社群語料審核表"""
    parts = text.strip().split()
    if len(parts) < 2:
        return (
            "📝 提交新詞彙到社群語料庫！\n\n"
            "用法：/submit <排灣語> <中文釋義>\n"
            "例如：/submit tjengelay 愛\n\n"
            "提交後會由母語者審核，通過後自動加入詞彙表！"
        )

    paiwan = parts[0].strip()
    chinese = " ".join(parts[1:]).strip()

    if not paiwan or not chinese:
        return "排灣語和中文釋義都不能為空哦！"

    try:
        from bitable_dashboard import submit_community_word
        result = submit_community_word(paiwan, chinese, open_id[:8], "Bot提交")
        return result["message"]
    except Exception as e:
        return f"提交失敗: {e}"


def handle_stats(open_id: str) -> str:
    """個人學習統計 + 熱力圖"""
    profile = get_user_profile()
    stats = profile.get_stats(open_id)

    try:
        from bitable_dashboard import format_stats_reply, periodic_flush
        periodic_flush(open_id)
        return format_stats_reply(open_id, stats)
    except Exception as e:
        # Fallback：如果儀表板模組載入失敗，用簡單版本
        lines = [
            "📊 你的排灣語學習報告",
            "",
            f"📚 詞彙量：{stats.get('total_learned', 0)} 詞",
            f"✅ 答對：{stats.get('total_correct', 0)} 次",
            f"❌ 答錯：{stats.get('total_wrong', 0)} 次",
            f"🎯 正確率：{stats.get('accuracy', 0)}%",
            f"🔥 連續學習：{stats.get('streak', 0)} 天",
        ]
        return "\n".join(lines)


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
    # 檢查是否有待確認的翻譯
    if open_id in _confirmation_state:
        if text.strip() in ["✅", "正確", "對", "yes", "y", "是"]:
            state = _confirmation_state.pop(open_id)
            learner = get_active_learner()
            reply = learner.process_feedback(state["input"], "✅")
            return reply
        elif text.strip() in ["❌", "不正確", "錯", "no", "n", "否"]:
            state = _confirmation_state.pop(open_id)
            learner = get_active_learner()
            reply = learner.process_feedback(state["input"], "❌")
            return reply
        # 如果用戶說了別的（不是確認反饋），清除確認狀態，繼續正常對話
        _confirmation_state.pop(open_id, None)

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

        # === 熱力圖追蹤 ===
        profile = get_user_profile()
        is_correct = (user_answer == correct)
        is_close = (user_answer in correct or correct in user_answer)

        try:
            from bitable_dashboard import record_heatmap_event, update_word_stats, get_related_recommendation
            if is_correct:
                record_heatmap_event(open_id, "quiz_correct", answer)
                update_word_stats(answer, chinese, True)
            elif is_close:
                record_heatmap_event(open_id, "quiz_correct", answer)
                update_word_stats(answer, chinese, True)
            else:
                record_heatmap_event(open_id, "quiz_wrong", answer)
                update_word_stats(answer, chinese, False)
        except Exception:
            pass

        if is_correct:
            profile.record_correct(open_id, answer)
            reply = f"🎉 完全正確！太棒了！\n\n🟢 {answer} = {chinese}\n\n要再試一題嗎？輸入 /quiz"
        elif is_close:
            profile.record_correct(open_id, answer)
            reply = f"👍 很接近了！正確答案是：\n\n🟢 {answer}\n🔵 {chinese}\n\n輸入 /quiz 再挑戰！"
        else:
            profile.record_wrong(open_id, answer)
            reply = f"答案不一樣哦～正確的是：\n\n🟢 {answer}\n🔵 {chinese}\n\n輸入 /quiz 再試一題！"

        # 推薦相關詞
        try:
            learned_set = set(profile.get_learned_words(open_id))
            rec_msg = get_related_recommendation(answer, learned_set)
            if rec_msg and not is_correct:
                reply += f"\n\n{rec_msg}"
        except Exception:
            pass

        return reply

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

def route_message(text: str, open_id: str, message_id: str = "") -> str | None:
    if text.startswith("/help") or text.startswith("/幫助"):
        return handle_help()
    elif text.startswith("/translate ") or text.startswith("/翻譯 "):
        query = text.split(" ", 1)[1] if " " in text else ""
        return handle_translate(query, open_id)
    elif text.startswith("/lookup ") or text.startswith("/查詞 "):
        query = text.split(" ", 1)[1] if " " in text else ""
        return handle_lookup(query)
    elif text.startswith("/learn") or text.startswith("/學習"):
        return handle_learn(open_id)
    elif text.startswith("/quiz") or text.startswith("/測驗"):
        return handle_quiz(open_id)
    elif text.startswith("/notebook") or text.startswith("/單字本"):
        return handle_notebook(open_id)
    elif text.startswith("/daily") or text.startswith("/每日"):
        return handle_daily()
    elif text.startswith("/submit ") or text.startswith("/提交 "):
        query = text.split(" ", 1)[1] if " " in text else ""
        return handle_submit(query, open_id)
    elif text.startswith("/stats") or text.startswith("/統計"):
        return handle_stats(open_id)
    elif text.startswith("/speak") or text.startswith("/口說"):
        return handle_speak(open_id, message_id)
    elif text.startswith("/shadow") or text.startswith("/跟讀"):
        return handle_shadow(open_id, message_id)
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

            reply = route_message(text, open_id, message_id)
            if reply:
                send_reply(message_id, reply)
            
            # 寫入飛書多維表格（語料收集 — 主動學習 + 統計追蹤）
            try:
                from bitable_writer import write_transcription, stats_tracker
                learner = get_active_learner()
                
                if text.startswith("/translate") or text.startswith("/翻譯"):
                    # 翻譯指令：已在 handle_translate 裡處理 Bitable 寫入
                    pass
                else:
                    # 一般對話：只記錄用戶輸入（低置信度，需人工確認）
                    write_transcription(
                        asr_text=text,
                        chinese_translation="",
                        confidence=0.0,
                        source="飛書對話",
                        confidence_level="low",
                        method="chat",
                        needs_confirmation=True,
                    )
                    learner.log_interaction(
                        user_input=text,
                        translation_result={"method": "chat", "translation": ""},
                        confidence_eval={"confidence": 0.0, "level": "low", "needs_confirmation": True, "needs_human_review": True, "auto_accept": False},
                        source="飛書對話",
                    )
                    print(f"  ✅ 已寫入 Bitable (一般對話)")
            except Exception as e:
                print(f"  ⚠️ Bitable 寫入失敗: {e}")

            # 定期 flush 統計到 Bitable
            try:
                from bitable_writer import stats_tracker
                stats_tracker.flush_to_bitable()
            except Exception as e:
                print(f"  ⚠️ 統計 flush 失敗: {e}")

        elif msg_type == "audio":
            content_str = message.content if message else "{}"
            content = json.loads(content_str) if isinstance(content_str, str) else content_str
            file_key = content.get("file_key", "")
            print(f"  語音: file_key={file_key[:20]}...")

            # 如果在 speak/shadow 模式中，處理語音辨識
            if open_id in _speak_state:
                _handle_speak_audio(open_id, message_id, file_key)
            else:
                send_reply(message_id, "🎤 語音功能即將上線！目前請先用文字跟我聊天～\n💡 想練習口說？試試 /speak 或 /shadow")

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
    print("  🏔️ 语声同行 2.0_yu — 飛書機器人 (WebSocket)")
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
