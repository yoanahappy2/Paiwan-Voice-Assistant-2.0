#!/usr/bin/env python3
"""
test_bot_all.py — 語聲同行 2.0 全指令自動化測試
直接呼叫 Bot 的 route_message 函數，無需啟動飛書連線
"""

import sys
import json
import time
import traceback
from pathlib import Path

# 確保能 import bot 模組
sys.path.insert(0, str(Path(__file__).parent))

# ============================================
# 測試框架
# ============================================

results = []
_total = 0
_passed = 0
_failed = 0


def test(name: str, func):
    """跑一個測試"""
    global _total, _passed, _failed
    _total += 1
    result = None
    try:
        result = func()
        if result is None or result is False:
            raise AssertionError("返回 None/False")
        _passed += 1
        status = "✅ PASS"
        detail = str(result)[:200]
    except Exception as e:
        _failed += 1
        status = "❌ FAIL"
        detail = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    results.append({"name": name, "status": status, "detail": detail})
    print(f"  {status} | {name}")
    if len(detail) > 200:
        detail = detail[:200] + "..."
    print(f"         {detail}")
    return result


def report():
    """輸出測試報告"""
    print("\n" + "=" * 60)
    print(f"  測試報告：{_passed}/{_total} 通過，{_failed} 失敗")
    print("=" * 60)
    for r in results:
        print(f"  {r['status']} | {r['name']}")
    print()

    # 輸出 JSON 報告
    report_path = Path(__file__).parent / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": _total,
            "passed": _passed,
            "failed": _failed,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)
    print(f"  報告已儲存：{report_path}")
    return _failed == 0


# ============================================
# Import Bot 模組
# ============================================

print("=" * 60)
print("  語聲同行 2.0 — 全指令自動化測試")
print("=" * 60)
print()

print("📦 載入模組...")

# 先載入翻譯服務（測試核心）
from translate_service import PaiwanTranslator
translator = PaiwanTranslator()
translator.load()

# 載入 Bot 的 route_message
from feishu_bot.bot_ws import route_message

# 載入其他服務
try:
    from knowledge_graph import PaiwanKnowledgeGraph
    kg = PaiwanKnowledgeGraph()
except Exception as e:
    print(f"  ⚠️ 知識圖譜載入失敗: {e}")
    kg = None

TEST_OPEN_ID = "test_automated_001"
TEST_MESSAGE_ID = "test_msg_001"

print("✅ 模組載入完成")
print()


# ============================================
# 1. 翻譯服務測試（translate_service.py）
# ============================================

print("━━━ 1. 翻譯服務（translate_service.py）━━━")

# 精確匹配測試
exact_tests = [
    ("masalu", "謝謝", "p2c"),
    ("kina", "媽媽", "p2c"),
    ("kama", "爸爸", "p2c"),
    ("vuvu", "祖", "p2c"),  # 精確詞典裡是「祖；孫」
    ("zaljum", "水", "p2c"),
    ("lima", "五", "p2c"),
    ("tjengelay", "愛", "rag_or_exact"),  # 可能不在精確詞典
    ("qadaw", "太陽", "p2c"),
]

for paiwan, expected, direction in exact_tests:
    def _t(p=paiwan, e=expected, d=direction):
        r = translator.translate(p)
        if d == "rag_or_exact":
            assert r["method"] in ("exact", "rag_llm"), f"方法異常: {r['method']}"
        else:
            assert r["method"] == "exact", f"期望 exact 但得到 {r['method']}"
        assert e in r["translation"], f"期望包含「{e}」但得到「{r['translation']}」"
        return f"{p} → {r['translation']} ({r['method']})"
    test(f"精確翻譯: {paiwan} → {expected}", _t)

# 中文→排灣語
c2p_tests = [
    ("謝謝", "masalu"),
    ("媽媽", "kina"),
    ("爸爸", "kama"),
    ("水", "zaljum"),
    ("你好", None),  # 可能有多種翻譯，只測不 crash
]

for chinese, expected_paiwan in c2p_tests:
    def _t(c=chinese, ep=expected_paiwan):
        r = translator.translate(c)
        assert r["direction"] == "c2p", f"期望 c2p 但得到 {r['direction']}"
        if ep:
            assert ep in r["translation"].lower(), f"期望包含「{ep}」但得到「{r['translation']}」"
        return f"{c} → {r['translation']} ({r['method']})"
    test(f"中→排翻譯: {chinese}", _t)

# RAG+LLM 翻譯（非精確匹配）
rag_tests = [
    "我想吃飯",
    "今天天氣很好",
    "你好嗎",
]

for text in rag_tests:
    def _t(t=text):
        r = translator.translate(t)
        assert r["method"] in ("rag_llm", "exact"), f"方法: {r['method']}"
        assert r["translation"], "翻譯結果為空"
        return f"{t} → {r['translation']} ({r['method']})"
    test(f"RAG翻譯: {text}", _t)


# ============================================
# 2. Bot 指令測試（route_message）
# ============================================

print("\n━━━ 2. Bot 指令（route_message）━━━")

# /help
test("/help", lambda: route_message("/help", TEST_OPEN_ID, TEST_MESSAGE_ID))

# /translate
def _translate_test():
    r = route_message("/translate masalu", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert "謝謝" in r, f"翻譯結果缺少「謝謝」: {r[:100]}"
    return r[:150]
test("/translate masalu", _translate_test)

def _translate_kina():
    r = route_message("/translate kina", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert "媽媽" in r, f"翻譯結果缺少「媽媽」: {r[:100]}"
    return r[:150]
test("/translate kina（精確匹配修復）", _translate_kina)

def _translate_c2p():
    r = route_message("/translate 謝謝", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert "masalu" in r.lower(), f"翻譯結果缺少 masalu: {r[:100]}"
    return r[:150]
test("/translate 謝謝（中→排）", _translate_c2p)

# /learn
def _learn_test():
    r = route_message("/learn", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert "學新詞" in r, f"回覆缺少「學新詞」: {r[:100]}"
    return r[:150]
test("/learn", _learn_test)

# /daily
def _daily_test():
    r = route_message("/daily", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert r and len(r) > 10, f"回覆太短: {r}"
    return r[:150]
test("/daily", _daily_test)

# /quiz
def _quiz_test():
    r = route_message("/quiz", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert "測驗" in r, f"回覆缺少「測驗」: {r[:100]}"
    return r[:150]
test("/quiz", _quiz_test)

# /lookup
def _lookup_test():
    r = route_message("/lookup masalu", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert r and len(r) > 5, f"回覆太短: {r}"
    return r[:150]
test("/lookup masalu", _lookup_test)

# /notebook
def _notebook_test():
    r = route_message("/notebook", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert r and len(r) > 5, f"回覆太短: {r}"
    return r[:150]
test("/notebook", _notebook_test)

# /speak（不帶參數，隨機出題）
def _speak_random():
    r = route_message("/speak", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert "口說" in r, f"回覆缺少「口說」: {r[:100]}"
    return r[:150]
test("/speak（隨機出題）", _speak_random)

# /speak lima（指定詞）
def _speak_word():
    r = route_message("/speak lima", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert "lima" in r.lower() or "五" in r, f"回覆缺少 lima/五: {r[:200]}"
    return r[:200]
test("/speak lima（指定詞）", _speak_word)

# /shadow
def _shadow_test():
    r = route_message("/shadow", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert "跟讀" in r, f"回覆缺少「跟讀」: {r[:100]}"
    return r[:150]
test("/shadow", _shadow_test)

# /stats
def _stats_test():
    r = route_message("/stats", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert r and len(r) > 10, f"回覆太短: {r}"
    return r[:150]
test("/stats", _stats_test)

# /submit（格式錯誤）
def _submit_noarg():
    r = route_message("/submit", TEST_OPEN_ID, TEST_MESSAGE_ID)
    assert "提交" in r or "用法" in r, f"回覆缺少提示: {r[:100]}"
    return r[:150]
test("/submit（無參數）", _submit_noarg)


# ============================================
# 3. 批量詞彙翻譯準確率
# ============================================

print("\n━━━ 3. 批量詞彙翻譯準確率 ━━━")

# 核心詞彙測試集
core_vocab = [
    ("masalu", "謝謝"),
    ("kina", "媽媽"),
    ("kama", "爸爸"),
    ("vuvu", "祖"),
    ("zaljum", "水"),
    ("lima", "五"),
    ("tjengelay", "愛"),  # 可能不在精確詞典
    ("qadaw", "太陽"),
    ("tjelu", "三"),
    ("drusa", "二"),
    ("itua", "一"),
    ("sepacu", "七"),
    ("tapuluq", "十"),
    ("limeqa", "六"),
    ("siva", "九"),
    ("spat", "四"),
    ("tapuluq sa alu", "十八"),
    ("cavilj", "年"),
    ("gakku", "學校"),
    ("umaq", "家"),
    ("zaljum", "水"),
    ("ljamek", "忘記"),
    ("pacunan", "看"),
    ("keljang", "知道"),
    ("tjemeljes", "漂亮"),
    ("tarivak", "好"),
    ("na", "的"),
    ("aken", "我"),
    ("sun", "你"),
    ("madjulu", "會"),
]

correct = 0
wrong_list = []
for paiwan, expected in core_vocab:
    r = translator.translate(paiwan)
    translation = r["translation"]
    # 檢查期望詞是否在翻譯結果中
    if expected in translation:
        correct += 1
    else:
        wrong_list.append((paiwan, expected, translation))

accuracy = correct / len(core_vocab) * 100

test(f"批量翻譯準確率: {correct}/{len(core_vocab)} = {accuracy:.1f}%",
     lambda: f"正確 {correct}, 錯誤 {len(wrong_list)}" + 
             (f"\n         錯誤: {wrong_list[:5]}" if wrong_list else ""))

# 列出所有錯誤
if wrong_list:
    print("\n  ⚠️ 翻譯錯誤詳情：")
    for p, e, t in wrong_list:
        print(f"    {p} → 期望「{e}」實際「{t}」")


# ============================================
# 4. ASR 服務測試
# ============================================

print("\n━━━ 4. ASR 服務 ━━━")

def _asr_health():
    import requests
    r = requests.get("http://127.0.0.1:8000/health", timeout=5)
    data = r.json()
    assert data.get("status") == "ok", f"ASR 狀態異常: {data}"
    assert "small" in data.get("model", ""), f"模型不是 small: {data.get('model')}"
    return f"model={data.get('model')}, status=ok"
test("ASR 服務健康檢查", _asr_health)


# ============================================
# 報告
# ============================================

print()
success = report()
sys.exit(0 if success else 1)
