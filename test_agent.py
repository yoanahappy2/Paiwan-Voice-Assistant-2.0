"""
test_agent.py — 自動化 Agent 測試

一次跑完所有場景，輸出結構化報告。

用法：
    python3 test_agent.py

作者: 地陪
日期: 2026-05-11
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

from agent_core import PaiwanAgentSystem

# ============================================
# 測試場景定義
# ============================================

TEST_CASES = [
    # --- 翻譯類 ---
    {
        "id": "T01",
        "category": "翻譯",
        "input": "masalu 是什麼意思",
        "expected_tools": ["translate", "rag_search"],
        "expected_keywords": ["謝謝", "masalu"],
        "should_find": True,
    },
    {
        "id": "T02",
        "category": "翻譯",
        "input": "你好嗎用排灣語怎麼說",
        "expected_tools": ["translate", "rag_search"],
        "expected_keywords": [],
        "should_find": True,
    },
    {
        "id": "T03",
        "category": "翻譯",
        "input": "我想你，排灣族語怎麼說",
        "expected_tools": ["translate", "rag_search"],
        "expected_keywords": [],
        "should_find": False,  # 語料庫裡沒有
        "expected_behavior": "誠實說找不到或還沒學到",
    },
    {
        "id": "T04",
        "category": "翻譯",
        "input": "排灣語的謝謝和對不起怎麼說",
        "expected_tools": ["translate", "rag_search"],
        "expected_keywords": ["masalu"],
        "should_find": True,
    },

    # --- 知識查詢 ---
    {
        "id": "K01",
        "category": "知識查詢",
        "input": "kina 是什麼意思",
        "expected_tools": ["knowledge_lookup", "rag_search", "translate"],
        "expected_keywords": ["媽", "母親", "kina"],
        "should_find": True,
    },
    {
        "id": "K02",
        "category": "知識查詢",
        "input": "排灣語的前綴有什麼",
        "expected_tools": ["knowledge_lookup", "rag_search"],
        "expected_keywords": ["前綴", "綴詞"],
        "should_find": True,
    },
    {
        "id": "K03",
        "category": "知識查詢",
        "input": "排灣語的家族稱謂有哪些",
        "expected_tools": ["knowledge_lookup", "rag_search"],
        "expected_keywords": [],
        "should_find": True,
    },

    # --- 學習模式 ---
    {
        "id": "L01",
        "category": "學習",
        "input": "教我一個排灣語",
        "expected_tools": ["suggest_next_word", "rag_search"],
        "expected_keywords": [],
        "should_find": True,
    },
    {
        "id": "L02",
        "category": "學習",
        "input": "我想學排灣語問候",
        "expected_tools": ["rag_search", "knowledge_lookup"],
        "expected_keywords": [],
        "should_find": True,
    },
    {
        "id": "L03",
        "category": "學習",
        "input": "我想從零開始學排灣語",
        "expected_tools": ["suggest_next_word", "rag_search", "query_user_progress"],
        "expected_keywords": [],
        "should_find": True,
    },

    # --- 進度查詢 ---
    {
        "id": "P01",
        "category": "進度",
        "input": "我學了什麼",
        "expected_tools": ["query_user_progress"],
        "expected_keywords": ["學習", "進度"],
        "should_find": True,
    },

    # --- 對話 / 社交 ---
    {
        "id": "C01",
        "category": "對話",
        "input": "你好",
        "expected_tools": ["rag_search"],
        "expected_keywords": [],
        "should_find": True,
    },
    {
        "id": "C02",
        "category": "對話",
        "input": "vuvu 說個故事",
        "expected_tools": ["rag_search"],
        "expected_keywords": [],
        "should_find": True,
    },
    {
        "id": "C03",
        "category": "對話",
        "input": "今天天氣好好喔",
        "expected_tools": [],
        "expected_keywords": [],
        "should_find": True,
        "note": "閒聊，不一定需要工具",
    },
    {
        "id": "C04",
        "category": "對話",
        "input": "排灣族有什麼文化",
        "expected_tools": ["rag_search"],
        "expected_keywords": [],
        "should_find": True,
    },

    # --- 邊界案例 ---
    {
        "id": "E01",
        "category": "邊界",
        "input": "",
        "expected_tools": [],
        "expected_keywords": [],
        "should_find": True,
        "note": "空輸入",
    },
    {
        "id": "E02",
        "category": "邊界",
        "input": "asdfghjkl",
        "expected_tools": [],
        "expected_keywords": [],
        "should_find": True,
        "note": "亂碼",
    },
    {
        "id": "E03",
        "category": "邊界",
        "input": "我想學量子力學",
        "expected_tools": [],
        "expected_keywords": [],
        "should_find": True,
        "note": "無關領域",
    },
    {
        "id": "E04",
        "category": "邊界",
        "input": "你好" * 50,
        "expected_tools": [],
        "expected_keywords": [],
        "should_find": True,
        "note": "超長輸入",
    },
]


# ============================================
# 評分邏輯
# ============================================

def score_result(test_case: dict, result: dict) -> dict:
    """評分單個測試結果"""
    reply = result["reply"]
    trace = result["trace"]
    tools = [a["tool"] for a in trace.actions]
    
    scores = {}
    issues = []
    
    # 1. 是否 crash（有回覆）
    scores["no_crash"] = len(reply) > 0
    if not scores["no_crash"]:
        issues.append("❌ 沒有回覆")
    
    # 2. 回覆長度合理（10-500 字）
    reply_len = len(reply)
    scores["reply_length_ok"] = 10 <= reply_len <= 500
    if reply_len < 10:
        issues.append(f"⚠️ 回覆太短（{reply_len}字）")
    elif reply_len > 500:
        issues.append(f"⚠️ 回覆太長（{reply_len}字）")
    
    # 3. 工具調用次數合理（≤ 3）
    tool_count = len(tools)
    scores["tool_count_ok"] = tool_count <= 3
    if tool_count > 3:
        issues.append(f"⚠️ 調用太多工具（{tool_count}個）")
    
    # 4. 沒有重複調用同一個工具
    scores["no_duplicate_tools"] = len(tools) == len(set(tools))
    if not scores["no_duplicate_tools"]:
        from collections import Counter
        dupes = [t for t, c in Counter(tools).items() if c > 1]
        issues.append(f"⚠️ 重複調用工具: {dupes}")
    
    # 5. 關鍵詞匹配（如果有定義）
    if test_case.get("expected_keywords"):
        found = any(kw in reply for kw in test_case["expected_keywords"])
        scores["keywords_found"] = found
        if not found:
            issues.append(f"⚠️ 缺少關鍵詞: {test_case['expected_keywords']}")
    else:
        scores["keywords_found"] = True  # 沒有要求就 pass
    
    # 6. 「找不到」的行為是否正確
    if not test_case.get("should_find", True):
        good_response = any(
            phrase in reply
            for phrase in ["沒有", "還沒學", "找不到", "目前沒有", "不知道"]
        )
        scores["honest_not_found"] = good_response
        if not good_response:
            issues.append("⚠️ 語料庫裡沒有但沒有誠實說")
    else:
        scores["honest_not_found"] = True
    
    # 7. 耗時合理（< 10 秒）
    scores["time_ok"] = trace.elapsed_ms < 10000
    if trace.elapsed_ms >= 10000:
        issues.append(f"⚠️ 耗時太長（{trace.elapsed_ms}ms）")
    
    # 總分
    total = sum(1 for v in scores.values() if v)
    max_total = len(scores)
    
    return {
        "scores": scores,
        "total": total,
        "max": max_total,
        "issues": issues,
        "tools": tools,
        "tool_count": tool_count,
        "reply_len": reply_len,
        "elapsed_ms": trace.elapsed_ms,
    }


# ============================================
# 主測試流程
# ============================================

def run_all_tests():
    system = PaiwanAgentSystem()
    
    results = []
    pass_count = 0
    fail_count = 0
    
    print("=" * 70)
    print("  🧪 語聲同行 2.0 — Agent 自動化測試")
    print(f"  時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  測試場景: {len(TEST_CASES)} 個")
    print("=" * 70)
    print()
    
    for i, tc in enumerate(TEST_CASES):
        test_id = tc["id"]
        category = tc["category"]
        user_input = tc["input"]
        note = tc.get("note", "")
        
        # 顯示輸入（截斷超長的）
        display_input = user_input[:40] + "..." if len(user_input) > 40 else user_input
        if not display_input:
            display_input = "（空輸入）"
        
        print(f"[{i+1}/{len(TEST_CASES)}] {test_id} [{category}] 「{display_input}」", end=" ... ", flush=True)
        
        # 跳過空輸入（Agent 可能不接受）
        if not user_input.strip():
            print("⏭️ 跳過（空輸入）")
            results.append({
                "id": test_id,
                "status": "skip",
                "reason": "空輸入",
            })
            continue
        
        # 跳過超長輸入
        if len(user_input) > 200:
            print("⏭️ 跳過（超長輸入）")
            results.append({
                "id": test_id,
                "status": "skip",
                "reason": "超長輸入",
            })
            continue
        
        try:
            result = system.chat(user_input, user_id="test_auto")
            scoring = score_result(tc, result)
            
            passed = scoring["total"] >= scoring["max"] - 1  # 允許扣 1 分
            
            if passed:
                pass_count += 1
                status = "✅ PASS"
            else:
                fail_count += 1
                status = "❌ FAIL"
            
            print(f"{status} ({scoring['total']}/{scoring['max']}) [{scoring['elapsed_ms']}ms | {scoring['tool_count']} 工具]")
            
            if scoring["issues"]:
                for issue in scoring["issues"]:
                    print(f"         {issue}")
            
            results.append({
                "id": test_id,
                "category": category,
                "input": display_input,
                "status": "pass" if passed else "fail",
                "reply_preview": result["reply"][:150],
                "tools": scoring["tools"],
                "tool_count": scoring["tool_count"],
                "elapsed_ms": scoring["elapsed_ms"],
                "scores": scoring["scores"],
                "issues": scoring["issues"],
            })
            
        except Exception as e:
            fail_count += 1
            print(f"💥 ERROR: {e}")
            results.append({
                "id": test_id,
                "category": category,
                "input": display_input,
                "status": "error",
                "error": str(e),
            })
        
        # 避免觸發 API 限流
        time.sleep(1)
    
    # ============================================
    # 輸出報告
    # ============================================
    
    total_run = pass_count + fail_count
    pass_rate = (pass_count / total_run * 100) if total_run > 0 else 0
    
    print()
    print("=" * 70)
    print("  📊 測試報告")
    print("=" * 70)
    print()
    
    # 總覽
    print(f"  總場景: {len(TEST_CASES)} | 執行: {total_run} | 通過: {pass_count} | 失敗: {fail_count}")
    print(f"  通過率: {pass_rate:.0f}%")
    print()
    
    # 按類別統計
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"pass": 0, "fail": 0, "error": 0, "skip": 0}
        status = r.get("status", "error")
        categories[cat][status] += 1
    
    print("  按類別:")
    for cat, stats in categories.items():
        total_cat = sum(stats.values())
        passed_cat = stats.get("pass", 0)
        rate = (passed_cat / total_cat * 100) if total_cat > 0 else 0
        print(f"    {cat:6s}: {passed_cat}/{total_cat} ({rate:.0f}%)")
    print()
    
    # 工具使用統計
    all_tools = []
    for r in results:
        all_tools.extend(r.get("tools", []))
    
    if all_tools:
        from collections import Counter
        tool_counts = Counter(all_tools)
        print("  工具使用頻率:")
        for tool, count in tool_counts.most_common():
            print(f"    {tool}: {count} 次")
        print()
    
    # 平均耗時
    elapsed_list = [r["elapsed_ms"] for r in results if "elapsed_ms" in r]
    if elapsed_list:
        avg_ms = sum(elapsed_list) / len(elapsed_list)
        max_ms = max(elapsed_list)
        min_ms = min(elapsed_list)
        print(f"  耗時: 平均 {avg_ms:.0f}ms | 最快 {min_ms}ms | 最慢 {max_ms}ms")
    
    # 失敗案例
    failed = [r for r in results if r.get("status") in ("fail", "error")]
    if failed:
        print()
        print("  ❌ 失敗案例:")
        for r in failed:
            print(f"    {r['id']} [{r.get('category', '?')}] 「{r.get('input', '?')}」")
            if r.get("issues"):
                for issue in r["issues"]:
                    print(f"      {issue}")
            if r.get("error"):
                print(f"      Error: {r['error']}")
    
    print()
    print("=" * 70)
    
    # 存 JSON 報告
    report_path = Path(__file__).parent / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(TEST_CASES),
                "run": total_run,
                "passed": pass_count,
                "failed": fail_count,
                "pass_rate": pass_rate,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"  📁 詳細報告: {report_path}")
    print()


if __name__ == "__main__":
    run_all_tests()
