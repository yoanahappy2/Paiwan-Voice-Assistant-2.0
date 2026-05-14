"""
quality_agent.py — 品質控制專員

負責所有「品質相關」的任務：
- 翻譯品質評估（交叉驗證 Knowledge Agent 的翻譯）
- 語料品質檢查
- 回覆品質評估
- Self-Judge 判斷（目標是否達成）

這是 Multi-Agent 系統的「自我監督」層：
其他 Agent 的輸出可以交給 Quality Agent 審核，
不通過則打回重做（Self-Judge 循環）。

作者: 地陪
日期: 2026-05-12
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.agent import BaseAgent, AgentTrace
from core.message import AgentMessage, MessageType
from core.rate_limiter import APIGuard

logger = logging.getLogger(__name__)


class QualityAgent(BaseAgent):
    """
    品質控制專員

    支援的任務：
    - "review_translation"  審核翻譯品質
    - "review_corpus"       審核語料品質
    - "review_reply"        審核回覆品質（是否偏題、是否編造）
    - "self_judge"          Self-Judge：目標是否達成
    - "cross_validate"      交叉驗證（獨立翻譯一次，比對結果）
    """

    role = "quality"

    def __init__(self, client: OpenAI = None, api_guard: APIGuard = None,
                 model: str = None, config_path: Path = None,
                 project_root: Path = None):
        super().__init__(client=client, api_guard=api_guard,
                         model=model, config_path=config_path)
        self.project_root = project_root or Path(__file__).parent.parent.parent

    def get_system_prompt(self) -> str:
        return (
            "你是排灣族語品質控制專員（Quality Agent）。\n\n"
            "職責：\n"
            "1. 審核翻譯的準確性和自然度\n"
            "2. 檢查語料品質\n"
            "3. 驗證回覆是否偏題或編造\n"
            "4. 判斷目標是否達成\n\n"
            "品質標準：\n"
            "- 排灣語拼寫符合東排灣方言標準\n"
            "- 中文翻譯準確、自然\n"
            "- 不編造語料庫中沒有的排灣語\n"
            "- 回覆簡潔、不偏題\n\n"
            "輸出格式：JSON，包含 passed (bool) + score (0-1) + feedback\n"
        )

    def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        task = message.payload.get("task", "")
        params = message.payload.get("params", {})
        logger.info(f"[QualityAgent] 收到任務: {task}")

        handlers = {
            "review_translation": self._handle_review_translation,
            "review_corpus": self._handle_review_corpus,
            "review_reply": self._handle_review_reply,
            "self_judge": self._handle_self_judge,
            "cross_validate": self._handle_cross_validate,
        }

        handler = handlers.get(task)
        if not handler:
            return self._make_error(message, f"未知任務類型: {task}")

        try:
            return handler(message, params)
        except Exception as e:
            logger.error(f"[QualityAgent] 任務 {task} 失敗: {e}", exc_info=True)
            return self._make_error(message, str(e))

    # ── 翻譯品質審核 ──

    def _handle_review_translation(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """審核翻譯品質"""
        original = params.get("original", "")
        translation = params.get("translation", "")
        direction = params.get("direction", "auto")
        method = params.get("method", "unknown")

        if not original or not translation:
            return self._make_error(msg, "審核需要 original 和 translation 參數")

        # 快速規則檢查（不需要 LLM）
        rule_checks = self._rule_check_translation(original, translation)

        # 如果規則檢查分數很高，直接通過
        if rule_checks["score"] >= 0.9:
            return self._make_response(
                msg, task="review_translation", status="completed",
                data={
                    "passed": True,
                    "score": rule_checks["score"],
                    "method": "rule_check",
                    "checks": rule_checks["checks"],
                },
                confidence=rule_checks["score"],
            )

        # 分數不夠高，用 LLM 進一步評估
        if self.client:
            llm_review = self._llm_review_translation(original, translation, direction)
            combined_score = (rule_checks["score"] + llm_review["score"]) / 2
            passed = combined_score >= 0.7

            return self._make_response(
                msg, task="review_translation", status="completed",
                data={
                    "passed": passed,
                    "score": round(combined_score, 2),
                    "method": "rule+llm",
                    "checks": rule_checks["checks"],
                    "llm_feedback": llm_review.get("feedback", ""),
                },
                confidence=round(combined_score, 2),
            )

        # LLM 不可用，只用規則
        passed = rule_checks["score"] >= 0.6
        return self._make_response(
            msg, task="review_translation", status="completed",
            data={
                "passed": passed,
                "score": rule_checks["score"],
                "method": "rule_only",
                "checks": rule_checks["checks"],
            },
            confidence=rule_checks["score"],
        )

    def _rule_check_translation(self, original: str, translation: str) -> dict:
        """規則基礎的翻譯品質檢查"""
        checks = {}
        score_parts = []

        # 檢查 1：是否有內容
        if translation and not translation.startswith("[不確定]"):
            checks["has_content"] = True
            score_parts.append(1.0)
        else:
            checks["has_content"] = False
            score_parts.append(0.0)

        # 檢查 2：長度是否合理（翻譯不應該比原文長太多或短太多）
        len_ratio = len(translation) / max(len(original), 1)
        if 0.2 <= len_ratio <= 5.0:
            checks["length_reasonable"] = True
            score_parts.append(0.8)
        else:
            checks["length_reasonable"] = False
            score_parts.append(0.3)

        # 檢查 3：是否包含可疑標記
        suspicious = any(
            marker in translation
            for marker in ["[不確定]", "[unsure]", "我無法", "我不確定"]
        )
        checks["no_uncertainty_markers"] = not suspicious
        score_parts.append(1.0 if not suspicious else 0.2)

        avg_score = sum(score_parts) / len(score_parts) if score_parts else 0.5
        return {"score": round(avg_score, 2), "checks": checks}

    def _llm_review_translation(self, original: str, translation: str,
                                 direction: str) -> dict:
        """LLM 評估翻譯品質"""
        prompt = (
            f"評估以下排灣語翻譯的品質。\n\n"
            f"原文: {original}\n"
            f"翻譯: {translation}\n"
            f"方向: {direction}\n\n"
            f"請評估：\n"
            f"1. score: 準確度 0-1\n"
            f"2. feedback: 簡短評語\n\n"
            f"輸出 JSON: {{\"score\": 0.8, \"feedback\": \"...\"}}"
        )

        llm_result = self.call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
            tools=None,
        )

        content = llm_result["message"].content.strip()
        try:
            if "```" in content:
                content = content.split("```")[1]
            return json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            return {"score": 0.5, "feedback": "無法解析 LLM 評估"}

    # ── 語料品質檢查 ──

    def _handle_review_corpus(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """審核語料品質"""
        paiwan = params.get("paiwan", "")
        chinese = params.get("chinese", "")
        source = params.get("source", "unknown")

        checks = {}

        # 基本完整性
        checks["has_paiwan"] = bool(paiwan.strip())
        checks["has_chinese"] = bool(chinese.strip())

        # 排灣語應該包含拉丁字母（不含中文）
        has_chinese_chars = any("\u4e00" <= c <= "\u9fff" for c in paiwan)
        checks["paiwan_no_chinese"] = not has_chinese_chars

        # 中文不應該包含拉丁字母為主
        latin_ratio = sum(1 for c in chinese if c.isascii() and c.isalpha()) / max(len(chinese), 1)
        checks["chinese_not_latin"] = latin_ratio < 0.5

        score = sum(1 for v in checks.values() if v) / len(checks) if checks else 0
        passed = score >= 0.75

        return self._make_response(
            msg, task="review_corpus", status="completed",
            data={
                "passed": passed,
                "score": round(score, 2),
                "checks": checks,
                "source": source,
            },
        )

    # ── 回覆品質審核 ──

    def _handle_review_reply(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """審核 Agent 回覆品質"""
        reply = params.get("reply", "")
        user_input = params.get("user_input", "")
        tool_results = params.get("tool_results", [])

        checks = {}

        # 檢查 1：回覆非空
        checks["non_empty"] = len(reply.strip()) > 0

        # 檢查 2：長度合理（不太短也不太長）
        checks["length_ok"] = 5 <= len(reply) <= 500

        # 檢查 3：有使用工具結果（如果有提供的話）
        if tool_results:
            # 簡單檢查：回覆是否引用了工具結果中的關鍵詞
            all_result_text = json.dumps(tool_results, ensure_ascii=False)
            overlap = any(
                word in reply for word in all_result_text.split()
                if len(word) > 2
            )
            checks["references_tool_results"] = overlap
        else:
            checks["references_tool_results"] = True  # 沒工具結果時不扣分

        score = sum(1 for v in checks.values() if v) / len(checks) if checks else 0
        passed = score >= 0.6

        return self._make_response(
            msg, task="review_reply", status="completed",
            data={
                "passed": passed,
                "score": round(score, 2),
                "checks": checks,
            },
        )

    # ── Self-Judge ──

    def _handle_self_judge(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """
        Self-Judge：判斷目標是否達成

        這是唐杰老師說的第二種長程任務方法的核心。
        用規則 + LLM 綜合判斷。
        """
        goal = params.get("goal", "")
        current_metrics = params.get("metrics", {})
        target_metrics = params.get("target_metrics", {})
        iteration = params.get("iteration", 0)
        max_iterations = params.get("max_iterations", 3)

        # 規則檢查：量化指標是否達標
        rule_achieved = True
        gaps = []
        for key, target in target_metrics.items():
            current = current_metrics.get(key, 0)
            if current < target:
                rule_achieved = False
                gaps.append(f"{key}: {current:.2f} / {target:.2f}")

        if rule_achieved:
            return self._make_response(
                msg, task="self_judge", status="completed",
                data={
                    "achieved": True,
                    "confidence": 1.0,
                    "gaps": [],
                    "next_action": "complete",
                    "reason": "所有量化指標已達標",
                },
            )

        # 未達標：決定下一步
        if iteration >= max_iterations:
            next_action = "stop"
            reason = f"達到最大迭代次數 ({max_iterations})，目標未完全達成"
        else:
            next_action = "continue"
            reason = f"差距: {', '.join(gaps)}"

        # 如果有 LLM，讓它判斷是否值得繼續
        if self.client and gaps:
            llm_judgment = self._llm_self_judge(goal, gaps, iteration, max_iterations)
            next_action = llm_judgment.get("next_action", next_action)
            reason = llm_judgment.get("reason", reason)

        return self._make_response(
            msg, task="self_judge", status="completed",
            data={
                "achieved": False,
                "confidence": round(
                    sum(current_metrics.values()) / max(sum(target_metrics.values()), 1), 2
                ),
                "gaps": gaps,
                "next_action": next_action,
                "reason": reason,
                "iteration": iteration,
            },
        )

    def _llm_self_judge(self, goal: str, gaps: list,
                        iteration: int, max_iterations: int) -> dict:
        """LLM 輔助的 Self-Judge"""
        prompt = (
            f"你是 Agent 系統的自我監督模組。\n\n"
            f"目標: {goal}\n"
            f"當前差距: {', '.join(gaps)}\n"
            f"迭代次數: {iteration}/{max_iterations}\n\n"
            f"判斷下一步：\n"
            f"- continue: 繼續執行\n"
            f"- adjust: 調整策略\n"
            f"- stop: 停止（不可能達成或成本太高）\n\n"
            f'輸出 JSON: {{"next_action": "continue", "reason": "..."}}'
        )

        try:
            llm_result = self.call_llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,
                tools=None,
            )
            content = llm_result["message"].content.strip()
            if "```" in content:
                content = content.split("```")[1]
            return json.loads(content.strip())
        except Exception:
            return {"next_action": "continue", "reason": "LLM 判斷失敗，預設繼續"}

    # ── 交叉驗證 ──

    def _handle_cross_validate(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """
        交叉驗證：獨立翻譯一次，比對結果

        用於驗證 Knowledge Agent 的翻譯是否可靠
        """
        text = params.get("text", "")
        original_translation = params.get("translation", "")
        direction = params.get("direction", "auto")

        if not self.client:
            return self._make_error(msg, "交叉驗證需要 LLM")

        # 獨立翻譯
        dir_hint = "排灣語→中文" if direction == "p2c" else "中文→排灣語" if direction == "c2p" else "自動偵測"
        prompt = (
            f"翻譯以下文本（{dir_hint}）。只輸出翻譯結果。\n\n"
            f"{text}"
        )

        llm_result = self.call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            tools=None,
        )

        independent_translation = llm_result["message"].content.strip()

        # 簡單字串相似度比對
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(
            None,
            original_translation.lower(),
            independent_translation.lower(),
        ).ratio()

        passed = similarity >= 0.5  # 翻譯不需要完全一致

        return self._make_response(
            msg, task="cross_validate", status="completed",
            data={
                "original_translation": original_translation,
                "independent_translation": independent_translation,
                "similarity": round(similarity, 2),
                "passed": passed,
                "note": "交叉驗證：兩次獨立翻譯的相似度",
            },
        )


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    print("=" * 60)
    print("  🔍 QualityAgent 測試")
    print("=" * 60)

    client = OpenAI(
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    agent = QualityAgent(client=client)

    # 測試 1：翻譯審核（高品質）
    print("\n--- 測試 1：翻譯審核（好翻譯）---")
    msg1 = AgentMessage.task_assign(
        from_agent="orchestrator", to_agent="quality",
        task="review_translation",
        params={"original": "謝謝", "translation": "masalu", "direction": "c2p"},
    )
    resp1 = agent.handle_message(msg1)
    d1 = resp1.payload.get("data", {})
    print(f"  通過: {d1.get('passed')} | 分數: {d1.get('score')} | 方法: {d1.get('method')}")

    # 測試 2：翻譯審核（差翻譯）
    print("\n--- 測試 2：翻譯審核（差翻譯）---")
    msg2 = AgentMessage.task_assign(
        from_agent="orchestrator", to_agent="quality",
        task="review_translation",
        params={"original": "謝謝", "translation": "[不確定]", "direction": "c2p"},
    )
    resp2 = agent.handle_message(msg2)
    d2 = resp2.payload.get("data", {})
    print(f"  通過: {d2.get('passed')} | 分數: {d2.get('score')}")

    # 測試 3：語料審核
    print("\n--- 測試 3：語料審核 ---")
    msg3 = AgentMessage.task_assign(
        from_agent="orchestrator", to_agent="quality",
        task="review_corpus",
        params={"paiwan": "masalu", "chinese": "謝謝", "source": "test"},
    )
    resp3 = agent.handle_message(msg3)
    d3 = resp3.payload.get("data", {})
    print(f"  通過: {d3.get('passed')} | 檢查: {d3.get('checks')}")

    # 測試 4：Self-Judge（未達標）
    print("\n--- 測試 4：Self-Judge ---")
    msg4 = AgentMessage.task_assign(
        from_agent="orchestrator", to_agent="quality",
        task="self_judge",
        params={
            "goal": "翻譯品質達到 0.85",
            "metrics": {"translation_quality": 0.65},
            "target_metrics": {"translation_quality": 0.85},
            "iteration": 1,
            "max_iterations": 3,
        },
    )
    resp4 = agent.handle_message(msg4)
    d4 = resp4.payload.get("data", {})
    print(f"  達成: {d4.get('achieved')} | 下一步: {d4.get('next_action')} | 差距: {d4.get('gaps')}")

    # 測試 5：Self-Judge（已達標）
    print("\n--- 測試 5：Self-Judge（已達標）---")
    msg5 = AgentMessage.task_assign(
        from_agent="orchestrator", to_agent="quality",
        task="self_judge",
        params={
            "goal": "翻譯品質達到 0.85",
            "metrics": {"translation_quality": 0.92},
            "target_metrics": {"translation_quality": 0.85},
            "iteration": 2,
            "max_iterations": 3,
        },
    )
    resp5 = agent.handle_message(msg5)
    d5 = resp5.payload.get("data", {})
    print(f"  達成: {d5.get('achieved')} | 下一步: {d5.get('next_action')}")

    print("\n✅ QualityAgent 所有測試完成")
