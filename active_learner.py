"""
active_learner.py — Human-in-the-Loop 主動學習引擎

核心理念：
瀕危語言的語料不會從天上掉下來。我們設計了一個持續增長的閉環系統：
用戶每次使用都在貢獻語料，社區成員可以驗證轉寫正確性，
經驗證的語料自動回流到 RAG 知識庫，形成正向循環。

流程：
1. 用戶輸入（文字/語音）→ ASR/翻譯服務處理
2. 系統評估置信度（基於翻譯方法 + RAG 匹配度）
3. 高置信度 → 自動標記為「待驗證」寫入 Bitable
4. 低置信度 → 標記為「需確認」+ 提示用戶確認
5. 社區驗證 → 通過後加入語料庫
6. 定期重訓 RAG 索引

引用：
- Speechless (Dao et al. 2025): 低資源語言語音指令數據稀缺問題
- Synthetic Voice Data for ASR in African Languages (DeRenzi et al. 2025): 合成數據管線

作者: 地陪
日期: 2026-04-29
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/Users/sbb-mei/Desktop/paiwan_competition_2026")
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "merged_corpus.json"
FEEDBACK_LOG = DATA_DIR / "active_learning_log.jsonl"


class ActiveLearner:
    """主動學習引擎 — 瀕危語言數據持續增長的閉環系統"""

    # 置信度閾值
    HIGH_CONFIDENCE = 0.85    # 自動接受，標記「待驗證」
    MEDIUM_CONFIDENCE = 0.50  # 提示用戶確認
    # < 0.50: 標記「需人工確認」

    def __init__(self):
        self._log_file = FEEDBACK_LOG
        self._ensure_log()

    def _ensure_log(self):
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_file.exists():
            self._log_file.touch()

    def evaluate_confidence(self, result: dict) -> dict:
        """
        評估翻譯結果的置信度

        Args:
            result: translate_service.translate() 的返回值

        Returns:
            {
                "confidence": float,       # 0-1 置信度
                "level": str,              # high/medium/low
                "needs_confirmation": bool, # 是否需要用戶確認
                "needs_human_review": bool, # 是否需要語言學家審核
                "auto_accept": bool,       # 是否自動接受
            }
        """
        method = result.get("method", "unknown")
        rag_results = result.get("rag_results", [])

        # 方法置信度基線
        if method == "exact":
            # 精確匹配 = 在語料庫中找到了完全一致的翻譯
            base_confidence = 0.95
        elif method == "rag_llm":
            # RAG + LLM = 檢索到相關語料後由 LLM 生成
            base_confidence = 0.70
            # 如果有 Top-1 結果且相似度高，提升置信度
            if rag_results:
                top1 = rag_results[0] if isinstance(rag_results[0], dict) else {}
                # 簡單啟發式：RAG 有結果就加分
                base_confidence += min(0.15, len(rag_results) * 0.03)
        else:
            # error 或 unknown
            base_confidence = 0.20

        confidence = min(base_confidence, 1.0)

        # 分級
        if confidence >= self.HIGH_CONFIDENCE:
            level = "high"
            needs_confirmation = False
            needs_human_review = False
            auto_accept = True
        elif confidence >= self.MEDIUM_CONFIDENCE:
            level = "medium"
            needs_confirmation = True
            needs_human_review = False
            auto_accept = False
        else:
            level = "low"
            needs_confirmation = True
            needs_human_review = True
            auto_accept = False

        return {
            "confidence": round(confidence, 3),
            "level": level,
            "needs_confirmation": needs_confirmation,
            "needs_human_review": needs_human_review,
            "auto_accept": auto_accept,
        }

    def log_interaction(
        self,
        user_input: str,
        translation_result: dict,
        confidence_eval: dict,
        user_feedback: str = None,
        source: str = "飛書對話",
    ):
        """
        記錄一次交互到學習日誌

        Args:
            user_input: 用戶原始輸入
            translation_result: 翻譯服務返回值
            confidence_eval: 置信度評估結果
            user_feedback: 用戶反饋（如果有）
            source: 數據來源
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "translation": translation_result.get("translation", ""),
            "direction": translation_result.get("direction", "unknown"),
            "method": translation_result.get("method", "unknown"),
            "confidence": confidence_eval["confidence"],
            "level": confidence_eval["level"],
            "auto_accept": confidence_eval["auto_accept"],
            "needs_confirmation": confidence_eval["needs_confirmation"],
            "user_feedback": user_feedback,
            "source": source,
            "verified": False,
        }

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry

    def generate_confirmation_prompt(self, original: str, translation: str, level: str) -> str:
        """
        生成用戶確認提示

        根據置信度等級，生成不同程度的確認提示
        """
        if level == "medium":
            return (
                f"📝 vuvu 不太確定這個翻譯對不對：\n\n"
                f"你說的：{original}\n"
                f"我翻譯的：{translation}\n\n"
                f"這個翻譯正確嗎？回覆 ✅ 正確 / ❌ 不正確 來幫助我學習！"
            )
        else:  # low
            return (
                f"🤔 vuvu 對這句話不太有把握：\n\n"
                f"你說的：{original}\n"
                f"我翻譯的：{translation}\n\n"
                f"可以幫我確認一下嗎？回覆 ✅ 正確 / ❌ 不正確\n"
                f"如果知道正確答案，也可以直接告訴我！"
            )

    def process_feedback(self, original_input: str, user_feedback: str) -> str | None:
        """
        處理用戶對翻譯結果的反饋

        Returns:
            回覆訊息，或 None（如果無法處理）
        """
        feedback = user_feedback.strip()

        if feedback in ["✅", "正確", "對", "yes", "y"]:
            return "太好了！謝謝你幫我確認 💛 我會記住這個翻譯！"
        elif feedback in ["❌", "不正確", "錯", "no", "n"]:
            return "謝謝告訴我！我會把這個記下來，下次翻譯更準確 💪\n（如果你知道正確答案，可以告訴我～）"
        else:
            # 用戶提供了正確答案
            return f"謝謝教我！我記住了：\n\n{feedback}\n\n下次會更準確！💛"

    def get_pending_verifications(self, limit: int = 50) -> list:
        """獲取待驗證的語料"""
        pending = []
        if not self._log_file.exists():
            return pending

        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if not entry.get("verified") and entry.get("needs_confirmation"):
                        pending.append(entry)
                except json.JSONDecodeError:
                    continue

        return pending[-limit:]

    def get_stats(self) -> dict:
        """獲取主動學習統計"""
        total = 0
        auto_accepted = 0
        needs_confirm = 0
        verified = 0
        by_level = {"high": 0, "medium": 0, "low": 0}

        if not self._log_file.exists():
            return {"total": 0}

        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    total += 1
                    level = entry.get("level", "unknown")
                    if level in by_level:
                        by_level[level] += 1
                    if entry.get("auto_accept"):
                        auto_accepted += 1
                    if entry.get("needs_confirmation"):
                        needs_confirm += 1
                    if entry.get("verified"):
                        verified += 1
                except json.JSONDecodeError:
                    continue

        return {
            "total_interactions": total,
            "auto_accepted": auto_accepted,
            "needs_confirmation": needs_confirm,
            "verified": verified,
            "by_level": by_level,
            "verification_rate": round(verified / max(total, 1), 3),
        }

    def export_verified_to_corpus(self, output_path: str = None) -> dict:
        """
        將已驗證的語料導出為語料庫格式

        Returns:
            {"exported": int, "total_verified": int, "path": str}
        """
        output_path = output_path or str(DATA_DIR / "community_verified_corpus.json")
        verified = []

        if not self._log_file.exists():
            return {"exported": 0, "total_verified": 0, "path": output_path}

        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("verified") and entry.get("direction") == "p2c":
                        verified.append({
                            "paiwan": entry["input"],
                            "chinese": entry["translation"],
                            "source": f"社區驗證_{entry.get('source', 'unknown')}",
                            "verified_at": entry.get("timestamp", ""),
                        })
                except json.JSONDecodeError:
                    continue

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"entries": verified, "metadata": {"exported_at": datetime.now().isoformat()}}, f, ensure_ascii=False, indent=2)

        return {
            "exported": len(verified),
            "total_verified": len(verified),
            "path": output_path,
        }


# ============================================
# Bitable 驗證狀態更新
# ============================================

def update_bitable_verification(record_id: str, status: str, correct_translation: str = "") -> dict:
    """
    更新 Bitable 中的驗證狀態

    Args:
        record_id: Bitable 記錄 ID
        status: "✅正確" / "⚠️需修正" / "❌錯誤"
        correct_translation: 修正後的正確翻譯
    """
    import requests
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    APP_ID = os.getenv("FEISHU_APP_ID", "")
    APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
    BITABLE_APP_TOKEN = "KfUUbiz5taJhqJsFrDocjGP2nQu"
    BITABLE_TABLE_ID = "tbl4WiIy8QIoa8pf"

    # 獲取 token
    resp = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": APP_ID, "app_secret": APP_SECRET},
        timeout=10,
    )
    token = resp.json().get("tenant_access_token", "")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    fields = {"是否正確": status}
    if correct_translation:
        fields["中文翻譯"] = correct_translation

    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}/tables/{BITABLE_TABLE_ID}/records/{record_id}"
    resp = requests.put(url, headers=headers, json={"fields": fields}, timeout=15)
    return resp.json()


# ============================================
# CLI 測試
# ============================================

if __name__ == "__main__":
    learner = ActiveLearner()

    # 模擬不同方法的翻譯結果
    test_cases = [
        {"method": "exact", "translation": "謝謝", "rag_results": [{"paiwan": "masalu", "chinese": "謝謝"}]},
        {"method": "rag_llm", "translation": "你好嗎", "rag_results": [{"paiwan": "nanguangua su?", "chinese": "你好嗎"}]},
        {"method": "error", "translation": "[翻譯失敗]", "rag_results": []},
    ]

    print("=== 主動學習引擎測試 ===\n")
    for tc in test_cases:
        eval_result = learner.evaluate_confidence(tc)
        print(f"方法: {tc['method']}")
        print(f"  置信度: {eval_result['confidence']}")
        print(f"  等級: {eval_result['level']}")
        print(f"  自動接受: {eval_result['auto_accept']}")
        print(f"  需確認: {eval_result['needs_confirmation']}")
        if eval_result['needs_confirmation']:
            prompt = learner.generate_confirmation_prompt("測試輸入", tc["translation"], eval_result['level'])
            print(f"  確認提示: {prompt[:60]}...")
        print()
