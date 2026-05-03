"""
feishu_bot/user_profile.py — 用戶學習檔案

追蹤每個用戶學過的詞、答錯的詞、學習曲線。
持久化到 data/user_profiles.json，Bot 重啟不丟失。

作者: 地陪
日期: 2026-05-01
"""

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_PATH = PROJECT_ROOT / "data" / "user_profiles.json"


class UserProfile:
    """用戶學習檔案 — 追蹤學過的詞、錯過的詞、學習曲線"""

    def __init__(self):
        self._profiles = {}  # {open_id: dict}
        self.load()

    def _ensure(self, open_id: str) -> dict:
        if open_id not in self._profiles:
            self._profiles[open_id] = {
                "learned": [],       # list of words (ordered)
                "learned_set": [],   # for quick lookup (serialized from set)
                "wrong_count": {},   # {word: count}
                "correct_count": {}, # {word: count}
                "streak": 0,
                "last_seen": None,
                "first_seen": date.today().isoformat(),
            }
        return self._profiles[open_id]

    def _in_learned(self, profile: dict, word: str) -> bool:
        return word in profile.get("learned_set", [])

    def record_learn(self, open_id: str, word: str):
        """記錄學過一個詞"""
        p = self._ensure(open_id)
        word = word.strip().lower()
        if not self._in_learned(p, word):
            p["learned"].append(word)
            p["learned_set"].append(word)
        today = date.today().isoformat()
        if p["last_seen"] != today:
            # check streak
            if p["last_seen"]:
                from datetime import timedelta
                try:
                    last = datetime.strptime(p["last_seen"], "%Y-%m-%d").date()
                    if (date.today() - last).days == 1:
                        p["streak"] += 1
                    elif (date.today() - last).days > 1:
                        p["streak"] = 1
                except Exception:
                    p["streak"] = 1
            else:
                p["streak"] = 1
            p["last_seen"] = today
        self.save()

    def record_wrong(self, open_id: str, word: str):
        """記錄答錯一個詞"""
        p = self._ensure(open_id)
        word = word.strip().lower()
        p["wrong_count"][word] = p["wrong_count"].get(word, 0) + 1
        self.record_learn(open_id, word)
        self.save()

    def record_correct(self, open_id: str, word: str):
        """記錄答對一個詞"""
        p = self._ensure(open_id)
        word = word.strip().lower()
        p["correct_count"][word] = p["correct_count"].get(word, 0) + 1
        self.record_learn(open_id, word)
        self.save()

    def get_learned_words(self, open_id: str) -> list:
        """取得用戶學過的所有詞"""
        p = self._ensure(open_id)
        return list(p.get("learned", []))

    def get_weak_words(self, open_id: str, threshold: int = 2) -> list:
        """取得用戶常錯的詞（錯 ≥ threshold 次）"""
        p = self._ensure(open_id)
        weak = []
        for word, count in p.get("wrong_count", {}).items():
            correct = p.get("correct_count", {}).get(word, 0)
            if count > correct or count >= threshold:
                weak.append(word)
        return weak

    def suggest_next(self, open_id: str, corpus: list) -> Optional[dict]:
        """根據已學詞彙推薦下一個學習內容"""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from knowledge_graph import WORD_GROUPS

        p = self._ensure(open_id)
        learned_set = set(p.get("learned_set", []))

        # 1. 找已學詞的相關詞（同一 WORD_GROUP 裡還沒學過的）
        related_candidates = []
        for group in WORD_GROUPS:
            for pw, cn in group["words"]:
                if pw.lower() in learned_set:
                    for pw2, cn2 in group["words"]:
                        if pw2.lower() not in learned_set:
                            related_candidates.append({"paiwan": pw2, "chinese": cn2, "theme": group["theme"]})

        if related_candidates:
            import random
            return random.choice(related_candidates)

        # 2. 從語料庫推薦還沒學過的
        unlearned = [
            e for e in corpus
            if e.get("paiwan", e.get("排灣語", "")).strip().lower() not in learned_set
               and e.get("paiwan", e.get("排灣語", ""))
               and e.get("chinese", e.get("中文", ""))
        ]
        if unlearned:
            import random
            e = random.choice(unlearned)
            return {
                "paiwan": e.get("paiwan", e.get("排灣語", "")),
                "chinese": e.get("chinese", e.get("中文", "")),
            }

        return None

    def get_stats(self, open_id: str) -> dict:
        """學習統計摘要"""
        p = self._ensure(open_id)
        total_learned = len(p.get("learned", []))
        total_wrong = sum(p.get("wrong_count", {}).values())
        total_correct = sum(p.get("correct_count", {}).values())
        total_attempts = total_wrong + total_correct
        accuracy = round(total_correct / total_attempts * 100) if total_attempts > 0 else 0

        return {
            "total_learned": total_learned,
            "total_correct": total_correct,
            "total_wrong": total_wrong,
            "accuracy": accuracy,
            "streak": p.get("streak", 0),
            "last_seen": p.get("last_seen"),
            "first_seen": p.get("first_seen"),
        }

    def format_notebook(self, open_id: str, corpus: list = None) -> str:
        """格式化 /notebook 回覆"""
        p = self._ensure(open_id)
        stats = self.get_stats(open_id)
        learned = p.get("learned", [])
        weak = self.get_weak_words(open_id)

        lines = ["📒 你的排灣語單字本", ""]

        # 已學會
        correct_words = [w for w in learned if p.get("correct_count", {}).get(w, 0) > p.get("wrong_count", {}).get(w, 0)]
        lines.append(f"✅ 已學會（{len(correct_words)}）：")
        if correct_words:
            for w in correct_words[:20]:
                # try to find chinese from knowledge graph
                cn = self._find_chinese(w)
                lines.append(f"  {w} ({cn})" if cn else f"  {w}")
        else:
            lines.append("  （還沒有，快去 /learn 學習吧！）")

        lines.append("")

        # 需複習
        if weak:
            lines.append(f"❌ 需複習（{len(weak)}）：")
            for w in weak[:10]:
                cn = self._find_chinese(w)
                lines.append(f"  {w} ({cn})" if cn else f"  {w}")
            lines.append("")

        # 統計
        lines.append("📊 學習統計：")
        lines.append(f"  連續學習：{stats['streak']} 天")
        lines.append(f"  總詞彙量：{stats['total_learned']}")
        lines.append(f"  正確率：{stats['accuracy']}%")

        # 建議複習
        if weak:
            lines.append(f"\n💡 建議複習：{weak[0]}")
        elif corpus:
            suggestion = self.suggest_next(open_id, corpus)
            if suggestion:
                lines.append(f"\n💡 建議學習：{suggestion['paiwan']} ({suggestion['chinese']})")

        return "\n".join(lines)

    def _find_chinese(self, word: str) -> str:
        """從知識圖譜找中文釋義"""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from knowledge_graph import WORD_GROUPS, KINSHIP
            word_l = word.lower()
            for group in WORD_GROUPS:
                for pw, cn in group["words"]:
                    if pw.lower() == word_l:
                        return cn
            for pw, info in KINSHIP.items():
                if pw.lower() == word_l:
                    return info["chinese"]
        except Exception:
            pass
        return ""

    def save(self):
        """保存到 JSON 檔案"""
        PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILES_PATH, "w", encoding="utf-8") as f:
            json.dump(self._profiles, f, ensure_ascii=False, indent=2)

    def load(self):
        """從 JSON 檔案載入"""
        if PROFILES_PATH.exists():
            try:
                with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                    self._profiles = json.load(f)
            except Exception:
                self._profiles = {}
        else:
            self._profiles = {}


# 全域實例
user_profile = UserProfile()
