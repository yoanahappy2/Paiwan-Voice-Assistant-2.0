"""
knowledge_graph.py — 排灣語輕量知識圖譜

利用飛書 Bitable 的關聯功能，構建：
1. 詞彙關聯表 — 綴詞變化、同義詞、反義詞
2. 親屬詞彙表 — 家族稱謂關係
3. 語法規則表 — 前綴/後綴用法

這是 Gemini 建議的「降級實現」：
不做 Neo4j，用 Bitable 關聯功能達到知識圖譜效果。
既有技術深度，又符合飛書競賽的「原生整合」要求。

場景：用戶查詞時，Bot 自動推薦相關詞彙和綴詞變化

作者: 地陪
日期: 2026-04-29
"""

import json
import os
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/Users/sbb-mei/Desktop/paiwan_competition_2026")
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "merged_corpus.json"


# ============================================
# 知識圖譜數據
# ============================================

# 排灣語前綴體系（來自語言學分析）
PREFIXES = {
    "ma-": {"meaning": "狀態/形容詞", "example": "maledep (美麗)", "category": "狀態前綴"},
    "na-": {"meaning": "過去/已完成", "example": "na tarivak (已經來了)", "category": "時態前綴"},
    "uri-": {"meaning": "未來/將要", "example": "uri sema (將要去)", "category": "時態前綴"},
    "ka-": {"meaning": "使役/造成", "example": "kemaledep (使美麗)", "category": "使役前綴"},
    "ki-": {"meaning": "獲得/取得", "example": "kipusalu (感謝/得到恩惠)", "category": "動作前綴"},
    "pi-": {"meaning": "做/製造", "example": "pisalisin (祭祀)", "category": "動作前綴"},
    "si-": {"meaning": "互相/一起", "example": "siasing (拍照)", "category": "互動前綴"},
    "tja-": {"meaning": "比較級/更", "example": "tjaljananguaq (更好)", "category": "比較前綴"},
    "ini-": {"meaning": "否定/不", "example": "ini (不)", "category": "否定前綴"},
    "se-": {"meaning": "群體/來自", "example": "sepadain (來自padain部落)", "category": "來源前綴"},
}

# 親屬稱謂關係圖
KINSHIP = {
    "vuvu": {"chinese": "祖父母/長輩", "relation": "長輩", "gender": "通用"},
    "vuvu a vavayan": {"chinese": "祖母/女性長輩", "relation": "長輩", "gender": "女"},
    "vuvu a uqaljay": {"chinese": "祖父/男性長輩", "relation": "長輩", "gender": "男"},
    "kina": {"chinese": "母親", "relation": "父母", "gender": "女"},
    "kama/ama": {"chinese": "父親", "relation": "父母", "gender": "男"},
    "kaka": {"chinese": "兄姊/年長者", "relation": "平輩(長)", "gender": "通用"},
    "kaka a uqaljay": {"chinese": "哥哥", "relation": "平輩(長)", "gender": "男"},
    "kaka a vavayan": {"chinese": "姐姐", "relation": "平輩(長)", "gender": "女"},
    "ali": {"chinese": "弟妹/年幼者", "relation": "平輩(幼)", "gender": "通用"},
    "tjuku": {"chinese": "阿姨/姑姑", "relation": "旁系長輩", "gender": "女"},
    "qalju/imu": {"chinese": "叔叔/舅舅", "relation": "旁系長輩", "gender": "男"},
    "aljak": {"chinese": "小孩/孩子", "relation": "晚輩", "gender": "通用"},
    "tjaljaljak": {"chinese": "最小的孩子", "relation": "晚輩", "gender": "通用"},
}

# 詞彙群組（同義詞、相關詞）
WORD_GROUPS = [
    {
        "theme": "問候",
        "words": [
            ("nanguangua", "你好"),
            ("aiyanga", "你好"),
            ("djavadjvai", "你好"),
            ("masalu", "謝謝"),
            ("kipusalu", "感謝"),
            ("navenec", "再見"),
        ]
    },
    {
        "theme": "自然",
        "words": [
            ("qadaw", "太陽/日子"),
            ("zung", "月亮"),
            ("veqalj", "雨"),
            ("qerepus", "雲"),
            ("kadjunangan", "土地"),
            ("gadu", "山"),
            ("ljavek", "海"),
        ]
    },
    {
        "theme": "動物",
        "words": [
            ("ngiyau", "貓"),
            ("vatu", "狗"),
            ("luni/drail", "猴子"),
            ("qaya", "魚"),
            ("ayam", "鳥"),
        ]
    },
    {
        "theme": "身體",
        "words": [
            ("mata", "眼睛"),
            ("tjalinga", "耳朵"),
            ("ngudjang", "鼻子"),
            ("djamulj", "手"),
            ("tjainan", "腳"),
            ("dridri", "眼睛/淚水"),
            ("qacang", "肚子"),
        ]
    },
    {
        "theme": "數字",
        "words": [
            ("itay", "一"),
            ("drusa", "二"),
            ("tjelj", "三"),
            ("sepatj", "四"),
            ("lima", "五"),
        ]
    },
    {
        "theme": "美麗與好",
        "words": [
            ("maledep", "美麗"),
            ("ananguaq", "好"),
            ("tjaljananguaq", "更好"),
            ("tjaljiqaca", "廣大遼闊"),
        ]
    },
]


class PaiwanKnowledgeGraph:
    """排灣語輕量知識圖譜"""

    def __init__(self):
        self.prefixes = PREFIXES
        self.kinship = KINSHIP
        self.word_groups = WORD_GROUPS
        self._corpus_index = None

    def _build_corpus_index(self):
        """建立語料庫索引"""
        if self._corpus_index is not None:
            return
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)["entries"]

        self._corpus_index = defaultdict(list)
        for e in entries:
            p = e.get("paiwan", "").strip().lower()
            c = e.get("chinese", "").strip()
            if p and c:
                self._corpus_index[p].append(c)

    def lookup(self, word: str) -> dict:
        """
        查詞：返回詞義、相關詞、綴詞分析、親屬關係
        """
        self._build_corpus_index()
        word_lower = word.strip().lower()
        result = {
            "word": word,
            "translation": None,
            "prefix_analysis": None,
            "related_words": [],
            "kinship": None,
            "theme": None,
        }

        # 1. 查語料庫翻譯
        if word_lower in self._corpus_index:
            result["translation"] = self._corpus_index[word_lower][0]

        # 2. 查親屬稱謂
        if word_lower in self.kinship:
            result["kinship"] = self.kinship[word_lower]

        # 3. 綴詞分析
        for prefix, info in self.prefixes.items():
            if word_lower.startswith(prefix.rstrip("-")):
                result["prefix_analysis"] = {
                    "prefix": prefix,
                    **info,
                    "root": word_lower[len(prefix.rstrip("-")):],
                }
                break

        # 4. 查詞彙群組（相關詞）
        for group in self.word_groups:
            for pw, cn in group["words"]:
                if word_lower == pw.lower() or word_lower == cn:
                    result["theme"] = group["theme"]
                    result["related_words"] = [
                        {"paiwan": w, "chinese": c}
                        for w, c in group["words"]
                        if w.lower() != word_lower and c != word
                    ]
                    break
            if result["related_words"]:
                break

        # 5. 反向查找（用中文找排灣語）
        if not result["translation"]:
            for group in self.word_groups:
                for pw, cn in group["words"]:
                    if word_lower in cn.lower():
                        result["translation"] = f"{pw} ({cn})"
                        result["related_words"] = [
                            {"paiwan": w, "chinese": c}
                            for w, c in group["words"]
                            if w != pw
                        ]
                        result["theme"] = group["theme"]
                        break

        return result

    def format_lookup_reply(self, result: dict) -> str:
        """格式化查詞結果為 Bot 回覆"""
        if not result["translation"] and not result["kinship"] and not result["prefix_analysis"]:
            return None

        lines = []
        lines.append(f"📖 詞彙解析：{result['word']}")
        lines.append("")

        if result["translation"]:
            lines.append(f"🔵 翻譯：{result['translation']}")

        if result["kinship"]:
            k = result["kinship"]
            lines.append(f"👨‍👩‍👧 親屬關係：{k['chinese']}（{k['relation']}，{k['gender']}性）")

        if result["prefix_analysis"]:
            p = result["prefix_analysis"]
            lines.append(f"🔧 綴詞分析：前綴 [{p['prefix']}] = {p['meaning']}")
            lines.append(f"   詞根：{p['root']}")

        if result["theme"]:
            lines.append(f"\n📂 詞彙主題：{result['theme']}")

        if result["related_words"]:
            lines.append("\n🔗 相關詞彙：")
            for rw in result["related_words"][:6]:
                lines.append(f"   • {rw['paiwan']} — {rw['chinese']}")

        return "\n".join(lines)

    def get_all_prefixes(self) -> list:
        """獲取所有前綴列表"""
        return [{"prefix": k, **v} for k, v in self.prefixes.items()]

    def get_kinship_tree(self) -> list:
        """獲取親屬稱謂列表"""
        return [{"paiwan": k, **v} for k, v in self.kinship.items()]

    def get_word_groups(self) -> list:
        """獲取詞彙群組"""
        return self.word_groups


# ============================================
# CLI 測試
# ============================================

if __name__ == "__main__":
    kg = PaiwanKnowledgeGraph()

    test_words = ["masalu", "kina", "maledep", "tjaljananguaq", "vuvu", "謝謝", "母親", "ngiyau"]
    print("=== 排灣語知識圖譜測試 ===\n")

    for word in test_words:
        result = kg.lookup(word)
        reply = kg.format_lookup_reply(result)
        if reply:
            print(reply)
            print("-" * 40)
