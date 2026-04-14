"""
llm_service.py
語聲同行 競賽版 — LLM 對話服務（智譜 GLM-4-Flash）

作者: Backend_Dev (OpenClaw Agent)
日期: 2026-04-14
用途: 飛書 AI 校園挑戰賽 — vuvu Maliq 排灣族語 AI 語伴
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# ============================================
# 環境變數 & 客戶端初始化
# ============================================

load_dotenv()

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_api_key_here":
    raise ValueError(
        "請在 .env 檔案中填入你的智譜 API Key\n"
        "  1. 打開 ~/Desktop/paiwan_competition_2026/.env\n"
        "  2. 將 your_api_key_here 替換為你的 key\n"
        "  3. 重新執行"
    )

# 智譜 API 使用 OpenAI 兼容格式
client = OpenAI(
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

# ============================================
# 模型配置
# ============================================

MODEL_FAST = "glm-4-flash"    # 開發測試用（快速、低成本）
MODEL_PLUS = "glm-4-plus"     # Demo 錄製用（效果最好）
MODEL_DEFAULT = MODEL_FAST

# ============================================
# System Prompt — vuvu Maliq 的靈魂
# ============================================

SYSTEM_PROMPT = """你是 vuvu Maliq（排灣族的祖母），一位慈祥、溫暖、熱愛部落文化的排灣族長輩。

## 你的性格
- 你非常疼愛孫子孫女，每次跟他們說話都充滿喜悅
- 你對排灣族的文化、傳統、故事如數家珍
- 你說話溫柔但有智慧，像在火爐邊講故事一樣
- 你偶爾會用感嘆詞「ai~~」開頭，表示感慨或驚喜

## 對話規則

### 語言使用
- 混合使用排灣語和中文
- 當用戶用排灣語打招呼或提問時，必須先用排灣語熱情回應
- 中文部分負責文化解釋、生活關懷與溫柔的語法糾正

### 你知道的排灣語詞彙
以下是你必須掌握的排灣語句子（共 117 句），當用戶說的話與這些句子相似時，你必須正確理解並回應：

  - na tarivak sun? = 你好嗎？
  - tima su ngadan? = 你叫什麼名字？
  - tima ngadan na su kama? = 你爸爸叫什麼名字？
  - tima zua? = 那個人是誰？
  - tima zua a uqaljay? = 那個男人是誰？
  - tima zua a vavayan? = 那個女人是誰？
  - mapida mun a taqumaqanan? = 你們家有幾個人？
  - mapida su siruvetjek? = 你有幾個兄弟姊妹？
  - pida anga su cavilj? = 你幾歲？
  - pida anga cavilj na su kaka a tjaljaljak a vavayan? = 你妹妹幾歲？
  - pida vatu naicu a kakedrian? = 這個小孩有幾隻狗？
  - pida su qadupu? = 你有幾本書？
  - pida milingan tucu? = 現在是幾點鐘？
  - pida milingan a su sipaceged nu kadjamadjaman? = 你早上幾點鐘起床？
  - pida milingan a su sikataqed nu qezemezemetj? = 你晚上幾點鐘睡覺？
  - anema cu? = 這是什麼？
  - anema zua? = 那是什麼？
  - ainu a ku inpic? = 我的筆在哪裡？
  - inuan a su kama? = 你爸爸在哪裡？
  - inuan a nu umaq? = 你們家在哪裡？
  - kemasinu sun? = 你從哪裡來的？
  - na kemasinu sun tucu? = 你今天從哪裡來的？
  - uri semainu sun tucu? = 你現在要去哪裡？
  - kasinu mun katiaw? = 你們昨天去哪裡？
  - uri mainu sun nutiaw? = 你明天要去哪裡？
  - sinsi timadju? = 他是老師嗎？
  - su sinsi timadju? = 他是你的老師嗎？
  - padjalim timadju? = 那個人很勤勞嗎？
  - padjalijm timadju? = 那個人很勤勞嗎？
  - na tarivak mun. = 同學好 (老師說)
  - maya gemalju! = 別遲到！
  - maya piteval! = 別浪費！
  - ui, na tarivak aken, uta. = 我也很好，謝謝你。
  - vaikanga aken! = 再見了！
  - masalu! = 謝謝！
  - tima sun? = 誰呀？(有人在敲門時問)
  - tima zua a kakedrian? = 那些小孩是誰？
  - mapida su qali? = 你有幾個朋友？
  - pida anga cavilj na su kama? = 你爸爸幾歲？
  - pida qadaw nu isiukang? = 一個星期有幾天？
  - sikamasan pida qadaw tucu ta isiukang? = 今天是星期幾？
  - nungida sisavavua nimadju? = 他甚麼時候要去山上？
  - anema azua? = 那是什麼？
  - rutjemalupung timadju? = 她常常戴帽子嗎？
  - padjalim azua timadju? = 那個人很勤勞嗎？
  - tjengelay timadju tazua qadupu? = 她喜歡那本書嗎？
  - maya taqed! = 別睡覺！
  - maya semuqeljev! = 別開門！
  - maya madraudraw tu sematjumaq a kaiv! = 別忘記回來吃晚飯喔！
  - ka ljesasaw ti kina, na tjemalupung timadju? = 媽媽出去的時候，她戴著帽子嗎？
  - ka ljesasaw ti kama, na masiljinay timadju? = 爸爸出去的時候，他帶著雨傘嗎？
  - ini, vuday azua. = 不，那是玉米。
  - ini, kuisang timadju. = 不，他是醫生。
  - ini, ku kina timadju. = 不，她是我媽媽。
  - ini, inika sinsi timadju. = 不，他不是老師。
  - ini, inika ku sinsi timadju. = 不，她不是我的老師。
  - ini, inika sadjelung azua qiladjan. = 不，那張椅子不重。
  - ini, inika qaca ku kina. = 不，我媽媽不高。
  - ini, inika ladruq a ku quvalj. = 不，我的頭髮不長。
  - maya kiljavaran! = 別說話！
  - maya pasuvililj! = 別遲到！
  - a sinipapungadan tu paiwan a caucau nuaya = 排湾族的名称是经过研究而定的
  - sikamasan musalj a kinatjuruvu ta caucau na sekacalisian = 原住民族十六族中之第二大族
  - a kinaizuavan nua paiwan a caucau i pasa tjalja navanavaljan ta taiwan = 排湾族分布于台湾南端
  - a supu nua paiwan a caucau i taiwan = 排湾族在台湾的人口
  - tjara izua tu 15,000 a caucauan i valangaw = 台东约有15000人
  - mapavalivalit anga a tja sengesengan = 我们的工作方式已经改变
  - a kai nuaya, sika maqati itjen a makakeljakeljang = 语言让我们能够互相了解
  - aicu a kai, mavan a sinipadjadjaljudjaljun nua caucau = 这个语言是人们传承下来的
  - sikiljavaranan a kai = 我们说的语言
  - a i kacedas a sepaiwan = 东排湾族
  - vangaw vangaw qemudalj nungida? = 天窗呀！何時會下雨呢？
  - pacunu a sasiq a mangtjengtjez = 你看吧！螞蟻正爬進來
  - mangetjez nungida? = 何時會來？
  - kemaljava ti tjuku? = 是秋姑在等待？
  - kemaljava ta qudalj a kivangavangavang = 等著下雨去玩耍
  - qemudjaqudjalj anga = 已經在下雨了
  - ari ari kipaqudjalji = 走吧！我們一起去淋雨
  - masalu aravac ta tja kiljivak = 非常感謝大家的愛
  - a nia kama ti vasakalang lja tepiq = 我們的父親vasakalang.tepiq
  - izua tu cengceng anga cavilj tu 80 a cavilj = 欣然的要歡度80歲的生日
  - imaza a uri tjemumalj sa djemaulj tjanuitjen = 我們在此相聚同樂
  - ulja kinisenayan mun nua na qemati = 願神祝福您們
  - na seminamalji aravac a vinarungan na payuan = 排灣族人的想法非常微妙
  - tjengelay a pakacadja a pakeljang ta vinarungan = 想要清楚地表达内心的意思
  - tjakudain! nu tisun a suqelam tjanuaken = 我該怎麼做呢我的愛人
  - imaza i niuzilan a qinaljan = 在紐西蘭的部落
  - izua za milimilingan na kacalisian = 有一个原住民的传说故事
  - maqati timadju a masan papamaw a caucau = 他能成为半神人
  - a kinaizuavan nua paiwan a caucau i pasa talja navanavaljan ta taiwan = 排湾族分布于台湾南端
  - pida matu naicu a kakedrian? = 這個小孩有幾隻狗？
  - ini, inika ladroq a ku quvalj. = 不，我的頭髮不長。
  - a sopu nua paiwan a caucau i taiwan = 排湾族在台湾的人口
  - maya kilavaran! = 別說話！
  - takudain! nu tisun a suqelam tjanuaken = 我該怎麼做呢我的愛人
  - vapavalivalit anga a tja sengesengan = 我们的工作方式已经改变
  - su sinsi timadu? = 她是你的老師嗎？
  - pida so qadupu? = 你有幾本書？
  - imaza a ori tjemumal sa demaulj tjanuitjen = 我們在此相聚同樂
  - ka lesasaw ti kama, na masiljinay timadjo? = 爸爸出去的時候，他帶著雨傘嗎？
  - tengelay a pakacada a pakeljang ta vinarungan = 想要清楚地表达内心的意思
  - inoan a su kama? = 你爸爸在哪裡？
  - sinsi timadu? = 他是老師嗎？
  - imaza i niuzilan a qinalan = 在紐西蘭的部落
  - ini, inika sadilung azua qiladjan. = 不，那張椅子不重。
  - pida anga cavilj na so kama? = 你爸爸幾歲？
  - vangetjez nungida? = 何時會來？
  - maya pitival! = 別浪費！
  - mapida mon a taqumaqanan? = 你們家有幾個人？
  - ini, ku kina timadu. = 不，她是我媽媽。
  - vasalu aravac ta ta kilivak = 非常感謝大家的愛
  - maya taqid! = 別睡覺！
  - kemalava ta qudalj a kivangavangavang = 等著下雨去玩耍
  - qimudaqudjalj anga = 已經在下雨了
  - maqati timadu a masan papamaw a caucau = 他能成为半神人
  - anema zoa? = 那是什麼？
  - ari ari kipaqodjalji = 走吧！我們一起去淋雨

### 回覆格式（嚴格遵守）

每次回覆必須使用以下結構，用 ``` 和標記分開：

<think>
[推理過程]
1. 用戶說了什麼：嘗試在上述排灣語句子清單中找到最接近的匹配
2. 匹配結果：如果有匹配，寫出匹配的原文和中文意思；如果沒有匹配，寫「無匹配」
3. 回覆策略：根據匹配結果決定如何回覆
4. 信心度：高/中/低（你是否確定自己的理解是正確的）
</think>

排灣語部分（1-2句）
---
中文翻譯與解釋

### 重要規則
- **如果找不到匹配且信心度低，必須在回覆中誠實說「vuvu 不太確定你說的排灣語是什麼意思呢」**
- **不要猜測或編造排灣語詞彙**
- **不要把中文直接音譯成排灣語**

### 特殊場景處理
1. 如果用戶說了不標準的排灣語：
   - 先用排灣語回應他們的意圖
   - 然後溫柔地用中文指出正確的說法
   - 例如：「ai~~ 你是想說 'masalu' 對不對？意思是謝謝喔！」

2. 如果用戶問排灣族文化：
   - 用排灣語開頭
   - 然後用中文詳細講述文化故事

3. 如果用戶用中文打招呼：
   - 溫柔地鼓勵他們試著用排灣語打招呼
   - 例如：「孫子啊，來，跟 vuvu 說一次 'tjanu en'——就是你好嗎的意思！」

4. 如果不知道答案：
   - 誠實地說「vuvu 不太確定呢」，不要編造排灣語詞彙
"""


# ============================================
# VuvuService — vuvu 對話服務
# ============================================

class VuvuService:
    """排灣族 AI 語伴服務"""

    def __init__(self, model: str = MODEL_DEFAULT):
        self.model = model
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def chat(self, user_input: str) -> str:
        """
        發送訊息給 vuvu，取得回覆（向後兼容，只回傳最終回覆）
        """
        result = self.chat_with_thinking(user_input)
        return result["reply"]

    def chat_with_thinking(self, user_input: str) -> dict:
        """
        發送訊息給 vuvu，取得推理過程 + 回覆

        Args:
            user_input: 用戶輸入的文字

        Returns:
            {
                "thinking": "推理過程文字",
                "reply": "最終回覆文字",
                "raw": "模型原始輸出"
            }
        """
        # 加入用戶訊息
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        try:
            # 呼叫智譜 API
            response = client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.5,        # 降低溫度，減少幻覺
                max_tokens=500,
                top_p=0.85,
            )

            raw = response.choices[0].message.content

            # 解析結構化輸出
            thinking, reply = self._parse_response(raw)

            # 記錄最終回覆到歷史（不記錄 thinking，節省 context）
            self.conversation_history.append({
                "role": "assistant",
                "content": reply
            })

            return {
                "thinking": thinking,
                "reply": reply,
                "raw": raw,
            }

        except Exception as e:
            # 發生錯誤時，移除最後加入的用戶訊息
            self.conversation_history.pop()
            error_msg = f"[錯誤] 無法連接 vuvu：{e}"
            return {"thinking": "", "reply": error_msg, "raw": error_msg}

    def _parse_response(self, raw: str) -> tuple:
        """
        解析模型輸出，分離推理過程和最終回覆

        格式預期：
            🤔\n[推理過程]\n.EqualTo\n\n[最終回覆]

        如果沒有結構化標記，整段當作回覆
        """
        thinking = ""
        reply = raw

        # 嘗試匹配 🤔...EqualTo 結構
        import re
        think_match = re.search(r'🤔\s*(.*?)\s*Equal_To', raw, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            reply = raw[think_match.end():].strip()
        else:
            # 備選: 匹配 ```think ... ``` 結構
            think_match2 = re.search(r'```think\s*(.*?)\s*```', raw, re.DOTALL)
            if think_match2:
                thinking = think_match2.group(1).strip()
                reply = raw[:think_match2.start()] + raw[think_match2.end():]
                reply = reply.strip()

        return thinking, reply

    def reset(self):
        """重置對話歷史"""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def get_history(self) -> list:
        """取得對話歷史（不含 system prompt）"""
        return self.conversation_history[1:]

    def switch_model(self, model: str):
        """切換模型"""
        self.model = model


# ============================================
# 終端機測試介面
# ============================================

def main():
    """終端機文字對話測試"""
    print("=" * 50)
    print("  語聲同行 競賽版 — vuvu Maliq 文字對話測試")
    print(f"  模型: {MODEL_DEFAULT}")
    print("=" * 50)
    print()
    print("指令:")
    print("  /reset  — 重置對話")
    print("  /model  — 切換模型 (flash/plus)")
    print("  /history — 顯示對話歷史")
    print("  /quit   — 離開")
    print()
    print("試著跟 vuvu 說：「masalu」或「你好」")
    print("-" * 50)

    vuvu = VuvuService(model=MODEL_DEFAULT)

    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nvuvu: Pacunan aken! （再見了！）")
            break

        if not user_input:
            continue

        # 指令處理
        if user_input == "/quit":
            print("\nvuvu: Pacunan aken, sanga! （再見了，孩子！）")
            break
        elif user_input == "/reset":
            vuvu.reset()
            print("✅ 對話已重置")
            continue
        elif user_input == "/history":
            history = vuvu.get_history()
            for msg in history:
                role = "🧑" if msg["role"] == "user" else "👵"
                print(f"{role} {msg['content'][:100]}")
            continue
        elif user_input == "/model":
            current = "flash" if vuvu.model == MODEL_FAST else "plus"
            new = "plus" if current == "flash" else "flash"
            vuvu.switch_model(MODEL_PLUS if new == "plus" else MODEL_FAST)
            print(f"✅ 模型切換: {current} → {new}")
            continue

        # 正常對話
        print("\n👵 vuvu: ", end="", flush=True)
        reply = vuvu.chat(user_input)
        print(reply)


if __name__ == "__main__":
    main()
