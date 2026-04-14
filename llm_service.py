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
問候與日常:
- masalu = 謝謝
- nanguaq = 好
- inu = 哪裡
- tima = 誰
- pacunan = 再見
- maljimalji = 很漂亮 / 排灣族八年祭
- tjanu en = 你好嗎
- madjadjun = 很高興

家人稱謂:
- vuvu = 祖父母
- kina = 母親
- ama = 父親
- kaka = 哥哥/姐姐
- ali = 弟弟/妹妹

否定詞:
- maya = 不要（禁止意圖）
- ini / inika = 不是（事實否定）

文化詞彙:
- palakuan = 男子會所
- sevung = 頭目
- taqaljan / taqaljan = 青年會所
- vecik = 排灣族手文 / 刺繡
- ljemiyuljuy = 祭典
- kavulungan = 山

### 回覆格式
每次回覆請使用以下格式：

排灣語部分（1-2句）
---
中文翻譯與解釋

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
        發送訊息給 vuvu，取得回覆

        Args:
            user_input: 用戶輸入的文字（可以是排灣語或中文）

        Returns:
            vuvu 的回覆文字
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
                temperature=0.7,        # 保持一些創意，但不會太隨機
                max_tokens=300,          # 回覆長度適中
                top_p=0.9,
            )

            reply = response.choices[0].message.content

            # 記錄回覆到歷史
            self.conversation_history.append({
                "role": "assistant",
                "content": reply
            })

            return reply

        except Exception as e:
            # 發生錯誤時，移除最後加入的用戶訊息（避免歷史污染）
            self.conversation_history.pop()
            return f"[錯誤] 無法連接 vuvu：{e}"

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
