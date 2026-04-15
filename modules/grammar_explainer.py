"""
modules/grammar_explainer.py
排灣語語法解釋器

LLM 的正確用法：不生成排灣語，只做教學解釋
- 解釋語法結構
- 生成練習題
- 說明文化背景

所有排灣語內容 100% 來自語料庫，LLM 只負責中文解釋。

作者: Backend_Dev
日期: 2026-04-15
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

MODEL = "glm-4-flash"


def explain_sentence(paiwan: str, chinese: str, grammar_note: str = "") -> dict:
    """
    讓 LLM 解釋一句排灣語的語法結構
    
    注意：LLM 不生成排灣語，只做中文解釋
    """
    prompt = f"""你是一位排灣族語言學專家和教學助手。

請用中文解釋以下排灣語句子的語法結構，幫助學習者理解。

排灣語句子：{paiwan}
中文翻譯：{chinese}
{'語法提示：' + grammar_note if grammar_note else ''}

請用以下 JSON 格式回答（不要其他內容）：
{{
  "word_breakdown": [
    {{"word": "排灣語詞", "meaning": "中文意思", "type": "詞性（代名詞/動詞/疑問詞等）"}}
  ],
  "grammar_explanation": "用中文解釋這句話的語法結構（2-3句話）",
  "learning_tip": "給學習者的一個記憶技巧或注意事項",
  "related_pattern": "這個句型可以用來造哪些類似的句子？（只列出中文想法，不要造排灣語）"
}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        
        # 嘗試解析 JSON
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return {"success": True, "data": result, "raw": raw}
    
    except json.JSONDecodeError:
        return {"success": False, "data": None, "raw": raw}
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_quiz(category: str, sentences: list) -> dict:
    """
    根據語料庫句子生成測驗題
    
    LLM 只生成題目描述（中文），排灣語答案來自語料庫
    """
    # 把語料中的句子整理給 LLM
    sentence_list = "\n".join(
        f"- {s['paiwan']} = {s['chinese']}" 
        for s in sentences[:10]
    )
    
    prompt = f"""你是一位排灣語教學測驗設計師。

以下是「{category}」類別的排灣語句子：
{sentence_list}

請設計 3 道選擇題，測驗學習者對這些句子的理解。

規則：
- 題目用中文描述
- 選項中的排灣語必須 **只從上面的列表中選取**，不可編造
- 正確答案必須是列表中的句子

JSON 格式：
{{
  "quiz": [
    {{
      "question": "中文題目描述",
      "options": ["排灣語選項A", "排灣語選項B", "排灣語選項C", "排灣語選項D"],
      "answer": 0,
      "explanation": "中文解釋為什麼這是正確答案"
    }}
  ]
}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return {"success": True, "data": result, "raw": raw}
    except Exception as e:
        return {"success": False, "error": str(e)}


def explain_affix(prefix: str, analysis: dict) -> str:
    """根據綴詞分析結果生成教學解釋"""
    templates = {
        'tense': f"「{prefix}」是時態標記。排灣語通過前綴來標示動作的時間，而不是像中文用「了」「過」「將」這樣的助詞。",
        'negation': f"「{prefix}」是否定標記。排灣語有三種否定：ini（陳述否定「不」）、inika（屬性否定「不是」）、maya（禁止「不要」）。",
        'voice': f"「{prefix}」是語態標記。排灣語是黏著語，通過前綴改變動詞的語態和語意。",
    }
    prefix_type = analysis.get('type', 'voice')
    return templates.get(prefix_type, f"「{prefix}」是排灣語的綴詞標記。")
