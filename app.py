"""
app.py
語聲同行 2.0 — 排灣語智能教學系統

定位：排灣語教學工具（不是聊天機器人）

核心功能：
1. 發音評測：學習者說排灣語 → ASR → 音韻比對 → 評分 + 分析
2. 語法解釋：點擊任意句子 → 詞彙拆解 + 語法分析 + 學習建議
3. 綴詞結構可視化：展示排灣語黏著語的綴詞體系
4. 測驗系統：基於語料庫的選擇題測驗

作者: Backend_Dev
日期: 2026-04-15
"""

import json
import gradio as gr
from pathlib import Path

# 載入模組
from modules.asr_evaluator import (
    PhonemeEngine, AffixAnalyzer, IntentClassifier,
    evaluate_pronunciation, evaluate_text,
)
from modules.grammar_explainer import explain_sentence, generate_quiz, explain_affix
from tts_service import get_paiwan_audio_for_text, list_available_paiwan_audio

# ============================================
# 載入語料庫
# ============================================

CORPUS_PATH = Path(__file__).parent / "modules" / "paiwan_corpus.json"
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    CORPUS = json.load(f)

# 展平語料供檢索
ALL_SENTENCES = []
for cat_id, cat in CORPUS["categories"].items():
    for s in cat["sentences"]:
        s["category_id"] = cat_id
        s["category_name"] = cat["name"]
        ALL_SENTENCES.append(s)


# ============================================
# Tab 1: 發音評測
# ============================================

def do_pronunciation_eval(audio_path, sentence_idx):
    """發音評測"""
    if sentence_idx is None or sentence_idx >= len(ALL_SENTENCES):
        return "請先選擇一個句子", "", ""
    
    target = ALL_SENTENCES[sentence_idx]
    
    if audio_path:
        result = evaluate_pronunciation(
            audio_path, target["paiwan"], target["chinese"]
        )
    else:
        return "請先錄音", "", ""
    
    if result.error:
        return f"❌ {result.error}", "", ""
    
    # 評測報告
    grade_emoji = {"Perfect": "🌟", "Excellent": "⭐", "Good": "✅", "Fair": "🤔", "Try Again": "❌"}
    emoji = grade_emoji.get(result.grade, "")
    
    report = f"""## {emoji} 發音評測結果

**目標句子**: {target['paiwan']}（{target['chinese']}）
**你說的**: {result.recognized_text}
**分數**: **{result.score}/100**（{result.grade}）

### 音韻分析
- 音韻相似度: {result.similarity:.1%}
- Levenshtein 距離: {result.levenshtein_distance if hasattr(result, 'levenshtein_distance') else 'N/A'}
- 精確匹配: {'✅ 是' if result.exact_match else '❌ 否'}
"""
    if result.corrections:
        report += f"- 校正規則: {', '.join(result.corrections)}\n"
    
    # 綴詞分析
    affix_text = format_affix_analysis(result.affix_analysis)
    
    # 意圖
    intent_text = ""
    if result.intent and result.intent["intent"] != "unknown":
        intent_text = f"**意圖類別**: {result.intent['intent']}（置信度 {result.intent['confidence']:.0%}）"
    
    return report, affix_text, intent_text


def format_affix_analysis(analysis: dict) -> str:
    if not analysis or not analysis.get("affixes_found"):
        return "此句為基礎句型，無明顯綴詞標記。"
    
    lines = ["### 綴詞結構分析\n"]
    for affix in analysis["affixes_found"]:
        lines.append(f"- **前綴「{affix['prefix']}」**: {affix['label']} — {affix['description']}")
    lines.append(f"\n{analysis['structure_description']}")
    return "\n".join(lines)


# ============================================
# Tab 2: 語法解釋
# ============================================

def do_grammar_explain(sentence_idx):
    """語法解釋"""
    if sentence_idx is None or sentence_idx >= len(ALL_SENTENCES):
        return "請先選擇一個句子"
    
    target = ALL_SENTENCES[sentence_idx]
    result = explain_sentence(
        target["paiwan"], target["chinese"], target.get("grammar_note", "")
    )
    
    if not result["success"]:
        return f"解釋生成失敗，請重試。\n\n**原句**: {target['paiwan']} = {target['chinese']}\n**語法提示**: {target.get('grammar_note', '無')}"
    
    data = result["data"]
    
    output = f"""## 語法解析：{target['paiwan']}
**翻譯**: {target['chinese']}

### 詞彙拆解
"""
    for w in data.get("word_breakdown", []):
        output += f"- **{w['word']}** ({w['type']}) — {w['meaning']}\n"
    
    output += f"""
### 語法說明
{data.get('grammar_explanation', '')}

### 學習建議
💡 {data.get('learning_tip', '')}

### 舉一反三
{data.get('related_pattern', '')}
"""
    
    if target.get("grammar_note"):
        output += f"\n### 語料庫註解\n> {target['grammar_note']}"
    
    return output


# ============================================
# Tab 3: 語料庫瀏覽
# ============================================

def build_corpus_display():
    """生成語料庫瀏覽內容"""
    lines = ["# 📖 排灣語語料庫\n"]
    
    for cat_id, cat in CORPUS["categories"].items():
        lines.append(f"## {cat['icon']} {cat['name']}\n")
        lines.append(f"*{cat['description']}*\n")
        lines.append("| 排灣語 | 中文 | 語法註解 |")
        lines.append("|--------|------|----------|")
        for s in cat["sentences"]:
            note = s.get("grammar_note", "")[:30] + "..." if len(s.get("grammar_note", "")) > 30 else s.get("grammar_note", "")
            lines.append(f"| {s['paiwan']} | {s['chinese']} | {note} |")
        lines.append("")
    
    # 綴詞體系
    lines.append("---\n## 🔧 排灣語綴詞體系\n")
    for section_name, section in CORPUS["grammar_guide"].items():
        lines.append(f"### {section['title']}\n")
        lines.append("| 排灣語 | 中文 | 說明 |")
        lines.append("|--------|------|------|")
        for entry in section["entries"]:
            p = entry.get("paiwan", entry.get("marker", ""))
            c = entry.get("chinese", entry.get("label", ""))
            n = entry.get("note", entry.get("description", entry.get("example", "")))
            lines.append(f"| {p} | {c} | {n} |")
        lines.append("")
    
    return "\n".join(lines)


# ============================================
# Tab 4: 測驗
# ============================================

current_quiz = {"questions": [], "answers": []}

def do_generate_quiz(category_name):
    """生成測驗題"""
    # 找到對應類別
    cat = None
    for cat_id, c in CORPUS["categories"].items():
        if c["name"] == category_name:
            cat = c
            break
    
    if not cat:
        return "找不到此類別", ""
    
    result = generate_quiz(cat["name"], cat["sentences"])
    
    if not result["success"]:
        return f"測驗生成失敗: {result.get('error', '未知錯誤')}", ""
    
    quiz = result["data"].get("quiz", [])
    current_quiz["questions"] = quiz
    current_quiz["answers"] = [q["answer"] for q in quiz]
    
    output = f"## 📝 {category_name} 測驗\n\n"
    for i, q in enumerate(quiz):
        output += f"**第 {i+1} 題**: {q['question']}\n\n"
        for j, opt in enumerate(q["options"]):
            output += f"- {chr(65+j)}. {opt}\n"
        output += "\n"
    
    output += "---\n*回答後點擊「查看答案」*\n"
    return output, ""


def do_check_answers(answer_text):
    """檢查答案"""
    if not current_quiz["questions"]:
        return "請先生成測驗題"
    
    output = "## 📊 測驗結果\n\n"
    try:
        user_answers = [int(a.strip()) for a in answer_text.split(",")]
    except:
        return "請用逗號分隔答案編號，例如：0,2,1（從 0 開始）"
    
    correct = 0
    for i, q in enumerate(current_quiz["questions"]):
        ans = current_quiz["answers"][i]
        user_ans = user_answers[i] if i < len(user_answers) else -1
        is_correct = user_ans == ans
        
        if is_correct:
            correct += 1
        
        emoji = "✅" if is_correct else "❌"
        output += f"{emoji} 第 {i+1} 題: {q['question']}\n"
        output += f"   正確答案: {chr(65+ans)}. {q['options'][ans]}\n"
        if not is_correct and i < len(user_answers):
            output += f"   你的答案: {chr(65+user_ans)}. {q['options'][user_ans]}\n"
        output += f"   💡 {q['explanation']}\n\n"
    
    output += f"**得分**: {correct}/{len(current_quiz['questions'])}"
    return output


# ============================================
# Gradio 介面
# ============================================

with gr.Blocks(
    title="語聲同行 2.0 — 排灣語智能教學系統",
) as demo:
    
    gr.Markdown("""
    # 🌾 語聲同行 2.0 — 排灣語智能教學系統
    ### 低資源語言的 AI 輔助教學框架
    
    *ASR 微調 + 音韻規則引擎 + 綴詞結構分析 + 語法解釋*
    """)
    
    # 句子選擇器（全域）
    sentence_choices = [
        f"{s['paiwan']}（{s['chinese']}）" for s in ALL_SENTENCES
    ]
    
    # ==========================================
    # Tab 1: 發音評測
    # ==========================================
    with gr.Tab("🎤 發音評測"):
        gr.Markdown("""
        ### 發音評測系統
        選擇一個排灣語句子，錄下你的發音，系統會：
        1. ASR 辨識你說的排灣語（Whisper-tiny + LoRA 微調）
        2. 音韻規則校正（處理 tj/t、m/v 等排灣語特有音變）
        3. 計算發音分數（Levenshtein 距離 + 音韻相似度）
        4. 分析綴詞結構
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                eval_sentence = gr.Dropdown(
                    choices=sentence_choices,
                    label="選擇要練習的句子",
                    type="index",
                )
                eval_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="🎤 錄下你的發音",
                )
                # 正確發音播放
                correct_audio = gr.Audio(
                    label="🔊 聽聽正確發音（TTS 合成）",
                    type="filepath",
                    visible=True,
                )
                
                def get_correct_pronunciation_audio(sentence_idx):
                    if sentence_idx is None or sentence_idx >= len(ALL_SENTENCES):
                        return None
                    target = ALL_SENTENCES[sentence_idx]
                    return get_paiwan_audio_for_text(target["paiwan"])
                
                eval_sentence.change(
                    fn=get_correct_pronunciation_audio,
                    inputs=[eval_sentence],
                    outputs=[correct_audio],
                )
                
                eval_btn = gr.Button("📊 開始評測", variant="primary")
            
            with gr.Column(scale=1):
                eval_result = gr.Markdown(label="評測結果")
                eval_affix = gr.Markdown(label="綴詞分析")
                eval_intent = gr.Markdown(label="意圖分類")
        
        eval_btn.click(
            fn=do_pronunciation_eval,
            inputs=[eval_audio, eval_sentence],
            outputs=[eval_result, eval_affix, eval_intent],
        )
    
    # ==========================================
    # Tab 2: 語法解釋
    # ==========================================
    with gr.Tab("📖 語法解釋"):
        gr.Markdown("""
        ### 語法結構解析
        選擇一個句子，系統會分析：
        - 詞彙拆解（每個詞的意思和詞性）
        - 語法結構（句型分析）
        - 學習建議（記憶技巧）
        - 舉一反三（類似句型）
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                grammar_sentence = gr.Dropdown(
                    choices=sentence_choices,
                    label="選擇要學習的句子",
                    type="index",
                )
                grammar_btn = gr.Button("🔍 分析語法", variant="primary")
            
            with gr.Column(scale=2):
                grammar_result = gr.Markdown()
        
        grammar_btn.click(
            fn=do_grammar_explain,
            inputs=[grammar_sentence],
            outputs=[grammar_result],
        )
    
    # ==========================================
    # Tab 3: 語料庫
    # ==========================================
    with gr.Tab("📚 語料庫瀏覽"):
        gr.Markdown(build_corpus_display())
    
    # ==========================================
    # Tab 3.5: 語音播放
    # ==========================================
    with gr.Tab("🔊 排灣語語音"):
        gr.Markdown("""
        ### 排灣語 TTS 語音合成
        以下音檔來自 XTTS v2 微調模型（零樣本語音克隆），使用排灣族長老錄音作為參考音色。
        """)
        
        tts_available = list_available_paiwan_audio()
        
        gr.Markdown(f"**📊 詞彙音檔**: {len(tts_available['words'])} 個 | **句子音檔**: {len(tts_available['sentences'])} 個")
        
        with gr.Accordion("🎵 詞彙發音（點擊展開）", open=False):
            word_choices = tts_available["words"]
            tts_word_dropdown = gr.Dropdown(
                choices=word_choices,
                label="選擇排灣語詞彙",
                value=word_choices[0] if word_choices else None,
            )
            tts_word_btn = gr.Button("▶️ 播放", variant="primary")
            tts_word_audio = gr.Audio(label="排灣語語音", type="filepath")
            
            def play_paiwan_word(word):
                from tts_service import get_paiwan_audio
                return get_paiwan_audio(word)
            
            tts_word_btn.click(
                fn=play_paiwan_word,
                inputs=[tts_word_dropdown],
                outputs=[tts_word_audio],
            )
        
        with gr.Accordion("🗣️ 句子發音（點擊展開）", open=True):
            sentence_choices = tts_available["sentences"]
            tts_sentence_dropdown = gr.Dropdown(
                choices=sentence_choices,
                label="選擇排灣語句子",
                value=sentence_choices[0] if sentence_choices else None,
            )
            tts_sentence_btn = gr.Button("▶️ 播放", variant="primary")
            tts_sentence_audio = gr.Audio(label="排灣語句子語音", type="filepath")
            
            def play_paiwan_sentence(sentence):
                from tts_service import get_paiwan_audio
                return get_paiwan_audio(sentence)
            
            tts_sentence_btn.click(
                fn=play_paiwan_sentence,
                inputs=[tts_sentence_dropdown],
                outputs=[tts_sentence_audio],
            )
        
        gr.Markdown("""
        ---
        **技術說明**：
        - 語音合成模型：XTTS v2（零樣本語音克隆）
        - 參考音色：排灣族長老真實錄音
        - 擴展詞彙表：+104 個排灣語 tokens
        - 合成詞彙數：50 個常用詞 + 8 個完整句子
        """)
    
    # ==========================================
    # Tab 4: 測驗
    # ==========================================
    with gr.Tab("📝 測驗"):
        gr.Markdown("### 排灣語測驗\n選擇類別，生成測驗題，測試你的學習成果。")
        
        with gr.Row():
            cat_choices = [cat["name"] for cat in CORPUS["categories"].values()]
            quiz_cat = gr.Dropdown(choices=cat_choices, label="選擇測驗類別")
            quiz_gen_btn = gr.Button("🎲 生成測驗", variant="primary")
        
        quiz_display = gr.Markdown()
        
        with gr.Row():
            quiz_answer = gr.Textbox(
                label="你的答案（填入題號，用逗號分隔。例如：0,2,1）",
                placeholder="0,2,1",
                lines=1,
            )
            quiz_check_btn = gr.Button("✅ 查看答案")
        
        quiz_result = gr.Markdown()
        
        quiz_gen_btn.click(
            fn=do_generate_quiz,
            inputs=[quiz_cat],
            outputs=[quiz_display, quiz_result],
        )
        quiz_check_btn.click(
            fn=do_check_answers,
            inputs=[quiz_answer],
            outputs=[quiz_result],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
