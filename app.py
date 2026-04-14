"""
app.py
語聲同行 競賽版 — Gradio Web Demo

語音對話介面: 錄音 → ASR → LLM → 文字回覆

作者: Backend_Dev (OpenClaw Agent)
日期: 2026-04-15
用途: 飛書 AI 校園挑戰賽 — Phase 3 Web 介面
"""

import gradio as gr
from voice_chat import recognize_audio, VuvuService

# ============================================
# 全域 vuvu 實例（保持對話上下文）
# ============================================

vuvu = VuvuService()


# ============================================
# 核心處理函式
# ============================================

def process_audio(audio_path):
    """
    處理錄音: ASR → LLM → 回覆

    Args:
        audio_path: Gradio 錄音元件傳入的音檔路徑

    Returns:
        (recognized_text, reply_text)
    """
    if audio_path is None:
        return "", "請先錄音或上傳音檔，vuvu 在等你說話呢！"

    # Step 1: ASR
    asr_result = recognize_audio(audio_path)

    if asr_result.get("error"):
        return "", f"❌ ASR 錯誤: {asr_result['error']}"

    recognized = asr_result.get("text", "")
    confidence = asr_result.get("confidence", 0.0)
    raw_text = asr_result.get("raw_text", "")
    logic = asr_result.get("logic", "")

    if not recognized:
        return "", "🔇 辨識結果為空，請再錄一次試試！"

    # 組合辨識資訊
    display_parts = [f"🎤 你說的排灣語: **{recognized}**"]
    if raw_text and raw_text != recognized:
        display_parts.append(f"📝 原始辨識: {raw_text}")
    if logic:
        display_parts.append(f"🔧 校正: {logic}")
    display_parts.append(f"📊 信心度: {confidence:.0%}")

    recognized_display = "\n".join(display_parts)

    # Step 2: LLM
    reply = vuvu.chat(recognized)

    return recognized_display, reply


def process_text(text):
    """純文字對話（備用模式）"""
    if not text.strip():
        return "", "請輸入文字跟 vuvu 聊天！"

    reply = vuvu.chat(text.strip())
    return "", reply


def reset_chat():
    """重置對話"""
    vuvu.reset()
    return "", "", "✅ 對話已重置，vuvu 重新開始了！"


# ============================================
# Gradio 介面
# ============================================

# 預設場景按鈕
SCENARIOS = [
    ("🏠 問候", "tjanu en"),
    ("🙏 謝謝", "masalu"),
    ("👍 好", "nanguaq"),
    ("👋 再見", "pacunan"),
    ("📖 說故事", "vuvu，可以講一個排灣族的故事嗎？"),
    ("📚 學單字", "vuvu，教我一些排灣語的日常用語"),
]


def load_scenario(btn_text):
    """載入預設場景"""
    # btn_text 格式: "🏠 問候" → 取得對應文字
    for label, text in SCENARIOS:
        if label in btn_text or btn_text in label:
            return text
    return ""


with gr.Blocks(title="語聲同行 — 排灣族語 AI 語伴") as demo:

    gr.Markdown(
        """
        # 🌾 語聲同行 — 排灣族語 AI 語伴
        ### 跟 vuvu Maliq 用排灣語聊天吧！
        """,
        elem_classes=["title"]
    )

    gr.Markdown(
        "錄音或輸入排灣語，AI vuvu 會用排灣語 + 中文回覆你。",
        elem_classes=["subtitle"]
    )

    with gr.Row():
        with gr.Column(scale=1):
            # 語音輸入
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 錄音或上傳音檔",
                show_label=True,
            )

            gr.Markdown("**⚡ 快速場景（點擊自動填入）:**")
            with gr.Row():
                scenario_btns = []
                for i, (label, _) in enumerate(SCENARIOS[:3]):
                    btn = gr.Button(label, size="sm")
                    scenario_btns.append(btn)
            with gr.Row():
                for i, (label, _) in enumerate(SCENARIOS[3:]):
                    btn = gr.Button(label, size="sm")
                    scenario_btns.append(btn)

            # 文字輸入（備用）
            text_input = gr.Textbox(
                label="⌨️ 或直接輸入文字",
                placeholder="輸入排灣語或中文...",
                lines=2,
            )

            with gr.Row():
                submit_audio = gr.Button("🎙️ 送出語音", variant="primary")
                submit_text = gr.Button("📝 送出文字")
                reset_btn = gr.Button("🔄 重置對話")

        with gr.Column(scale=1):
            # ASR 結果
            recognized_output = gr.Markdown(
                label="🎤 辨識結果",
                value="等待輸入...",
            )
            # vuvu 回覆
            reply_output = gr.Markdown(
                label="👵 vuvu Maliq 的回覆",
                value="ai~~ 孫子孫女，跟 vuvu 說話吧！",
            )

    # ============================================
    # 事件綁定
    # ============================================

    # 語音送出
    submit_audio.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[recognized_output, reply_output],
    )

    # 文字送出
    submit_text.click(
        fn=process_text,
        inputs=[text_input],
        outputs=[recognized_output, reply_output],
    )

    # 重置
    reset_btn.click(
        fn=reset_chat,
        outputs=[audio_input, text_input, reply_output],
    )

    # 場景按鈕
    for btn, (label, text) in zip(scenario_btns, SCENARIOS):
        btn.click(
            fn=lambda t=text: process_text(t),
            inputs=[],
            outputs=[recognized_output, reply_output],
        )


# ============================================
# 啟動
# ============================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 本地測試用；錄 Demo 時改 True
    )
