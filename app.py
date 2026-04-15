"""
app.py
語聲同行 2.0 — Gradio Web Demo（RAG + TTS 版）

語音對話介面: 錄音 → ASR → RAG 檢索 → LLM → 文字 + 語音回覆

作者: Backend_Dev
日期: 2026-04-15
用途: 飛書 AI 校園挑戰賽 — Web 介面
"""

import gradio as gr
from llm_service import VuvuService
from tts_service import synthesize

# ============================================
# 全域 vuvu 實例（保持對話上下文）
# ============================================

vuvu = VuvuService(use_rag=True)


# ============================================
# 核心處理函式
# ============================================

def process_audio(audio_path):
    """處理錄音: ASR → RAG → LLM → 文字 + 語音回覆"""
    if audio_path is None:
        return "", "", None, "請先錄音或上傳音檔，vuvu 在等你說話呢！"

    # Step 1: ASR
    from voice_chat import recognize_audio
    asr_result = recognize_audio(audio_path)

    if asr_result.get("error"):
        return "", "", None, f"❌ ASR 錯誤: {asr_result['error']}"

    recognized = asr_result.get("text", "")
    confidence = asr_result.get("confidence", 0.0)
    raw_text = asr_result.get("raw_text", "")
    logic = asr_result.get("logic", "")

    if not recognized:
        return "", "", None, "🔇 辨識結果為空，請再錄一次試試！"

    # ASR 資訊展示
    display_parts = [f"🎤 你說的排灣語: **{recognized}**"]
    if raw_text and raw_text != recognized:
        display_parts.append(f"📝 原始辨識: {raw_text}")
    if logic:
        display_parts.append(f"🔧 校正: {logic}")
    display_parts.append(f"📊 信心度: {confidence:.0%}")

    recognized_display = "\n".join(display_parts)

    # Step 2: LLM + RAG
    llm_result = vuvu.chat_with_thinking(recognized)
    reply = llm_result["reply"]

    # Step 3: TTS 語音合成
    audio_path_out = synthesize(reply, engine="macos")

    # 思考過程
    thinking_display = ""
    if llm_result.get("thinking"):
        thinking_display = f"🧠 **vuvu 的思考:**\n{llm_result['thinking']}"

    return recognized_display, thinking_display, audio_path_out, reply


def process_text(text):
    """純文字對話: RAG → LLM → 文字 + 語音回覆"""
    if not text.strip():
        return "", "", None, "請輸入文字跟 vuvu 聊天！"

    llm_result = vuvu.chat_with_thinking(text.strip())
    reply = llm_result["reply"]

    # TTS
    audio_path_out = synthesize(reply, engine="macos")

    thinking_display = ""
    if llm_result.get("thinking"):
        thinking_display = f"🧠 **vuvu 的思考:**\n{llm_result['thinking']}"

    return "", thinking_display, audio_path_out, reply


def reset_chat():
    """重置對話"""
    vuvu.reset()
    return "", "", None, "", "✅ 對話已重置，vuvu 重新開始了！"


# ============================================
# Gradio 介面
# ============================================

SCENARIOS = [
    ("🏠 問候", "tjanu en"),
    ("🙏 謝謝", "masalu"),
    ("👍 好", "nanguaq"),
    ("👋 再見", "pacunan"),
    ("📖 說故事", "vuvu，可以講一個排灣族的故事嗎？"),
    ("📚 學單字", "vuvu，教我一些排灣語的日常用語"),
]


with gr.Blocks(
    title="語聲同行 2.0 — 排灣族語 AI 語伴",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🌾 語聲同行 2.0 — 排灣族語 AI 語伴
        ### 跟 vuvu Maliq 用排灣語聊天吧！
        """
    )

    mode_label = "🔍 RAG + 🔊 TTS" if vuvu.use_rag else "📝 Legacy"
    gr.Markdown(f"錄音或輸入排灣語，AI vuvu 會用排灣語 + 中文回覆你。({mode_label})")

    with gr.Row():
        with gr.Column(scale=1):
            # 語音輸入
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 錄音或上傳音檔",
            )

            gr.Markdown("**⚡ 快速場景:**")
            with gr.Row():
                scenario_btns = []
                for label, _ in SCENARIOS[:3]:
                    btn = gr.Button(label, size="sm")
                    scenario_btns.append(btn)
            with gr.Row():
                for label, _ in SCENARIOS[3:]:
                    btn = gr.Button(label, size="sm")
                    scenario_btns.append(btn)

            # 文字輸入
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
            # vuvu 思考過程
            thinking_output = gr.Markdown(
                label="🧠 vuvu 的思考過程",
                value="",
            )
            # vuvu 語音回覆
            tts_output = gr.Audio(
                label="🔊 vuvu 的聲音",
                type="filepath",
                autoplay=True,
            )
            # vuvu 文字回覆
            reply_output = gr.Markdown(
                label="👵 vuvu Maliq 的回覆",
                value="ai~~ 孫子孫女，跟 vuvu 說話吧！",
            )

    # ============================================
    # 事件綁定
    # ============================================

    output_fields = [recognized_output, thinking_output, tts_output, reply_output]

    submit_audio.click(fn=process_audio, inputs=[audio_input], outputs=output_fields)
    submit_text.click(fn=process_text, inputs=[text_input], outputs=output_fields)
    reset_btn.click(
        fn=reset_chat,
        outputs=[audio_input, text_input, thinking_output, tts_output, reply_output],
    )

    # 場景按鈕
    for btn, (label, text) in zip(scenario_btns, SCENARIOS):
        btn.click(
            fn=lambda t=text: process_text(t),
            inputs=[],
            outputs=output_fields,
        )


# ============================================
# 啟動
# ============================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
