#!/usr/bin/env python3
"""
dashboard.py — Multi-Agent 自主運行系統 Dashboard

啟動方式：
    streamlit run dashboard.py

功能：
    - 即時監控自主運行狀態
    - Agent 通訊圖
    - Plan 進度追蹤
    - Token 消耗統計
    - 歷史時間線
    - 手動啟動/停止任務

作者: 地陪
日期: 2026-05-12
"""

import streamlit as st
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 確保 import 路徑
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "agent_framework"))

# ============================================
# 頁面配置
# ============================================

st.set_page_config(
    page_title="🏔️ 語聲同行 Multi-Agent Dashboard",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# 數據讀取
# ============================================

STATE_PATH = PROJECT_ROOT / "agent_framework" / "storage" / "state"
SIGNAL_PATH = PROJECT_ROOT / "agent_framework" / "storage" / "STOP.signal"


def load_checkpoints():
    """載入所有 checkpoint"""
    checkpoints = []
    if not STATE_PATH.exists():
        return checkpoints
    for f in sorted(STATE_PATH.glob("ckpt_*.json")):
        try:
            data = json.loads(f.read_text())
            checkpoints.append(data)
        except Exception:
            pass
    return checkpoints


def load_latest_state():
    """載入最新 checkpoint 的 state"""
    latest_path = STATE_PATH / "latest.json"
    if not latest_path.exists():
        return None
    try:
        latest = json.loads(latest_path.read_text())
        ckpt_path = STATE_PATH / f"{latest['checkpoint_id']}.json"
        data = json.loads(ckpt_path.read_text())
        return data.get("state")
    except Exception:
        return None


def parse_iteration_results(results):
    """安全解析 iteration_results（可能是 str 或 dict）"""
    import ast as _ast
    parsed = []
    for r in results:
        if isinstance(r, dict):
            parsed.append(r)
        elif isinstance(r, str):
            try:
                obj = _ast.literal_eval(r)
                parsed.append(obj)
            except Exception:
                parsed.append({"loop": "?", "result": {"step": "", "status": "parse_error", "data": {}}})
        else:
            parsed.append({"loop": "?", "result": {"step": "", "status": "unknown", "data": {}}})
    return parsed


def get_step_info(parsed_result):
    """從解析後的 result 中提取步驟資訊（兼容 Plan 和 ReAct 結構）"""
    res = parsed_result.get("result", {})
    if not isinstance(res, dict):
        return {"agent": "?", "name": str(res)[:80], "id": "?", "status": "?", "reply": ""}
    
    # ---- ReAct 模式：{mode: "react", action: "...", reply: "..."} ----
    if res.get("mode") == "react":
        return {
            "agent": "orchestrator",
            "name": res.get("action", "ReAct 決策")[:80],
            "id": "react",
            "status": "react",
            "reply": res.get("reply", "")[:500],
        }
    
    status = res.get("status", "?")
    step_name = res.get("step", "?")
    
    # ---- Plan 模式：result.data.step 有完整資訊 ----
    data = res.get("data", {})
    if isinstance(data, dict):
        real_step = data.get("step", {})
        if isinstance(real_step, dict):
            return {
                "agent": real_step.get("agent", "?"),
                "name": real_step.get("name", step_name if isinstance(step_name, str) else "?"),
                "id": real_step.get("id", "?"),
                "status": status,
                "reply": data.get("reply", "")[:500],
            }
    
    # ---- step 本身就是 dict ----
    if isinstance(step_name, dict):
        return {
            "agent": step_name.get("agent", "?"),
            "name": step_name.get("name", "?"),
            "id": step_name.get("id", "?"),
            "status": status,
            "reply": "",
        }
    
    return {"agent": "?", "name": str(step_name)[:80], "id": "?", "status": status, "reply": ""}


def fmt_time(ts):
    """格式化時間戳"""
    if not ts:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def fmt_duration(seconds):
    """格式化時長"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}分鐘"
    else:
        return f"{seconds / 3600:.2f}小時"


# ============================================
# 繪圖：Agent 通訊圖（用 graphviz）
# ============================================

def draw_agent_graph(results, plan_data):
    """繪製 Agent 分層架構流程圖"""
    try:
        import graphviz
    except ImportError:
        st.warning("安裝 graphviz 可看通訊圖: `pip install graphviz`")
        return

    dot = graphviz.Digraph(comment="Agent 架構流程")
    dot.attr(rankdir="TB", bgcolor="transparent", dpi="72", size="7,6")
    dot.attr('graph', fontname='sans-serif')
    dot.attr('node', fontname='sans-serif')
    dot.attr('edge', fontname='sans-serif')

    # ── 第 1 層：主控 ──
    dot.node('main_loop', '🔄 MainLoop\n自主運行循環',
             shape='box', style='filled,rounded', fillcolor='#FFEAA7',
             fontcolor='#333', fontsize='10', width='2', height='0.7')

    # ── 第 2 層：Orchestrator ──
    dot.node('orchestrator', '🎯 Orchestrator\n理解意圖 → 分派任務 → 組織回覆',
             shape='box', style='filled,rounded,bold', fillcolor='#FF6B6B',
             fontcolor='white', fontsize='10', width='2.8', height='0.7')

    # ── 第 3 層：執行 Agent ──
    dot.node('knowledge', '🧠 Knowledge\n翻譯 · RAG搜尋\n知識圖譜 · 發音',
             shape='box', style='filled,rounded', fillcolor='#4ECDC4',
             fontcolor='white', fontsize='9', width='2', height='0.7')
    dot.node('teaching', '📚 Teaching\n學習計畫 · 詞彙推薦\n測驗生成 · 進度記錄',
             shape='box', style='filled,rounded', fillcolor='#45B7D1',
             fontcolor='white', fontsize='9', width='2', height='0.7')
    dot.node('quality', '🔍 Quality\n翻譯審核 · 品質評估\nSelf-Judge',
             shape='box', style='filled,rounded', fillcolor='#96CEB4',
             fontcolor='white', fontsize='9', width='2', height='0.7')

    # ── 連線：主流程（固定架構，不依賴數據）──
    # 1. MainLoop → Orchestrator
    dot.edge('main_loop', 'orchestrator',
             label='① Plan 分派 或 ReAct 自主決策',
             color='#E17055', fontsize='9', style='bold', penwidth='2')

    # 2. Orchestrator → 各執行 Agent
    dot.edge('orchestrator', 'knowledge',
             label='② 翻譯/搜尋/查詞',
             color='#4ECDC4', fontsize='8', penwidth='1.5')
    dot.edge('orchestrator', 'teaching',
             label='③ 教學/測驗/記錄',
             color='#45B7D1', fontsize='8', penwidth='1.5')
    dot.edge('orchestrator', 'quality',
             label='④ 審核/評估',
             color='#96CEB4', fontsize='8', penwidth='1.5')

    # 3. Quality → MainLoop（Self-Judge 回報）
    dot.edge('quality', 'main_loop',
             label='⑤ Self-Judge\n目標達成評估',
             color='#96CEB4', fontsize='8', style='dashed',
             constraint='false')

    # 4. Knowledge → Teaching（語料支持）
    dot.edge('knowledge', 'teaching',
             label='⑥ 語料驅動推薦',
             color='#B2BEC3', fontsize='7', style='dotted',
             constraint='false')

    st.graphviz_chart(dot, use_container_width=True)


# ============================================
# 繪圖：Token 消耗圖
# ============================================

def draw_token_chart(checkpoints):
    """繪製 Token 消耗趨勢"""
    if not checkpoints:
        return
    
    loops = []
    tokens = []
    llm_calls = []
    
    for ckpt in checkpoints:
        state = ckpt.get("state", {})
        loop = state.get("loop_count", 0)
        token = state.get("tokens_used_total", 0)
        calls = state.get("llm_calls_total", 0)
        if loop > 0:
            loops.append(loop)
            tokens.append(token)
            llm_calls.append(calls)
    
    if not loops:
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Token 消耗趨勢")
        chart_data = {"Loop": loops, "Tokens": tokens}
        st.line_chart(chart_data, x="Loop", y="Tokens", height=250)
    
    with col2:
        st.subheader("📈 LLM 調用次數")
        chart_data = {"Loop": loops, "Calls": llm_calls}
        st.line_chart(chart_data, x="Loop", y="Calls", height=250)


# ============================================
# 主頁面
# ============================================

def main():
    # 標題
    st.title("🏔️ 語聲同行 2.0 — Multi-Agent Dashboard")
    st.caption("可遷移的瀕危語言 AI 保護框架 | 5 Agents · 3 長程任務模式")
    
    # 自動刷新
    refresh = st.sidebar.selectbox("自動刷新", ["關閉", "5秒", "10秒", "30秒"], index=0)
    if refresh != "關閉":
        interval = int(refresh.replace("秒", ""))
        st.sidebar.info(f"每 {interval} 秒自動刷新")
        time.sleep(0.1)  # 讓 UI 先渲染
    
    # 載入數據
    state = load_latest_state()
    checkpoints = load_checkpoints()
    
    if not state:
        st.warning("⚠️ 尚無運行數據。啟動方式：")
        st.code("python3 run_autonomous.py --goal '學3個排灣語詞彙'", language="bash")
        st.info("或在下方手動啟動")
        return
    
    # ========================================
    # 頂部狀態列
    # ========================================
    goal = state.get("goal", {})
    goal_name = goal.get("name", "N/A") if isinstance(goal, dict) else str(goal)
    
    health = state.get("health_status", "unknown")
    health_emoji = {"healthy": "🟢", "degraded": "🟡", "critical": "🔴"}.get(health, "⚪")
    
    stop_reason = state.get("stop_reason", "")
    is_running = not state.get("should_stop", False) and not stop_reason
    
    status_text = "🟢 運行中" if is_running else f"⏹ 已停止（{stop_reason or '手動停止'}）"
    signal_exists = SIGNAL_PATH.exists()
    
    # 狀態卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("狀態", status_text)
    with col2:
        st.metric("循環次數", state.get("loop_count", 0))
    with col3:
        st.metric("LLM 調用", state.get("llm_calls_total", 0))
    with col4:
        st.metric("Token 用量", f"{state.get('tokens_used_total', 0):,}")
    with col5:
        elapsed = time.time() - state.get("started_at", time.time())
        st.metric("運行時間", fmt_duration(elapsed))
    
    # ========================================
    # 目標 & 健康狀態
    # ========================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Plan 進度", "💬 Agent 通訊", "📊 統計圖表", "📝 歷史時間線"
    ])
    
    # --- Tab 1: Plan 進度 ---
    with tab1:
        st.subheader(f"🎯 目標: {goal_name}")
        
        # 成功標準
        if isinstance(goal, dict) and goal.get("success_criteria"):
            st.write("**成功標準：**")
            for k, v in goal["success_criteria"].items():
                if isinstance(v, dict):
                    current = v.get("current", 0)
                    target = v.get("min", v.get("target", "?"))
                    ratio = current / target if isinstance(target, (int, float)) and target > 0 else 0
                    st.progress(min(ratio, 1.0), text=f"{k}: {current}/{target}")
        
        # Plan 步驟
        plan_data = state.get("plan", {})
        if plan_data and isinstance(plan_data, dict) and plan_data.get("steps"):
            steps = plan_data["steps"]
            st.write(f"**Plan 進度：{sum(1 for s in steps if s.get('status') == 'done')}/{len(steps)} 完成**")
            
            for s in steps:
                if not isinstance(s, dict):
                    continue
                status = s.get("status", "pending")
                emoji = {"done": "✅", "running": "🔄", "failed": "❌", "pending": "⏳"}.get(status, "⏳")
                agent = s.get("agent", "?")
                name = s.get("name", s.get("id", "?"))
                desc = s.get("description", "")[:100]
                
                with st.expander(f"{emoji} [{status}] {name} → {agent}", expanded=(status == "running")):
                    st.write(f"**描述：** {desc}")
                    if s.get("result"):
                        result = s["result"]
                        if isinstance(result, dict):
                            reply = result.get("data", {}).get("reply", "")
                            if reply:
                                st.markdown(reply[:500])
                    if s.get("error"):
                        st.error(s["error"])
        else:
            st.info("Plan 已完成或進入 ReAct 自主模式")
        
        # STOP.signal 狀態
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if signal_exists:
                st.error("🛑 STOP.signal 已觸發，系統將停止")
            else:
                st.success("✅ 無停止信號")
        with col_b:
            if st.button("🛑 發送 STOP 信號", disabled=signal_exists or not is_running):
                SIGNAL_PATH.touch()
                st.success("停止信號已發送！")
                st.rerun()
    
    # --- Tab 2: Agent 通訊 ---
    with tab2:
        results = parse_iteration_results(state.get("iteration_results", []))
        
        # 通訊圖
        st.subheader("🕸️ Agent 通訊網絡")
        draw_agent_graph(results, plan_data)
        
        # Agent 活動統計
        st.subheader("📊 Agent 活動統計")
        agent_stats = defaultdict(lambda: {"tasks": 0, "done": 0, "failed": 0})
        for r in results:
            info = get_step_info(r)
            agent = info.get("agent", "unknown")
            status = info.get("status", "unknown")
            agent_stats[agent]["tasks"] += 1
            if status == "done":
                agent_stats[agent]["done"] += 1
            elif status == "failed":
                agent_stats[agent]["failed"] += 1
        
        agent_emojis = {"knowledge": "🧠", "teaching": "📚", "quality": "🔍", "orchestrator": "🎯"}
        cols = st.columns(len(agent_stats) or 1)
        for i, (agent, stats) in enumerate(agent_stats.items()):
            with cols[i % len(cols)]:
                emoji = agent_emojis.get(agent, "🤖")
                name = agent if agent != "unknown" else "❓ 未知"
                st.metric(
                    f"{emoji} {name}",
                    f"{stats['done']}/{stats['tasks']} 完成",
                    delta=f"{stats['failed']} 失敗" if stats['failed'] > 0 else None
                )
        
        # 通訊記錄
        st.subheader("📨 通訊記錄")
        for r in reversed(results[-15:]):
            loop = r.get("loop", "?")
            info = get_step_info(r)
            step_name = info.get("name", "?")
            agent = info.get("agent", "?")
            status = info.get("status", "?")
            reply = info.get("reply", "")
            
            status_emoji = {"done": "✅", "failed": "❌"}.get(status, "🔄")
            agent_display = agent if agent != "?" else "❓"
            
            with st.expander(f"Loop {loop} | {status_emoji} {step_name} → {agent_display}"):
                if reply:
                    st.markdown(reply[:600])
                else:
                    st.text("（無回覆內容）")
    
    # --- Tab 3: 統計圖表 ---
    with tab3:
        draw_token_chart(checkpoints)
        
        # Checkpoint 歷史
        st.subheader("💾 Checkpoint 歷史")
        ckpt_data = []
        for ckpt in checkpoints:
            s = ckpt.get("state", {})
            ckpt_data.append({
                "ID": ckpt.get("id", "")[:20],
                "時間": fmt_time(ckpt.get("saved_at")),
                "Loop": s.get("loop_count", 0),
                "Tokens": s.get("tokens_used_total", 0),
                "LLM": s.get("llm_calls_total", 0),
                "健康": s.get("health_status", ""),
            })
        if ckpt_data:
            st.dataframe(ckpt_data, use_container_width=True, hide_index=True)
    
    # --- Tab 4: 歷史時間線 ---
    with tab4:
        st.subheader("🕐 執行時間線")
        results = parse_iteration_results(state.get("iteration_results", []))
        
        if not results:
            st.info("尚無執行記錄")
        else:
            start_ts = state.get("started_at", 0)
            for r in results:
                loop = r.get("loop", "?")
                ts = r.get("timestamp", 0)
                elapsed = ts - start_ts if ts and start_ts else 0
                
                info = get_step_info(r)
                step_name = info.get("name", "?")
                agent = info.get("agent", "?")
                status = info.get("status", "?")
                
                icon = "✅" if status == "done" else "❌" if status == "failed" else "🔄"
                agent_display = agent if agent != "?" else "❓"
                
                st.markdown(
                    f"**+{fmt_duration(elapsed)}** | {icon} Loop {loop} | "
                    f"`{step_name}` → `{agent_display}`"
                )
    
    # ========================================
    # 側邊欄
    # ========================================
    st.sidebar.divider()
    st.sidebar.subheader("🎯 當前任務")
    st.sidebar.write(f"**目標：** {goal_name}")
    if isinstance(goal, dict):
        st.sidebar.write(f"**最大時長：** {goal.get('max_duration_hours', '?')}h")
        st.sidebar.write(f"**Token 預算：** {goal.get('max_budget_tokens', '?'):,}")
    st.sidebar.write(f"**Session：** `{state.get('session_id', 'N/A')}`")
    
    st.sidebar.divider()
    st.sidebar.subheader("📖 快速指南")
    st.sidebar.markdown("""
    **啟動新任務：**
    ```bash
    python3 run_autonomous.py \\
      --goal "學5個排灣語詞彙" \\
      --max-hours 0.5
    ```
    
    **從 checkpoint 恢復：**
    ```bash
    python3 run_autonomous.py --resume
    ```
    
    **停止運行：**
    ```bash
    touch agent_framework/storage/STOP.signal
    ```
    """)
    
    st.sidebar.divider()
    st.sidebar.caption("Built with ❤️ by 地陪 | 語聲同行 2.0")
    
    # 自動刷新邏輯
    if refresh != "關閉":
        interval = int(refresh.replace("秒", ""))
        st_autorefresh(interval * 1000)


def st_autorefresh(interval_ms):
    """使用 HTML meta refresh 實現自動刷新"""
    st.markdown(
        f'<meta http-equiv="refresh" content="{interval_ms // 1000}">',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
