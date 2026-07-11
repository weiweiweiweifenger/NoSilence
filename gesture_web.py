"""
无声之声 · Voice of Silence
iOS 18 风格 Web 仪表板 — 连接 OpenCV 识别进程实时数据

运行: streamlit run gesture_web.py
静态设计稿: design/voice_of_silence.html
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path

import streamlit as st

from gesture_recognize import GESTURE_MODEL_PATH, HAND_MODEL_PATH
from gesture_speech import to_chinese
from gesture_state import read_runtime_state, request_clear_history

REFRESH_SEC = 0.3

# ---------------------------------------------------------------------------
# iOS 18 Design System CSS
# ---------------------------------------------------------------------------

def _css(theme: str) -> str:
    dark = theme == "dark"
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --blue: {"#0A84FF" if dark else "#007AFF"};
  --blue-bg: {"rgba(10,132,255,0.12)" if dark else "rgba(0,122,255,0.08)"};
  --bg: {"#000000" if dark else "#F2F2F7"};
  --card: {"rgba(44,44,46,0.82)" if dark else "rgba(255,255,255,0.82)"};
  --card-border: {"rgba(255,255,255,0.08)" if dark else "rgba(255,255,255,0.95)"};
  --nav-bg: {"rgba(28,28,30,0.80)" if dark else "rgba(255,255,255,0.80)"};
  --text: {"#F5F5F7" if dark else "#1C1C1E"};
  --text-2: {"#AEAEB2" if dark else "#636366"};
  --text-3: {"#636366" if dark else "#AEAEB2"};
  --sep: {"rgba(84,84,88,0.36)" if dark else "rgba(60,60,67,0.12)"};
  --shadow: {"0 2px 20px rgba(0,0,0,0.35)" if dark else "0 2px 16px rgba(0,0,0,0.06)"};
  --success: #34C759;
  --warning: #FF9500;
  --danger: #FF3B30;
  --radius: 24px;
  --font: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter",
          "PingFang SC", "Microsoft YaHei", sans-serif;
}}

html, body, [class*="css"] {{ font-family: var(--font); }}
.stApp {{ background: var(--bg); color: var(--text); }}
.main .block-container {{
  padding-top: 0.5rem; padding-bottom: 3rem;
  max-width: 1180px;
}}

#MainMenu, footer, header {{ visibility: hidden; height: 0; }}

/* 导航栏 */
.vos-nav {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 4px 20px;
  border-bottom: 1px solid var(--sep);
  margin-bottom: 24px;
}}
.vos-logo {{ display: flex; align-items: center; gap: 10px; }}
.vos-logo-icon {{
  width: 36px; height: 36px; border-radius: 10px;
  background: var(--blue); color: #fff;
  display: flex; align-items: center; justify-content: center; font-size: 18px;
}}
.vos-logo-title {{ font-size: 17px; font-weight: 600; letter-spacing: -0.02em; color: var(--text); }}
.vos-logo-sub {{ font-size: 11px; color: var(--text-3); letter-spacing: 0.04em; }}
.vos-status {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-2); }}
.vos-dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--success); }}
.vos-dot.off {{ background: var(--text-3); }}

/* 卡片 */
.vos-section {{ font-size: 13px; font-weight: 600; color: var(--text-2);
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; }}
.vos-card {{
  background: var(--card);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border: 1px solid var(--card-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 22px;
  margin-bottom: 14px;
}}
.vos-card-title {{ font-size: 15px; font-weight: 600; color: var(--text-2); margin-bottom: 14px; }}

/* 摄像头占位 */
.vos-camera {{
  aspect-ratio: 4/3; border-radius: 20px;
  background: {"#1C1C1E" if dark else "#2C2C2E"};
  position: relative; overflow: hidden; margin-bottom: 14px;
}}
.vos-camera-inner {{
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  color: rgba(255,255,255,0.55); font-size: 13px; gap: 6px;
}}
.vos-badge {{
  position: absolute; top: 10px; left: 10px;
  background: rgba(0,0,0,0.45); color: #fff;
  font-size: 11px; padding: 4px 10px; border-radius: 20px;
}}
.vos-fps {{
  position: absolute; top: 10px; right: 10px;
  background: rgba(0,0,0,0.45); color: #fff;
  font-size: 11px; padding: 4px 10px; border-radius: 20px;
  font-variant-numeric: tabular-nums;
}}

.vos-pill {{
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 14px; border-radius: 20px;
  font-size: 13px; font-weight: 500;
  background: var(--blue-bg); color: var(--blue);
  margin: 0 6px 6px 0;
}}
.vos-pill.ok {{ background: rgba(52,199,89,0.12); color: var(--success); }}
.vos-pill.wait {{ background: rgba(255,149,0,0.12); color: var(--warning); }}

/* 识别结果 */
.vos-result-word {{
  font-size: 52px; font-weight: 700; letter-spacing: -0.03em;
  color: var(--text); text-align: center; line-height: 1.1;
}}
.vos-result-sub {{ font-size: 15px; color: var(--text-2); text-align: center; margin-top: 8px; }}

.vos-mini {{
  background: var(--blue-bg); border-radius: 20px;
  padding: 16px 18px; border: 1px solid var(--sep); margin-bottom: 12px;
}}
.vos-mini-label {{
  font-size: 11px; font-weight: 600; color: var(--text-3);
  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;
}}
.vos-mini-value {{ font-size: 16px; font-weight: 500; color: var(--text); line-height: 1.45; }}
.vos-mini.danger {{ background: rgba(255,59,48,0.08); border-color: rgba(255,59,48,0.2); }}
.vos-mini.danger .vos-mini-label {{ color: var(--danger); }}
.vos-mini.danger .vos-mini-value {{ color: var(--danger); }}

/* 历史 */
.vos-history {{ max-height: 260px; overflow-y: auto; }}
.vos-hist-row {{
  display: flex; gap: 16px; padding: 11px 0;
  border-bottom: 1px solid var(--sep); font-size: 15px;
}}
.vos-hist-row:last-child {{ border-bottom: none; }}
.vos-hist-time {{ color: var(--text-3); min-width: 44px; font-size: 14px;
  font-variant-numeric: tabular-nums; }}
.vos-hist-text {{ color: var(--text); flex: 1; }}
.vos-empty {{ color: var(--text-3); text-align: center; padding: 24px 0; font-size: 14px; }}

/* 统计 */
.vos-stats-row {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-bottom: 18px; }}
.vos-stat-box {{ background: var(--blue-bg); border-radius: 20px; padding: 14px; text-align: center; }}
.vos-stat-num {{ font-size: 28px; font-weight: 700; color: var(--blue);
  font-variant-numeric: tabular-nums; }}
.vos-stat-lbl {{ font-size: 11px; color: var(--text-2); margin-top: 4px; }}

.vos-freq-row {{ display: flex; align-items: center; gap: 10px; padding: 7px 0; font-size: 14px; }}
.vos-freq-name {{ min-width: 56px; font-weight: 500; color: var(--text); }}
.vos-freq-bar {{ flex: 1; height: 6px; background: var(--sep); border-radius: 3px; overflow: hidden; }}
.vos-freq-fill {{ height: 100%; background: var(--blue); border-radius: 3px; }}
.vos-freq-n {{ min-width: 24px; text-align: right; color: var(--text-2); font-size: 13px; }}

.stButton button {{
  border-radius: 20px !important; font-weight: 500 !important;
  border: 1px solid var(--sep) !important;
  background: var(--card) !important; color: var(--text) !important;
}}
[data-testid="stSidebar"] {{
  background: var(--nav-bg) !important;
  border-right: 1px solid var(--sep) !important;
}}

@media (max-width: 768px) {{
  .vos-result-word {{ font-size: 36px; }}
  .vos-stats-row {{ grid-template-columns: 1fr; }}
}}
</style>
"""


def _esc(text: str) -> str:
    return html_lib.escape(str(text))


def _format_duration(seconds: int) -> str:
    m, s = divmod(max(0, seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m:02d}m {s:02d}s"


def _last_recognition_time(state) -> str:
    for item in state.chat_log or []:
        if item.get("kind") != "raw" and item.get("text"):
            return str(item.get("time", "—"))
    if state.history:
        t = str(state.history[0].get("time", ""))
        return t[:5] if t else "—"
    return "—"


def _ai_sentence(state) -> str:
    pending = state.pending_words or []
    if pending:
        return f"组合中：{' · '.join(pending)}（停手 2 秒后生成）"
    display = str(state.display_zh or "")
    if len(display) > 3 and not display.startswith("…") and display not in (
        "未检测到手", "—", "📡 离线", "⚠️ 模型缺失", "…",
    ):
        if "。" in display or "，" in display or len(display) > 4:
            return display
    for item in reversed(state.chat_log or []):
        if item.get("kind") != "raw" and item.get("text"):
            return str(item["text"])
    return "等待手势输入…"


def _is_emergency(state) -> bool:
    if state.raw_label == "danger" and state.is_stable:
        return True
    pending = state.pending_words or []
    return "危险" in pending or state.display_zh == "危险"


def _render_nav(state, theme: str) -> None:
    alive = state.is_alive
    status_txt = "系统运行中 · OpenCV 已连接" if alive else "等待 OpenCV 进程"
    dot_cls = "" if alive else " off"

    col_logo, col_status, col_actions = st.columns([2.2, 2, 1.2])
    with col_logo:
        st.markdown(
            f'<div class="vos-nav" style="border:none;padding:0;margin:0">'
            f'<div class="vos-logo">'
            f'<div class="vos-logo-icon">🤟</div>'
            f'<div><div class="vos-logo-title">无声之声</div>'
            f'<div class="vos-logo-sub">Voice of Silence</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with col_status:
        st.markdown(
            f'<div style="padding-top:10px">'
            f'<div class="vos-status"><span class="vos-dot{dot_cls}"></span>'
            f'{_esc(status_txt)}</div></div>',
            unsafe_allow_html=True,
        )
    with col_actions:
        c1, c2 = st.columns(2)
        with c1:
            icon = "☀️" if theme == "dark" else "🌙"
            if st.button(icon, key="theme_btn", help="切换深色模式"):
                st.session_state.vos_theme = "dark" if theme == "light" else "light"
                st.rerun()
        with c2:
            if st.button("🗑️", help="清空交流记录"):
                request_clear_history()
                st.toast("已清空", icon="✓")


def _render_left_column(state) -> None:
    fps = int(state.fps) if state.is_alive else 0
    fps_html = f'<span class="vos-fps">{fps} FPS</span>' if state.is_alive else ""

    if state.is_stable:
        pill = '<span class="vos-pill ok">● 已稳定识别</span>'
    elif state.is_alive:
        pill = '<span class="vos-pill wait">识别中</span>'
    else:
        pill = '<span class="vos-pill">未连接</span>'

    pending = state.pending_words or []
    pending_pill = ""
    if pending:
        joined = _esc(" · ".join(pending))
        pending_pill = f'<span class="vos-pill">组合中：{joined}</span>'

    st.markdown(f'<p class="vos-section">实时识别</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="vos-card">'
        f'<div class="vos-camera" role="img" aria-label="OpenCV 实时摄像头与骨架">'
        f'<span class="vos-badge">MediaPipe 骨架</span>{fps_html}'
        f'<div class="vos-camera-inner">'
        f'<span>OpenCV 实时窗口</span>'
        f'<span style="font-size:11px;opacity:0.55">python gesture_recognize.py</span>'
        f'</div></div>'
        f'<div>{pill}{pending_pill}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_right_column(state) -> None:
    if not GESTURE_MODEL_PATH.exists():
        display, sub = "模型缺失", "请运行 train_gesture.py"
    elif not state.is_alive:
        display, sub = "离线", "等待 OpenCV 进程"
    else:
        display, sub = str(state.display_zh), str(state.sub_text)

    ai_text = _ai_sentence(state)
    emergency = _is_emergency(state)

    if emergency:
        em_label, em_value, em_cls = "紧急求助", "⚠️ 检测到「危险」手势", " danger"
    else:
        em_label, em_value, em_cls = "紧急求助", "正常状态", ""

    speech = "缓存就绪 · 稳定后自动播报" if state.is_alive else "等待连接"
    if state.is_stable and state.confidence >= 0.45:
        speech = f"已播报 · {_esc(to_chinese(state.raw_label) or display)}"

    st.markdown('<p class="vos-section">识别结果</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="vos-card">'
        f'<p class="vos-card-title">当前识别</p>'
        f'<div class="vos-result-word">{_esc(display)}</div>'
        f'<div class="vos-result-sub">{_esc(sub)}</div>'
        f'</div>'
        f'<div class="vos-mini">'
        f'<p class="vos-mini-label">AI 润色</p>'
        f'<p class="vos-mini-value">{_esc(ai_text)}</p></div>'
        f'<div class="vos-mini">'
        f'<p class="vos-mini-label">语音播报</p>'
        f'<p class="vos-mini-value">🔊 {_esc(speech)}</p></div>'
        f'<div class="vos-mini{em_cls}">'
        f'<p class="vos-mini-label">{em_label}</p>'
        f'<p class="vos-mini-value">{em_value}</p></div>',
        unsafe_allow_html=True,
    )


def _render_history(state) -> None:
    entries = state.chat_log or []
    parts = []
    for item in entries:
        if item.get("kind") == "raw":
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        parts.append(
            f'<div class="vos-hist-row">'
            f'<span class="vos-hist-time">{_esc(item.get("time", "--:--"))}</span>'
            f'<span class="vos-hist-text">{_esc(text)}</span></div>'
        )
    body = (
        "".join(parts)
        if parts
        else '<div class="vos-empty">连续打手势后停 2 秒，自动生成完整语句</div>'
    )
    st.markdown(
        f'<p class="vos-section">交流历史</p>'
        f'<div class="vos-card"><div class="vos-history">{body}</div></div>',
        unsafe_allow_html=True,
    )


def _render_stats(state) -> None:
    stats = state.stats or {}
    counts: dict = stats.get("gesture_counts", {})
    total = stats.get("total_stable", 0)
    last_time = _last_recognition_time(state)
    kinds = len(counts)

    freq_html = ""
    if counts:
        max_n = max(counts.values())
        for en, n in sorted(counts.items(), key=lambda x: -x[1]):
            pct = (n / max_n) * 100 if max_n else 0
            freq_html += (
                f'<div class="vos-freq-row">'
                f'<span class="vos-freq-name">{_esc(to_chinese(en))}</span>'
                f'<div class="vos-freq-bar"><div class="vos-freq-fill" style="width:{pct:.0f}%"></div></div>'
                f'<span class="vos-freq-n">{n}</span></div>'
            )
    else:
        freq_html = '<div class="vos-empty" style="padding:12px">暂无数据</div>'

    st.markdown(
        f'<p class="vos-section">数据统计</p>'
        f'<div class="vos-card">'
        f'<div class="vos-stats-row">'
        f'<div class="vos-stat-box"><div class="vos-stat-num">{total}</div>'
        f'<div class="vos-stat-lbl">识别次数</div></div>'
        f'<div class="vos-stat-box"><div class="vos-stat-num">{kinds}</div>'
        f'<div class="vos-stat-lbl">手势种类</div></div>'
        f'<div class="vos-stat-box"><div class="vos-stat-num" style="font-size:20px;padding-top:6px">'
        f'{_esc(last_time)}</div><div class="vos-stat-lbl">最近识别</div></div>'
        f'</div>'
        f'<p class="vos-card-title">手势频率</p>{freq_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every=REFRESH_SEC)
def _live_dashboard(theme: str) -> None:
    state = read_runtime_state()
    _render_nav(state, theme)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        _render_left_column(state)
    with col_r:
        _render_right_column(state)

    col_h, col_s = st.columns([1.2, 1], gap="large")
    with col_h:
        _render_history(state)
    with col_s:
        _render_stats(state)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.caption(f"手势模型 {'✓' if GESTURE_MODEL_PATH.exists() else '✗'}")
    with m2:
        st.caption(f"手部模型 {'✓' if HAND_MODEL_PATH.exists() else '…'}")
    with m3:
        st.caption(f"刷新 {REFRESH_SEC}s")


def main() -> None:
    st.set_page_config(
        page_title="无声之声 · Voice of Silence",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "vos_theme" not in st.session_state:
        st.session_state.vos_theme = "light"

    theme = st.session_state.vos_theme
    st.markdown(_css(theme), unsafe_allow_html=True)

    if not GESTURE_MODEL_PATH.exists():
        st.error("未找到 gesture_model.pth，请先运行 python train_gesture.py")
        st.stop()

    _live_dashboard(theme)


if __name__ == "__main__":
    main()
