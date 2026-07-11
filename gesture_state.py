"""
OpenCV 识别进程 ↔ Streamlit UI 的共享状态（JSON 文件 IPC）。

OpenCV 进程：GestureStateWriter 写入当前结果、历史、统计
Streamlit 进程：read_runtime_state() 只读展示

运行方式（两个终端）:
  python gesture_recognize.py          # OpenCV 窗口 + 推理
  streamlit run gesture_web.py         # 仅 UI
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "runtime_state.json"
COMMAND_FILE = DATA_DIR / "runtime_command.json"

MAX_HISTORY = 50
MAX_CHAT_LOG = 100
STALE_SEC = 3.0
PAUSE_SEC = 0.8  # 无手超过此秒数后，将累积词润色成句并写入交流记录


@dataclass
class ChatEntry:
    time: str
    text: str


@dataclass
class HistoryEntry:
    time: str
    en: str
    zh: str
    confidence: float


@dataclass
class RuntimeState:
    running: bool = False
    updated_at: float = 0.0
    display_zh: str = "—"
    sub_text: str = "等待 OpenCV 进程启动"
    raw_label: str = ""
    is_stable: bool = False
    confidence: float = 0.0
    fps: float = 0.0
    pending_words: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    chat_log: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    polishing: bool = False

    @property
    def is_alive(self) -> bool:
        return self.running and (time.time() - self.updated_at) < STALE_SEC


def _atomic_write(path: Path, payload: dict) -> None:
    """写入 JSON 状态文件。Windows 下 Streamlit 读取时 replace 会失败，故优先原地覆写。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(".tmp")

    for attempt in range(5):
        try:
            path.write_text(text, encoding="utf-8")
            tmp.unlink(missing_ok=True)
            return
        except (PermissionError, OSError):
            if attempt < 4:
                time.sleep(0.03 * (attempt + 1))

    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def read_runtime_state() -> RuntimeState:
    if not STATE_FILE.exists():
        return RuntimeState()
    for attempt in range(3):
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return RuntimeState(
                running=data.get("running", False),
                updated_at=data.get("updated_at", 0.0),
                display_zh=data.get("display_zh", "—"),
                sub_text=data.get("sub_text", ""),
                raw_label=data.get("raw_label", ""),
                is_stable=data.get("is_stable", False),
                confidence=data.get("confidence", 0.0),
                fps=data.get("fps", 0.0),
                pending_words=data.get("pending_words", []),
                history=data.get("history", []),
                chat_log=data.get("chat_log", []),
                stats=data.get("stats", {}),
                polishing=data.get("polishing", False),
            )
        except (json.JSONDecodeError, OSError):
            if attempt < 2:
                time.sleep(0.02)
    return RuntimeState(sub_text="状态文件读取失败")


def request_clear_history() -> None:
    _atomic_write(COMMAND_FILE, {"action": "clear_history", "ts": time.time()})


def _consume_command() -> str | None:
    if not COMMAND_FILE.exists():
        return None
    try:
        data = json.loads(COMMAND_FILE.read_text(encoding="utf-8"))
        COMMAND_FILE.unlink(missing_ok=True)
        return data.get("action")
    except (json.JSONDecodeError, OSError):
        COMMAND_FILE.unlink(missing_ok=True)
        return None


class GestureStateWriter:
    """OpenCV 主循环内使用，周期性 flush 到 JSON。"""

    def __init__(self, max_history: int = MAX_HISTORY) -> None:
        self._history: deque[HistoryEntry] = deque(maxlen=max_history)
        self._chat_log: deque[ChatEntry] = deque(maxlen=MAX_CHAT_LOG)
        self._word_buffer: list[str] = []
        self._last_history_key: tuple[str, float] | None = None
        self._gesture_counts: dict[str, int] = {}
        self._total_stable = 0
        self._session_start = time.time()
        self._last_flush = 0.0
        self._flush_interval = 0.05
        self._lock = threading.Lock()
        self._running = True
        self._no_hand_since: float | None = None
        self._showing_sentence = False
        self._flushing = False
        self._polishing = False

        self.display_zh = "—"
        self.sub_text = "正在启动…"
        self.raw_label = ""
        self.is_stable = False
        self.confidence = 0.0
        self.fps = 0.0

    def handle_commands(self) -> None:
        if _consume_command() == "clear_history":
            with self._lock:
                self._history.clear()
                self._chat_log.clear()
                self._word_buffer.clear()
                self._last_history_key = None
                self._gesture_counts.clear()
                self._total_stable = 0
                self._session_start = time.time()
                self._no_hand_since = None
                self._showing_sentence = False
                self._flushing = False
                self._polishing = False
            self.flush(running=True)

    def _append_chat(self, text: str) -> None:
        with self._lock:
            self._chat_log.appendleft(
                ChatEntry(time=datetime.now().strftime("%H:%M"), text=text)
            )

    def _on_stable_gesture(self, raw_label: str, zh: str, confidence: float) -> None:
        key = (raw_label, round(confidence, 3))
        if key == self._last_history_key:
            return
        self._last_history_key = key
        self._showing_sentence = False

        if not self._word_buffer or self._word_buffer[-1] != zh:
            self._word_buffer.append(zh)

        self._history.appendleft(
            HistoryEntry(
                time=datetime.now().strftime("%H:%M:%S"),
                en=raw_label,
                zh=zh,
                confidence=confidence,
            )
        )
        self._gesture_counts[raw_label] = self._gesture_counts.get(raw_label, 0) + 1
        self._total_stable += 1
        self.flush(running=self._running)

    def _flush_sentence_async(self, words: list[str]) -> None:
        from gesture_polish import polish_words

        try:
            sentence = polish_words(words)
            if sentence:
                self._append_chat(sentence)
                self.display_zh = sentence
                self.sub_text = " · ".join(words)
                self._showing_sentence = True
        finally:
            self._flushing = False
            self._polishing = False
            self.flush(running=self._running)

    def _start_polish_words(self, words: list[str], *, running: bool | None = None) -> bool:
        """将缓冲词异步润色成句，并立即写入交流历史。"""
        if not words or self._flushing:
            return False

        if running is None:
            running = self._running

        self._word_buffer.clear()
        self._no_hand_since = None
        self._last_history_key = None
        self._flushing = True
        self._polishing = True
        self.display_zh = "…"
        self.sub_text = "正在生成语句"
        self.flush(running=running)

        threading.Thread(
            target=self._flush_sentence_async,
            args=(list(words),),
            name="polish-sentence",
            daemon=True,
        ).start()
        return True

    def _try_flush_on_pause(self, now: float) -> None:
        if not self._word_buffer or self._flushing:
            return
        if self._no_hand_since is None:
            return
        if now - self._no_hand_since < PAUSE_SEC:
            return
        self._start_polish_words(list(self._word_buffer))

    def flush_pending_words(self) -> bool:
        """停止识别时，将剩余词语立即润色成句。"""
        if not self._word_buffer:
            return False
        return self._start_polish_words(list(self._word_buffer), running=False)

    def update_frame(
        self,
        *,
        display_zh: str,
        sub_text: str,
        raw_label: str,
        is_stable: bool,
        confidence: float,
        fps: float,
        hands_detected: bool,
    ) -> None:
        now = time.time()

        if hands_detected:
            self._no_hand_since = None
            if not self._polishing:
                self._showing_sentence = False
                self.display_zh = display_zh
                self.sub_text = sub_text
            if is_stable and raw_label and confidence >= 0.45:
                from gesture_speech import to_chinese

                self._on_stable_gesture(raw_label, to_chinese(raw_label), confidence)
        else:
            if self._word_buffer and not self._flushing:
                if self._no_hand_since is None:
                    self._no_hand_since = now
                self._try_flush_on_pause(now)

            if not self._showing_sentence and not self._polishing:
                self.display_zh = display_zh
                self.sub_text = sub_text

        self.raw_label = raw_label
        self.is_stable = is_stable
        self.confidence = confidence
        self.fps = fps

        if now - self._last_flush >= self._flush_interval:
            self.flush(running=True)
            self._last_flush = now

    def flush(self, *, running: bool) -> None:
        self._running = running
        session_sec = max(0, int(time.time() - self._session_start))
        with self._lock:
            history = [
                {"time": h.time, "en": h.en, "zh": h.zh, "confidence": h.confidence}
                for h in self._history
            ]
            chat_log = [{"time": c.time, "text": c.text} for c in self._chat_log]
            pending = list(self._word_buffer)
        payload = {
            "running": running,
            "updated_at": time.time(),
            "display_zh": self.display_zh,
            "sub_text": self.sub_text,
            "raw_label": self.raw_label,
            "is_stable": self.is_stable,
            "confidence": self.confidence,
            "fps": self.fps,
            "pending_words": pending,
            "polishing": self._polishing,
            "history": history,
            "chat_log": chat_log,
            "stats": {
                "total_stable": self._total_stable,
                "gesture_counts": dict(self._gesture_counts),
                "session_seconds": session_sec,
            },
        }
        _atomic_write(STATE_FILE, payload)

    def shutdown(self) -> None:
        words = list(self._word_buffer) if self._word_buffer and not self._flushing else []
        if words:
            self._start_polish_words(words, running=False)
        else:
            try:
                self.flush(running=False)
            except OSError:
                pass
        self._running = False
