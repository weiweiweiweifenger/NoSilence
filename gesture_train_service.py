"""
Web 端自定义手势训练服务：采集 10 次样本 → CPU 训练 → 热加载模型
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from gesture_custom import find_label_by_zh, make_label_slug, save_custom_mapping
from gesture_dataset import append_rows, hands_to_row, remove_label_rows, resolve_data_csv
from gesture_engine import (
    HAND_CONNECTIONS,
    create_hand_landmarker,
    decode_frame,
    ensure_hand_model,
    next_monotonic_timestamp_ms,
    reset_hand_landmarker,
    _landmarks_to_json,
)
from train_gesture import train_gesture_model

SAMPLES_REQUIRED = 10
COUNTDOWN_SEC = 3


@dataclass
class TrainStatus:
    phase: str = "idle"  # idle | collecting | training | done | error
    label: str = ""
    zh_meaning: str = ""
    samples_collected: int = 0
    samples_required: int = SAMPLES_REQUIRED
    message: str = ""
    error: str = ""
    epoch: int = 0
    total_epochs: int = 0
    test_acc: float | None = None
    device: str = "cpu"
    last_num_hands: int = 0
    photo_uploaded: bool = False


class GestureTrainService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = TrainStatus()
        self._sample_rows: list[list] = []
        self._landmarker = None
        self._stream_start = 0.0
        self._last_timestamp_ms = 0
        self._train_thread: threading.Thread | None = None
        self._overwrite: bool = False

    def get_status(self) -> dict:
        with self._lock:
            s = self._status
            return {
                "phase": s.phase,
                "label": s.label,
                "zh_meaning": s.zh_meaning,
                "samples_collected": s.samples_collected,
                "samples_required": s.samples_required,
                "message": s.message,
                "error": s.error,
                "epoch": s.epoch,
                "total_epochs": s.total_epochs,
                "test_acc": s.test_acc,
                "device": s.device,
                "countdown_sec": COUNTDOWN_SEC,
                "last_num_hands": s.last_num_hands,
                "photo_uploaded": s.photo_uploaded,
                "can_capture": s.phase == "collecting" and s.samples_collected < s.samples_required,
                "can_train": (
                    s.phase == "collecting"
                    and s.samples_collected >= s.samples_required
                    and s.photo_uploaded
                ),
                "needs_photo": (
                    s.phase == "collecting"
                    and s.samples_collected >= s.samples_required
                    and not s.photo_uploaded
                ),
            }

    def begin(
        self,
        zh_meaning: str,
        label_hint: str | None = None,
        *,
        overwrite: bool = False,
        replace_label: str | None = None,
    ) -> dict:
        zh_meaning = (zh_meaning or "").strip()
        if not zh_meaning:
            return {"ok": False, "error": "请输入手势的中文含义"}

        if self._train_thread and self._train_thread.is_alive():
            return {"ok": False, "error": "正在训练中，请稍候"}

        existing_label = find_label_by_zh(zh_meaning)
        if existing_label and not overwrite:
            return {
                "ok": False,
                "error": f"手势「{zh_meaning}」已存在，请确认是否覆盖",
                "duplicate": True,
                "existing_label": existing_label,
            }

        if overwrite and replace_label:
            label = replace_label.strip().lower()
        elif existing_label and overwrite:
            label = existing_label
        else:
            label = make_label_slug(zh_meaning, label_hint)

        if overwrite:
            message = (
                f"将覆盖「{zh_meaning}」并重新采集 {SAMPLES_REQUIRED} 次"
                f"（每次倒计时 {COUNTDOWN_SEC} 秒，支持单手或双手）"
            )
        else:
            message = (
                f"即将自动采集「{zh_meaning}」共 {SAMPLES_REQUIRED} 次"
                f"（每次倒计时 {COUNTDOWN_SEC} 秒，支持单手或双手）"
            )

        with self._lock:
            self._status = TrainStatus(
                phase="collecting",
                label=label,
                zh_meaning=zh_meaning,
                samples_collected=0,
                message=message,
            )
            self._sample_rows = []
            self._overwrite = bool(overwrite)
            self._stream_start = time.time()

        self._landmarker = reset_hand_landmarker(self._landmarker)
        return {"ok": True, **self.get_status()}

    def _ensure_landmarker(self) -> None:
        if self._landmarker is None:
            self._landmarker = create_hand_landmarker(ensure_hand_model())

    def _detect_hands(self, image_b64: str) -> tuple[list, list[dict], int]:
        """检测手部关键点，返回 (原始 landmarks, JSON, 手数)。"""
        frame = decode_frame(image_b64)
        if frame is None:
            raise ValueError("无法解码摄像头画面")

        self._ensure_landmarker()
        import cv2
        import mediapipe as mp

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._last_timestamp_ms = next_monotonic_timestamp_ms(self._last_timestamp_ms)
        result = self._landmarker.detect_for_video(mp_image, self._last_timestamp_ms)

        hands = result.hand_landmarks or []
        return hands, _landmarks_to_json(hands), len(hands)

    def preview_frame(self, image_b64: str) -> dict:
        """采集模式下实时预览手部 21 关键点与骨架（不保存样本）。"""
        with self._lock:
            if self._status.phase != "collecting":
                return {"ok": False, "error": "未在采集模式", "hands": [], "num_hands": 0}

        try:
            _, hands_json, num_hands = self._detect_hands(image_b64)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "hands": [], "num_hands": 0}

        return {
            "ok": True,
            "hands": hands_json,
            "connections": HAND_CONNECTIONS,
            "num_hands": num_hands,
        }

    def capture_sample(self, image_b64: str) -> dict:
        with self._lock:
            if self._status.phase != "collecting":
                return {"ok": False, "error": "请先开始新的手势采集"}
            if self._status.samples_collected >= SAMPLES_REQUIRED:
                return {"ok": False, "error": f"已采集 {SAMPLES_REQUIRED} 次，可以开始训练"}

        try:
            hands, _, num_hands = self._detect_hands(image_b64)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "num_hands": 0}

        if num_hands < 1:
            return {"ok": False, "error": "未检测到手，请将单手或双手放入画面中央", "num_hands": 0}

        with self._lock:
            label = self._status.label
        row = hands_to_row(label, hands)
        if row is None:
            return {"ok": False, "error": "采样失败，请重试", "num_hands": num_hands}

        hand_desc = "双手" if num_hands >= 2 else "单手"
        with self._lock:
            self._sample_rows.append(row)
            self._status.samples_collected = len(self._sample_rows)
            self._status.last_num_hands = num_hands
            n = self._status.samples_collected
            zh = self._status.zh_meaning
            if n >= SAMPLES_REQUIRED:
                self._status.message = (
                    f"「{zh}」已采集 {n}/{SAMPLES_REQUIRED} 次（末次 {hand_desc}），"
                    f"请上传手势示意图后再开始训练"
                )
            else:
                self._status.message = (
                    f"第 {n}/{SAMPLES_REQUIRED} 次完成（{hand_desc}），"
                    f"请保持相同方式继续…"
                )

        result_status = self.get_status()
        result_status["num_hands"] = num_hands
        return {"ok": True, **result_status}

    def upload_photo(self, image_b64: str | None = None, raw_bytes: bytes | None = None) -> dict:
        with self._lock:
            if self._status.phase != "collecting":
                return {"ok": False, "error": "请先完成手势采集"}
            if self._status.samples_collected < SAMPLES_REQUIRED:
                return {"ok": False, "error": f"请先完成 {SAMPLES_REQUIRED} 次手势采集"}
            zh = self._status.zh_meaning

        try:
            from gesture_photo_store import save_gesture_photo, save_gesture_photo_bytes

            if raw_bytes:
                path = save_gesture_photo_bytes(zh, raw_bytes)
            elif image_b64:
                path = save_gesture_photo(zh, image_b64)
            else:
                return {"ok": False, "error": "缺少图片数据"}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        with self._lock:
            self._status.photo_uploaded = True
            self._status.message = f"示意图已保存，可以开始训练「{zh}」"

        result = self.get_status()
        result["photo_path"] = str(path.name)
        return {"ok": True, **result}

    def start_training(self) -> dict:
        with self._lock:
            if self._status.phase != "collecting":
                return {"ok": False, "error": "请先完成手势采集"}
            if self._status.samples_collected < SAMPLES_REQUIRED:
                return {"ok": False, "error": f"需要采集 {SAMPLES_REQUIRED} 次，当前 {self._status.samples_collected} 次"}
            if not self._status.photo_uploaded:
                return {"ok": False, "error": "请先上传手势示意图"}
            if self._train_thread and self._train_thread.is_alive():
                return {"ok": False, "error": "训练正在进行中"}

            rows = list(self._sample_rows)
            label = self._status.label
            zh = self._status.zh_meaning
            overwrite = self._overwrite
            self._status.phase = "training"
            self._status.message = "正在写入样本并启动 CPU 训练…"
            self._status.error = ""
            self._status.device = "cpu"

        self._train_thread = threading.Thread(
            target=self._run_training,
            args=(rows, label, zh, overwrite),
            name="gesture-train",
            daemon=True,
        )
        self._train_thread.start()
        return {"ok": True, **self.get_status()}

    def cancel(self) -> dict:
        with self._lock:
            if self._status.phase == "training" and self._train_thread and self._train_thread.is_alive():
                return {"ok": False, "error": "训练进行中，无法取消"}
            self._status = TrainStatus()
            self._sample_rows = []
            self._overwrite = False
        return {"ok": True, **self.get_status()}

    def _run_training(self, rows: list[list], label: str, zh: str, overwrite: bool) -> None:
        try:
            csv_path = resolve_data_csv()
            if overwrite:
                removed = remove_label_rows(label, csv_path)
                with self._lock:
                    if removed:
                        self._status.message = (
                            f"已移除「{zh}」旧样本 {removed} 条，正在写入新样本…"
                        )
            append_rows(rows, csv_path)
            save_custom_mapping(label, zh)

            def on_progress(info: dict) -> None:
                with self._lock:
                    self._status.epoch = info["epoch"]
                    self._status.total_epochs = info["total_epochs"]
                    self._status.message = (
                        f"CPU 训练中 Epoch {info['epoch']}/{info['total_epochs']} "
                        f"· 准确率 {info['test_acc']:.1f}%"
                    )

            result = train_gesture_model(
                csv_path,
                device="cpu",
                progress_callback=on_progress,
            )

            from gesture_speech import add_tts_for_word, reload_gesture_map
            from gesture_engine import get_engine

            reload_gesture_map()
            add_tts_for_word(zh)

            engine = get_engine()
            if engine.active:
                engine.stop_session()
            reloaded = engine.reload_model()

            with self._lock:
                self._status.phase = "done"
                self._status.test_acc = result["test_acc"]
                self._status.message = (
                    f"训练完成！「{zh}」已加入识别（测试准确率 {result['test_acc']:.1f}%）"
                )
                if not reloaded:
                    self._status.error = engine.init_error or "模型热加载失败，请重启服务"
        except Exception as exc:
            with self._lock:
                self._status.phase = "error"
                self._status.error = str(exc)
                self._status.message = "训练失败"


_service: GestureTrainService | None = None


def get_train_service() -> GestureTrainService:
    global _service
    if _service is None:
        _service = GestureTrainService()
    return _service
