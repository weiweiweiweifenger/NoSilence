"""
手势识别引擎 — 供 Flask Web 与 OpenCV CLI 共用。

浏览器摄像头帧 → MediaPipe → MLP → 稳定预测 → 状态写入 / TTS
"""

from __future__ import annotations

import base64
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_features import hands_to_feature_vector, sort_hands_left_to_right
from gesture_speech import init_speech, request_speak, reset_speak_cache, to_chinese
from gesture_state import GestureStateWriter
from train_gesture import GestureMLP

HAND_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
GESTURE_MODEL_PATH = Path(__file__).parent / "gesture_model.pth"

MAX_NUM_HANDS = 2
STABLE_FRAMES = 5
MIN_SPEAK_CONFIDENCE = 0.45
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HAND_COLORS = ["#34C759", "#FF9500"]

# MediaPipe 21 关键点连线（前端绘制骨架用）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def ensure_hand_model() -> Path:
    if HAND_MODEL_PATH.exists():
        return HAND_MODEL_PATH
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    return HAND_MODEL_PATH


def create_hand_landmarker(model_path: Path) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def load_gesture_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = GestureMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, checkpoint["classes"], checkpoint["scaler_mean"], checkpoint["scaler_scale"]


def scale_features(vec: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((vec - mean) / scale).astype(np.float32)


@torch.no_grad()
def predict_gesture(model, features: np.ndarray) -> tuple[int, float]:
    x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1)
    conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()


class StablePredictor:
    def __init__(self, classes: list[str], stable_frames: int = STABLE_FRAMES):
        self.classes = classes
        self.stable_frames = stable_frames
        self.history: deque[int] = deque(maxlen=stable_frames)
        self.stable_label: str | None = None
        self.stable_conf: float = 0.0

    def reset(self) -> None:
        self.history.clear()
        self.stable_label = None
        self.stable_conf = 0.0

    def update(self, pred_idx: int, confidence: float) -> tuple[str, str, bool]:
        self.history.append(pred_idx)
        if len(self.history) < self.stable_frames:
            en = self.classes[pred_idx]
            n = len(self.history)
            return en, f"识别中 {n}/{self.stable_frames}", False
        if len(set(self.history)) == 1:
            self.stable_label = self.classes[pred_idx]
            self.stable_conf = confidence
            return self.stable_label, "", True
        if self.stable_label is not None:
            return self.stable_label, "切换手势中...", False
        en = self.classes[pred_idx]
        return en, f"识别中 {self.stable_frames}/{self.stable_frames}", False


def decode_frame(image_b64: str) -> np.ndarray | None:
    """Base64 JPEG → BGR numpy array."""
    if not image_b64:
        return None
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except (ValueError, cv2.error):
        return None


def _landmarks_to_json(hands: list) -> list[dict]:
    result = []
    for idx, hand_lm in enumerate(sort_hands_left_to_right(hands)):
        points = [{"x": lm.x, "y": lm.y} for lm in hand_lm]
        result.append({
            "color": HAND_COLORS[idx % len(HAND_COLORS)],
            "landmarks": points,
        })
    return result


def next_monotonic_timestamp_ms(last_ts: int) -> int:
    """MediaPipe VIDEO 模式要求时间戳严格单调递增，不能随会话重置为 0。"""
    ts = int(time.time() * 1000)
    return ts if ts > last_ts else last_ts + 1


def reset_hand_landmarker(landmarker: vision.HandLandmarker | None) -> vision.HandLandmarker:
    """关闭并重建 HandLandmarker，避免跨会话状态异常。"""
    if landmarker is not None:
        landmarker.close()
    return create_hand_landmarker(ensure_hand_model())


class GestureEngine:
    """Flask 会话内单例：处理浏览器上传的帧。"""

    def __init__(self) -> None:
        self._landmarker: vision.HandLandmarker | None = None
        self._gesture_model = None
        self._classes: list[str] = []
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._predictor: StablePredictor | None = None
        self._state_writer: GestureStateWriter | None = None
        self._stream_start = 0.0
        self._prev_time = 0.0
        self._last_timestamp_ms = 0
        self._last_announced: str | None = None
        self._active = False
        self._ready = False
        self._init_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def active(self) -> bool:
        return self._active

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def ensure_loaded(self) -> bool:
        if self._ready:
            return True
        if not GESTURE_MODEL_PATH.exists():
            self._init_error = "未找到 gesture_model.pth，请先运行 python train_gesture.py"
            return False
        try:
            self._gesture_model, self._classes, self._scaler_mean, self._scaler_scale = (
                load_gesture_model(GESTURE_MODEL_PATH)
            )
            if self._landmarker is None:
                self._landmarker = create_hand_landmarker(ensure_hand_model())
            self._predictor = StablePredictor(self._classes)
            if self._state_writer is None:
                self._state_writer = GestureStateWriter()
            self._ready = True
            self._init_error = None
            return True
        except Exception as exc:
            self._init_error = str(exc)
            return False

    def reload_model(self) -> bool:
        """训练完成后热加载新模型与类别列表。"""
        from gesture_speech import reload_gesture_map

        reload_gesture_map()
        self._ready = False
        self._gesture_model = None
        self._classes = []
        self._scaler_mean = None
        self._scaler_scale = None
        if self._predictor:
            self._predictor.reset()
        self._predictor = None
        self._last_announced = None
        return self.ensure_loaded()

    def start_session(self) -> dict:
        if not self.ensure_loaded():
            return {"ok": False, "error": self._init_error or "模型加载失败"}
        init_speech()
        self._landmarker = reset_hand_landmarker(self._landmarker)
        self._stream_start = time.time()
        self._prev_time = time.time()
        self._last_announced = None
        if self._predictor:
            self._predictor.reset()
        self._active = True
        if self._state_writer:
            self._state_writer.handle_commands()
            self._state_writer.flush(running=True)
        return {"ok": True}

    def stop_session(self) -> dict:
        self._active = False
        if self._state_writer:
            self._state_writer.shutdown()
        self._last_announced = None
        if self._predictor:
            self._predictor.reset()
        return {"ok": True}

    def process_frame(self, image_b64: str) -> dict:
        if not self._active or not self._ready:
            return self._error_payload("识别未启动")

        frame = decode_frame(image_b64)
        if frame is None:
            return self._error_payload("无法解码图像")

        assert self._landmarker is not None
        assert self._gesture_model is not None
        assert self._predictor is not None
        assert self._state_writer is not None
        assert self._scaler_mean is not None
        assert self._scaler_scale is not None

        self._state_writer.handle_commands()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._last_timestamp_ms = next_monotonic_timestamp_ms(self._last_timestamp_ms)
        result = self._landmarker.detect_for_video(mp_image, self._last_timestamp_ms)

        hands = result.hand_landmarks or []
        is_stable = False
        raw_label = ""
        conf = 0.0

        if hands:
            vec = hands_to_feature_vector(hands)
            vec_scaled = scale_features(vec, self._scaler_mean, self._scaler_scale)
            pred_idx, conf = predict_gesture(self._gesture_model, vec_scaled)
            raw_label, sub, is_stable = self._predictor.update(pred_idx, conf)

            if is_stable and conf >= MIN_SPEAK_CONFIDENCE:
                display = to_chinese(raw_label)
                sub = f"置信度 {conf * 100:.1f}%"
                if raw_label != self._last_announced:
                    self._last_announced = raw_label
                    request_speak(raw_label)
            elif is_stable:
                display = to_chinese(raw_label)
                sub = f"置信度偏低 {conf * 100:.1f}%"
            else:
                display = f"… {to_chinese(raw_label)}"
                if not sub:
                    sub = "识别中"
        else:
            self._predictor.reset()
            reset_speak_cache()
            self._last_announced = None
            display, sub = "未检测到手", "请将手伸入画面"

        now = time.time()
        fps = 1.0 / (now - self._prev_time) if now > self._prev_time else 0.0
        self._prev_time = now

        self._state_writer.update_frame(
            display_zh=display,
            sub_text=sub,
            raw_label=raw_label,
            is_stable=is_stable,
            confidence=conf,
            fps=fps,
            hands_detected=bool(hands),
        )

        state = self._state_writer_payload()
        state.update({
            "ok": True,
            "display_zh": display,
            "sub_text": sub,
            "raw_label": raw_label,
            "is_stable": is_stable,
            "confidence": conf,
            "fps": round(fps, 1),
            "hands": _landmarks_to_json(hands),
            "connections": HAND_CONNECTIONS,
        })
        return state

    def get_state(self) -> dict:
        if not self._state_writer:
            from gesture_state import read_runtime_state
            s = read_runtime_state()
            return {
                "ok": True,
                "active": self._active,
                "display_zh": s.display_zh,
                "sub_text": s.sub_text,
                "is_stable": s.is_stable,
                "confidence": s.confidence,
                "fps": s.fps,
                "pending_words": s.pending_words,
                "chat_log": s.chat_log,
                "polishing": s.polishing,
                "stats": s.stats,
                "hands": [],
                "connections": HAND_CONNECTIONS,
            }
        payload = self._state_writer_payload()
        payload["ok"] = True
        payload["active"] = self._active
        payload["hands"] = []
        payload["connections"] = HAND_CONNECTIONS
        return payload

    def clear_history(self) -> dict:
        from gesture_state import request_clear_history
        request_clear_history()
        if self._state_writer:
            self._state_writer.handle_commands()
        return {"ok": True}

    def _state_writer_payload(self) -> dict:
        assert self._state_writer is not None
        from gesture_state import read_runtime_state
        s = read_runtime_state()
        return {
            "display_zh": s.display_zh,
            "sub_text": s.sub_text,
            "raw_label": s.raw_label,
            "is_stable": s.is_stable,
            "confidence": s.confidence,
            "fps": s.fps,
            "pending_words": s.pending_words,
            "chat_log": s.chat_log,
            "polishing": s.polishing,
            "stats": s.stats,
        }

    @staticmethod
    def _error_payload(msg: str) -> dict:
        return {"ok": False, "error": msg, "display_zh": "—", "sub_text": msg}

    def close(self) -> None:
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None


_engine: GestureEngine | None = None


def get_engine() -> GestureEngine:
    global _engine
    if _engine is None:
        _engine = GestureEngine()
    return _engine
