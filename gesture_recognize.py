"""
实时手势识别程序（支持单手/双手混合模型）
摄像头 → MediaPipe → MLP → 稳定后显示中文并语音播报
按 q 退出。
"""

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
from mediapipe.tasks.python.vision import HandLandmarksConnections
from PIL import Image, ImageDraw, ImageFont

from gesture_features import hands_to_feature_vector, sort_hands_left_to_right
from gesture_speech import init_speech, request_speak, reset_speak_cache, shutdown_speech, to_chinese
from gesture_state import GestureStateWriter
from train_gesture import GestureMLP

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
MAX_NUM_HANDS = 2
STABLE_FRAMES = 5
MIN_SPEAK_CONFIDENCE = 0.45   # 置信度过低不播报
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
WINDOW_NAME = "Gesture Recognition"

HAND_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
GESTURE_MODEL_PATH = Path(__file__).parent / "gesture_model.pth"

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAND_COLORS = [(0, 255, 0), (255, 128, 0)]

# Windows 中文字体路径
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


_FONT_LARGE = _load_font(48)
_FONT_SMALL = _load_font(24)
_FONT_TINY = _load_font(22)


def put_text(frame, text: str, xy: tuple[int, int], font, color_bgr: tuple[int, int, int]) -> None:
    """在 OpenCV 画面上绘制中文（PIL），避免问号。"""
    if not text:
        return
    x, y = xy
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=font, fill=color_rgb)
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def ensure_hand_model() -> Path:
    if HAND_MODEL_PATH.exists():
        return HAND_MODEL_PATH
    print("正在下载 MediaPipe 手部模型...")
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


def draw_hand_landmarks(frame, landmarks, color) -> None:
    h, w, _ = frame.shape
    for connection in HAND_CONNECTIONS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, color, -1)


def draw_result(frame, display_text: str, sub_text: str, is_stable: bool, fps: float) -> None:
    color = (0, 255, 0) if is_stable else (0, 200, 255)
    cv2.putText(
        frame, f"FPS: {int(fps)}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
    )
    put_text(frame, display_text, (10, 45), _FONT_LARGE, color)
    put_text(frame, sub_text, (10, 105), _FONT_SMALL, color)


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


def main() -> None:
    if not GESTURE_MODEL_PATH.exists():
        print(f"找不到模型: {GESTURE_MODEL_PATH}，请先运行 python train_gesture.py")
        return

    print(f"设备: {DEVICE}")
    gesture_model, classes, scaler_mean, scaler_scale = load_gesture_model(GESTURE_MODEL_PATH)
    print(f"类别: {classes}")

    landmarker = create_hand_landmarker(ensure_hand_model())
    predictor = StablePredictor(classes)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("无法打开摄像头。")
        landmarker.close()
        return

    print("正在加载语音缓存（首次需联网生成，之后秒开）...")
    init_speech()

    print(f"识别已启动（稳定 {STABLE_FRAMES} 帧后显示中文并播报），按 q 退出。")
    print("Streamlit UI: streamlit run gesture_web.py")
    prev_time = time.time()
    stream_start = time.time()
    last_announced: str | None = None
    state_writer = GestureStateWriter()

    try:
        while True:
            state_writer.handle_commands()

            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - stream_start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hands = result.hand_landmarks or []
            is_stable = False
            raw_label = ""
            conf = 0.0

            if hands:
                for idx, hand_lm in enumerate(sort_hands_left_to_right(hands)):
                    draw_hand_landmarks(frame, hand_lm, HAND_COLORS[idx % len(HAND_COLORS)])

                vec = hands_to_feature_vector(hands)
                vec_scaled = scale_features(vec, scaler_mean, scaler_scale)
                pred_idx, conf = predict_gesture(gesture_model, vec_scaled)
                raw_label, sub, is_stable = predictor.update(pred_idx, conf)

                if is_stable and conf >= MIN_SPEAK_CONFIDENCE:
                    display = to_chinese(raw_label)
                    sub = f"置信度 {conf * 100:.1f}%"
                    if raw_label != last_announced:
                        last_announced = raw_label
                        request_speak(raw_label)
                elif is_stable:
                    display = to_chinese(raw_label)
                    sub = f"置信度偏低 {conf * 100:.1f}%"
                else:
                    display = f"... {to_chinese(raw_label)}"
                    if not sub:
                        sub = f"识别中"
            else:
                predictor.reset()
                reset_speak_cache()
                last_announced = None
                display, sub = "未检测到手", "请将手伸入画面"

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now
            draw_result(frame, display, sub, is_stable, fps)
            state_writer.update_frame(
                display_zh=display,
                sub_text=sub,
                raw_label=raw_label,
                is_stable=is_stable,
                confidence=conf,
                fps=fps,
                hands_detected=bool(hands),
            )
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        state_writer.shutdown()
        cap.release()
        landmarker.close()
        shutdown_speech()
        cv2.destroyAllWindows()
        print("已退出。")


if __name__ == "__main__":
    main()
