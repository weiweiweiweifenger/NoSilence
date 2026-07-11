"""
实时手部检测程序
使用 OpenCV 读取摄像头，MediaPipe Hand Landmarker（Tasks API）检测手部并绘制 21 个关键点。
按 q 键退出。

说明：MediaPipe 0.10+ 已移除 mp.solutions，本程序使用新版 Tasks API。
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

# ---------------------------------------------------------------------------
# 配置参数（可按需修改）
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
WINDOW_NAME = "Hand Detection"

# 模型文件会下载到脚本同目录
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# 21 个关键点之间的骨架连线
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


def ensure_model() -> Path:
    """若本地没有模型文件，则从官方地址自动下载。"""
    if MODEL_PATH.exists():
        return MODEL_PATH

    print(f"正在下载手部检测模型到 {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("模型下载完成。")
    return MODEL_PATH


def calculate_fps(prev_time: float) -> tuple[float, float]:
    """根据上一帧时间戳计算当前 FPS，并返回 (fps, 当前时间)。"""
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if current_time > prev_time else 0.0
    return fps, current_time


def draw_fps(frame, fps: float) -> None:
    """在画面左上角绘制 FPS。"""
    text = f"FPS: {int(fps)}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_hand_landmarks(frame, landmarks) -> None:
    """
    在画面上绘制单只手的 21 个关键点及骨架连线。
    landmarks: 长度为 21 的关键点列表，坐标为归一化值（0~1）
    """
    h, w, _ = frame.shape

    for connection in HAND_CONNECTIONS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


def create_hand_landmarker(model_path: Path) -> vision.HandLandmarker:
    """创建手部关键点检测器（视频流模式，适合摄像头）。"""
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def main() -> None:
    model_path = ensure_model()
    landmarker = create_hand_landmarker(model_path)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("无法打开摄像头，请检查设备是否连接或被占用。")
        landmarker.close()
        return

    print("手部检测已启动。将手放入画面，按 q 键退出。")
    prev_time = time.time()
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面。")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 视频模式需要单调递增的时间戳（毫秒）
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks)

            fps, prev_time = calculate_fps(prev_time)
            draw_fps(frame, fps)

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("程序已退出。")


if __name__ == "__main__":
    main()
