"""
手势数据采集程序（双手）

操作流程：
  1~6   切换手势类别
  空格  开始一轮采集（仅空闲时有效）
  q     退出

每按一次空格自动完成一整轮：
  倒计时 3 秒（摆姿势，不保存）
  → 跳过开头 0.5 秒稳定期（不保存）
  → 连续保存 10 秒
  → 自动结束，可换类别或再按空格录下一段
"""

import csv
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
MAX_NUM_HANDS = 2
REQUIRE_BOTH_HANDS = False   # 重新录数据集时建议 True：必须双手入镜才保存
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
WINDOW_NAME = "Gesture Collection (2 Hands)"

RECORD_COUNTDOWN_SEC = 3    # 按空格后倒计时（秒）
WARMUP_SEC = 0.5            # 正式开始保存前跳过的稳定时间（秒）
AUTO_RECORD_SEC = 10        # 稳定期结束后，连续保存的时长（秒）

GESTURES = ["hello", "thanks", "help", "hospital", "danger", "me"]

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_CSV = OUTPUT_DIR / "gesture_data_2hands.csv"

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
NUM_LANDMARKS = 21
EMPTY_VALUE = ""

# idle | countdown | recording
_record_state = "idle"
_countdown_start = 0.0
_recording_start = 0.0      # 进入 recording 状态的时刻（含稳定期）
_save_start = 0.0           # 稳定期结束、开始写入 CSV 的时刻
_gesture_idx = 0


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"正在下载模型到 {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("模型下载完成。")
    return MODEL_PATH


def build_csv_header() -> list[str]:
    header = ["label", "num_hands"]
    for hand_id in (1, 2):
        for i in range(NUM_LANDMARKS):
            header.extend([f"h{hand_id}_lm{i}_x", f"h{hand_id}_lm{i}_y", f"h{hand_id}_lm{i}_z"])
    return header


def _landmarks_flat(landmarks) -> list:
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
    return row


def _empty_hand() -> list:
    return [EMPTY_VALUE] * (NUM_LANDMARKS * 3)


def sort_hands_left_to_right(hands: list) -> list:
    return sorted(hands, key=lambda lm: lm[0].x)


def hands_to_row(label: str, hand_landmarks_list: list) -> list | None:
    num_hands = len(hand_landmarks_list)
    if REQUIRE_BOTH_HANDS and num_hands < 2:
        return None

    sorted_hands = sort_hands_left_to_right(hand_landmarks_list)
    row = [label, num_hands]

    if num_hands >= 1:
        row.extend(_landmarks_flat(sorted_hands[0]))
    else:
        row.extend(_empty_hand())

    if num_hands >= 2:
        row.extend(_landmarks_flat(sorted_hands[1]))
    else:
        row.extend(_empty_hand())

    return row


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


def init_csv() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_CSV.exists():
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(build_csv_header())
        print(f"已创建数据文件：{OUTPUT_CSV}")


def append_row(row: list) -> None:
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def _start_countdown() -> None:
    global _record_state, _countdown_start
    _record_state = "countdown"
    _countdown_start = time.time()
    print(f"\n[{GESTURES[_gesture_idx]}] 倒计时 {RECORD_COUNTDOWN_SEC} 秒，请摆好手势...")


def _start_recording() -> None:
    global _record_state, _recording_start, _save_start
    _record_state = "recording"
    _recording_start = time.time()
    _save_start = 0.0
    print(f"[{GESTURES[_gesture_idx]}] 稳定期 {WARMUP_SEC} 秒（不保存）→ 随后录制 {AUTO_RECORD_SEC} 秒")


def _finish_recording() -> None:
    global _record_state, _save_start
    _record_state = "idle"
    _save_start = 0.0
    print(f"[{GESTURES[_gesture_idx]}] 本轮录制完成，可换类别或再按空格继续。\n")


def _cancel_countdown() -> None:
    global _record_state
    _record_state = "idle"
    print("已取消倒计时。")


def on_space_pressed() -> None:
    """空格：空闲时开始一轮；倒计时中可取消。"""
    if _record_state == "idle":
        _start_countdown()
    elif _record_state == "countdown":
        _cancel_countdown()
    # recording 过程中忽略空格，等待自动结束


def update_record_state() -> bool:
    """
    更新状态机。返回当前帧是否应写入 CSV。
    时间线：countdown(3s) → warmup(0.5s,不存) → saving(10s) → idle
    """
    global _record_state, _save_start

    now = time.time()

    if _record_state == "countdown":
        if now - _countdown_start >= RECORD_COUNTDOWN_SEC:
            _start_recording()
        return False

    if _record_state != "recording":
        return False

    elapsed = now - _recording_start

    # 稳定期内不保存
    if elapsed < WARMUP_SEC:
        return False

    # 稳定期刚结束，标记开始保存时刻
    if _save_start == 0.0:
        _save_start = now

    saved_duration = now - _save_start
    if saved_duration >= AUTO_RECORD_SEC:
        _finish_recording()
        return False

    return True


def get_countdown_display() -> int | None:
    if _record_state != "countdown":
        return None
    remaining = RECORD_COUNTDOWN_SEC - int(time.time() - _countdown_start)
    return max(1, remaining + 1)


def get_recording_progress() -> tuple[str, float] | None:
    """返回 (阶段说明, 进度0~1)，用于画面显示。"""
    if _record_state != "recording":
        return None

    elapsed = time.time() - _recording_start
    if elapsed < WARMUP_SEC:
        return ("WARMUP", elapsed / WARMUP_SEC)

    if _save_start == 0.0:
        return ("WARMUP", 1.0)

    saved = time.time() - _save_start
    return ("REC", min(saved / AUTO_RECORD_SEC, 1.0))


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


def draw_overlay(frame, gesture_idx: int, session_count: int, num_hands: int) -> None:
    gesture = GESTURES[gesture_idx]
    progress = get_recording_progress()

    if _record_state == "countdown":
        status = "GET READY"
        status_color = (0, 255, 255)
    elif progress and progress[0] == "WARMUP":
        status = f"WARMUP {WARMUP_SEC}s (not saving)"
        status_color = (0, 200, 255)
    elif progress and progress[0] == "REC":
        left = max(0.0, AUTO_RECORD_SEC - (time.time() - _save_start))
        status = f"REC {left:.1f}s / {AUTO_RECORD_SEC}s"
        status_color = (0, 0, 255)
    else:
        status = "READY (press SPACE)"
        status_color = (128, 128, 128)

    hand_text = f"Hands: {num_hands}/2" + (" OK" if num_hands >= 2 else "")

    lines = [
        f"Gesture [{gesture_idx + 1}/6]: {gesture}",
        f"Status: {status}  |  {hand_text}",
        f"Saved frames (session): {session_count}",
        f"SPACE start | 1-6 switch | q quit",
    ]

    y = 30
    for i, text in enumerate(lines):
        color = status_color if i == 1 else (0, 255, 0)
        cv2.putText(
            frame, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )
        y += 28

    countdown = get_countdown_display()
    if countdown is not None:
        cv2.putText(
            frame, str(countdown),
            (frame.shape[1] // 2 - 40, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8, cv2.LINE_AA,
        )

    if progress and progress[0] == "REC":
        bar_w = 300
        x0 = (frame.shape[1] - bar_w) // 2
        y0 = frame.shape[0] - 40
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 12), (80, 80, 80), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * progress[1]), y0 + 12), (0, 0, 255), -1)
        cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)


def print_help() -> None:
    print("\n========== 手势数据采集（双手）==========")
    for i, name in enumerate(GESTURES, start=1):
        print(f"  {i} -> {name}")
    print("  空格 -> 开始一轮采集（仅空闲时）")
    print("  q    -> 退出")
    print(f"  数据保存至: {OUTPUT_CSV}")
    print("\n  每按一次空格自动执行：")
    print(f"    倒计时 {RECORD_COUNTDOWN_SEC}s → 稳定期 {WARMUP_SEC}s → 录制 {AUTO_RECORD_SEC}s → 自动结束\n")


def main() -> None:
    global _gesture_idx

    print_help()
    init_csv()

    model_path = ensure_model()
    landmarker = create_hand_landmarker(model_path)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("无法打开摄像头。")
        landmarker.close()
        return

    cv2.namedWindow(WINDOW_NAME)

    _gesture_idx = 0
    session_count = 0
    start_time = time.time()
    hand_colors = [(0, 255, 0), (255, 128, 0)]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取画面。")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hands = result.hand_landmarks or []
            num_hands = len(hands)

            sorted_hands = sort_hands_left_to_right(hands) if hands else []
            for idx, hand_lm in enumerate(sorted_hands):
                color = hand_colors[idx] if idx < len(hand_colors) else (0, 255, 255)
                draw_hand_landmarks(frame, hand_lm, color)

            should_save = update_record_state()
            if should_save and hands:
                label = GESTURES[_gesture_idx]
                row = hands_to_row(label, hands)
                if row is not None:
                    append_row(row)
                    session_count += 1

            draw_overlay(frame, _gesture_idx, session_count, num_hands)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                on_space_pressed()
            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
                if _record_state != "idle":
                    print("请等待当前一轮录制结束后再切换类别。")
                else:
                    _gesture_idx = key - ord("1")
                    print(f"切换类别 -> {GESTURES[_gesture_idx]}")
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print(f"\n采集结束，本次共保存 {session_count} 帧。")
        print(f"数据文件：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
