"""
手势示意图存储（gesture_photo/{中文含义}.jpg）
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gesture_engine import decode_frame

GESTURE_PHOTO_DIR = Path(__file__).parent / "gesture_photo"


def _write_jpg(path: Path, frame: np.ndarray) -> None:
    """写入 JPG（避免 Windows 下 cv2.imwrite 中文路径失败）。"""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("图片编码失败")
    path.write_bytes(buf.tobytes())
    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError("图片保存失败")


def save_gesture_photo(zh: str, image_b64: str) -> Path:
    """将 base64 图像保存为 gesture_photo/{zh}.jpg。"""
    zh = zh.strip()
    if not zh:
        raise ValueError("中文含义不能为空")

    frame = decode_frame(image_b64)
    if frame is None:
        raise ValueError("无法解析图片，请上传 JPG/PNG 格式")

    GESTURE_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    path = GESTURE_PHOTO_DIR / f"{zh}.jpg"

    for old in GESTURE_PHOTO_DIR.glob(f"{zh}.*"):
        if old.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            old.unlink(missing_ok=True)

    _write_jpg(path, frame)
    return path


def save_gesture_photo_bytes(zh: str, raw: bytes) -> Path:
    """将二进制图像数据保存为 gesture_photo/{zh}.jpg。"""
    zh = zh.strip()
    if not zh:
        raise ValueError("中文含义不能为空")
    if not raw:
        raise ValueError("图片内容为空")

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法解析图片，请上传 JPG/PNG 格式")

    GESTURE_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    path = GESTURE_PHOTO_DIR / f"{zh}.jpg"

    for old in GESTURE_PHOTO_DIR.glob(f"{zh}.*"):
        if old.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            old.unlink(missing_ok=True)

    _write_jpg(path, frame)
    return path
