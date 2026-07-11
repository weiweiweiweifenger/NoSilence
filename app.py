"""
无声之声 · Voice of Silence
Flask Web 服务 — 浏览器摄像头 + MediaPipe + MLP + TTS

运行: python app.py
浏览器: http://127.0.0.1:5000
"""

from __future__ import annotations

import env_loader  # noqa: F401 — 启动时加载 .env

from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from gesture_engine import GESTURE_MODEL_PATH, HAND_MODEL_PATH, get_engine
from gesture_train_service import get_train_service

app = Flask(__name__)
GESTURE_PHOTO_DIR = Path(__file__).parent / "gesture_photo"
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    from gesture_polish import get_polish_info

    engine = get_engine()
    polish = get_polish_info()
    return jsonify({
        "ok": True,
        "model_exists": GESTURE_MODEL_PATH.exists(),
        "hand_model_exists": HAND_MODEL_PATH.exists(),
        "engine_ready": engine.ready,
        "active": engine.active,
        "error": engine.init_error,
        "polish_provider": polish["provider"],
        "polish_label": polish["label"],
        "polish_model": polish["model"],
    })


@app.route("/api/start", methods=["POST"])
def start():
    engine = get_engine()
    result = engine.start_session()
    status = 200 if result.get("ok") else 503
    return jsonify(result), status


@app.route("/api/stop", methods=["POST"])
def stop():
    engine = get_engine()
    return jsonify(engine.stop_session())


@app.route("/api/frame", methods=["POST"])
def frame():
    engine = get_engine()
    data = request.get_json(silent=True) or {}
    image = data.get("image", "")
    if not image:
        return jsonify({"ok": False, "error": "缺少图像数据"}), 400
    result = engine.process_frame(image)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/state")
def state():
    engine = get_engine()
    return jsonify(engine.get_state())


@app.route("/api/clear", methods=["POST"])
def clear():
    engine = get_engine()
    return jsonify(engine.clear_history())


@app.route("/gesture_photo/<path:filename>")
def gesture_photo(filename):
    return send_from_directory(GESTURE_PHOTO_DIR, filename)


def _find_photo_url(zh: str) -> str | None:
    for ext in PHOTO_EXTENSIONS:
        if (GESTURE_PHOTO_DIR / f"{zh}{ext}").exists():
            from urllib.parse import quote
            return f"/gesture_photo/{quote(f'{zh}{ext}')}"
    return None


def _list_recognized_gestures() -> list[dict]:
    from gesture_speech import GESTURE_ZH

    if GESTURE_MODEL_PATH.exists():
        import torch
        checkpoint = torch.load(GESTURE_MODEL_PATH, map_location="cpu", weights_only=False)
        labels = list(checkpoint.get("classes", []))
    else:
        labels = list(GESTURE_ZH.keys())

    gestures = []
    for label in labels:
        key = str(label).strip().lower()
        zh = GESTURE_ZH.get(key, label)
        gestures.append({
            "label": label,
            "zh": zh,
            "photo_url": _find_photo_url(zh),
        })
    return gestures


@app.route("/api/gestures")
def gestures():
    return jsonify({"ok": True, "gestures": _list_recognized_gestures()})


@app.route("/api/gesture-map")
def gesture_map():
    from gesture_speech import GESTURE_ZH
    return jsonify(GESTURE_ZH)


@app.route("/api/train/status")
def train_status():
    return jsonify({"ok": True, **get_train_service().get_status()})


@app.route("/api/train/begin", methods=["POST"])
def train_begin():
    data = request.get_json(silent=True) or {}
    zh_meaning = data.get("zh_meaning", "")
    label_hint = data.get("label_hint")
    overwrite = bool(data.get("overwrite"))
    replace_label = data.get("replace_label")
    result = get_train_service().begin(
        zh_meaning,
        label_hint,
        overwrite=overwrite,
        replace_label=replace_label,
    )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/train/sample", methods=["POST"])
def train_sample():
    data = request.get_json(silent=True) or {}
    image = data.get("image", "")
    if not image:
        return jsonify({"ok": False, "error": "缺少图像数据"}), 400
    result = get_train_service().capture_sample(image)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/train/preview", methods=["POST"])
def train_preview():
    data = request.get_json(silent=True) or {}
    image = data.get("image", "")
    if not image:
        return jsonify({"ok": False, "error": "缺少图像数据", "hands": [], "num_hands": 0}), 400
    result = get_train_service().preview_frame(image)
    return jsonify(result)


@app.route("/api/train/photo", methods=["POST"])
def train_photo():
    service = get_train_service()
    if request.files.get("photo"):
        result = service.upload_photo(raw_bytes=request.files["photo"].read())
    else:
        data = request.get_json(silent=True) or {}
        image = data.get("image", "")
        if not image:
            return jsonify({"ok": False, "error": "缺少图片数据"}), 400
        result = service.upload_photo(image_b64=image)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/train/run", methods=["POST"])
def train_run():
    result = get_train_service().start_training()
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/train/cancel", methods=["POST"])
def train_cancel():
    return jsonify(get_train_service().cancel())


if __name__ == "__main__":
    print("=" * 50)
    print("无声之声 · Voice of Silence")
    print("打开浏览器: http://127.0.0.1:5000")
    print("点击「开始手语识别」启动摄像头")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, load_dotenv=False)
