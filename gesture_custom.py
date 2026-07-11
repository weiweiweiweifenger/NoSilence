"""
自定义手势中文映射持久化（英文 label → 中文含义）
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

CUSTOM_MAP_PATH = Path(__file__).parent / "data" / "gesture_custom.json"

DEFAULT_GESTURE_ZH: dict[str, str] = {
    "hello": "你好",
    "thanks": "谢谢",
    "help": "帮助",
    "hospital": "医院",
    "danger": "危险",
    "me": "我",
}


def _load_custom_only() -> dict[str, str]:
    if not CUSTOM_MAP_PATH.exists():
        return {}
    try:
        data = json.loads(CUSTOM_MAP_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k).strip().lower(): str(v).strip() for k, v in data.items() if k and v}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def load_full_gesture_map() -> dict[str, str]:
    merged = dict(DEFAULT_GESTURE_ZH)
    merged.update(_load_custom_only())
    return merged


def save_custom_mapping(label: str, zh_meaning: str) -> None:
    label = label.strip().lower()
    zh_meaning = zh_meaning.strip()
    if not label or not zh_meaning:
        raise ValueError("label 与中文含义不能为空")

    custom = _load_custom_only()
    custom[label] = zh_meaning
    CUSTOM_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_MAP_PATH.write_text(
        json.dumps(custom, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_label_by_zh(zh_meaning: str) -> str | None:
    """根据中文含义查找已有手势的英文 label。"""
    zh_meaning = zh_meaning.strip()
    if not zh_meaning:
        return None
    for label, meaning in load_full_gesture_map().items():
        if meaning == zh_meaning:
            return label
    return None


def make_label_slug(zh_meaning: str, preferred: str | None = None) -> str:
    """生成唯一英文 label，供模型分类使用。"""
    if preferred:
        slug = re.sub(r"[^a-z0-9_]", "", preferred.strip().lower())
        if slug and slug not in DEFAULT_GESTURE_ZH:
            return slug

    base = re.sub(r"[^a-z0-9_]", "", zh_meaning.strip().lower()) or "gesture"
    if base not in DEFAULT_GESTURE_ZH and base not in _load_custom_only():
        return base[:32]

    return f"custom_{uuid.uuid4().hex[:8]}"
