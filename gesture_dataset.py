"""
手势数据集 CSV 读写（采集与 Web 训练共用）
"""

from __future__ import annotations

import csv
from pathlib import Path

from gesture_features import NUM_LANDMARKS

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CSV = DATA_DIR / "gesture_data_2hands.csv"
EMPTY_VALUE = ""


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
    """MediaPipe 手部关键点 → CSV 行。"""
    num_hands = len(hand_landmarks_list)
    if num_hands < 1:
        return None

    sorted_hands = sort_hands_left_to_right(hand_landmarks_list)
    row = [label, num_hands]
    row.extend(_landmarks_flat(sorted_hands[0]))
    if num_hands >= 2:
        row.extend(_landmarks_flat(sorted_hands[1]))
    else:
        row.extend(_empty_hand())
    return row


def ensure_csv(csv_path: Path = DEFAULT_CSV) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(build_csv_header())
    return csv_path


def remove_label_rows(label: str, csv_path: Path | None = None) -> int:
    """从 CSV 中删除指定 label 的所有样本行。"""
    csv_path = csv_path or resolve_data_csv()
    if not csv_path.exists():
        return 0

    target = label.strip().lower()
    kept: list[list] = []
    removed = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        kept.append(header)
        for row in reader:
            if row and str(row[0]).strip().lower() == target:
                removed += 1
            else:
                kept.append(row)

    if removed:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(kept)
    return removed


def append_rows(rows: list[list], csv_path: Path = DEFAULT_CSV) -> int:
    ensure_csv(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    return len(rows)


def resolve_data_csv() -> Path:
    candidates = [
        DATA_DIR / "gesture_data_2hand.csv",
        DATA_DIR / "gesture_data_2hands.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return ensure_csv(DEFAULT_CSV)
