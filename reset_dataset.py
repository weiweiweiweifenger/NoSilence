"""
重置手势数据集：将现有 CSV 备份到 data/backup/，并创建新的空数据文件。

用法：
  python reset_dataset.py
"""

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
BACKUP_DIR = DATA_DIR / "backup"
# 重新录制后写入此文件（与 gesture_collect.py 中 OUTPUT_CSV 保持一致）
NEW_CSV = DATA_DIR / "gesture_data_2hands.csv"

HEADER = ["label", "num_hands"]
for hand_id in (1, 2):
    for i in range(21):
        HEADER.extend([f"h{hand_id}_lm{i}_x", f"h{hand_id}_lm{i}_y", f"h{hand_id}_lm{i}_z"])


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / stamp
    backup_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(DATA_DIR.glob("*.csv"))
    if csv_files:
        print("正在备份旧数据...")
        for f in csv_files:
            dest = backup_path / f.name
            shutil.copy2(f, dest)
            try:
                df = pd.read_csv(f)
                counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
                print(f"  {f.name}: {len(df)} 行  {counts}")
            except Exception:
                print(f"  {f.name}: 已备份")
        print(f"备份目录: {backup_path}\n")
    else:
        print("data/ 下没有 CSV，跳过备份。\n")

    pd.DataFrame(columns=HEADER).to_csv(NEW_CSV, index=False, encoding="utf-8")
    print(f"已创建新数据集: {NEW_CSV}")
    print("表头已写好，可直接运行: python gesture_collect.py")
    print("\n建议每类录制 8~12 轮（每轮约 10 秒 ≈ 300 帧），六类尽量数量接近。")


if __name__ == "__main__":
    main()
