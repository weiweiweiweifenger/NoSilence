"""
手势分类模型训练（PyTorch MLP）
输入：双手 CSV（单手时第二只手填 0）+ num_hands 标志 → 127 维
数据：data/gesture_data_2hands.csv（也兼容 gesture_data_2hand.csv）
输出：gesture_model.pth、测试集准确率
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from gesture_dataset import resolve_data_csv
from gesture_features import INPUT_DIM, dataframe_to_features

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "gesture_model.pth"

HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_SEED = 42


class GestureMLP(nn.Module):
    """多层感知机：关键点向量 → 手势类别。"""

    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label"])

    X = dataframe_to_features(df)
    y_raw = df["label"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = list(label_encoder.classes_)

    return X, y, classes, len(df)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        correct += (logits.argmax(dim=1) == batch_y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def train_gesture_model(
    csv_path: Path | None = None,
    *,
    device: torch.device | str | None = None,
    epochs: int = EPOCHS,
    model_path: Path = MODEL_PATH,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """
    训练手势模型并保存 checkpoint。
    Web 训练默认使用 CPU。
    """
    csv_path = csv_path or resolve_data_csv()
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {csv_path}")

    train_device = torch.device(device) if device is not None else torch.device("cpu")
    X, y, classes, sample_count = load_dataset(csv_path)
    num_classes = len(classes)

    if num_classes < 2:
        raise ValueError("至少需要 2 个手势类别才能训练，请先采集更多数据")

    if sample_count < num_classes * 2:
        raise ValueError(f"样本过少（{sample_count} 条），建议每个手势至少采集 10 次")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    stratify = y if num_classes > 1 and len(y) >= 4 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = GestureMLP(INPUT_DIM, HIDDEN_DIMS, num_classes, DROPOUT).to(train_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, train_device)
        test_loss, test_acc = evaluate(model, test_loader, train_device)
        best_acc = max(best_acc, test_acc)

        if progress_callback:
            progress_callback({
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": round(train_loss, 4),
                "test_loss": round(test_loss, 4),
                "test_acc": round(test_acc * 100, 2),
            })

    _, final_test_acc = evaluate(model, test_loader, train_device)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "num_classes": num_classes,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "classes": classes,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "data_format": "2hands_mixed",
    }
    torch.save(checkpoint, model_path)

    return {
        "ok": True,
        "test_acc": round(final_test_acc * 100, 2),
        "best_acc": round(best_acc * 100, 2),
        "classes": classes,
        "num_classes": num_classes,
        "sample_count": sample_count,
        "device": str(train_device),
        "model_path": str(model_path),
    }


def main() -> None:
    csv_path = resolve_data_csv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"数据文件: {csv_path.name}")
    print(f"使用设备: {device}\n")

    def on_progress(info: dict) -> None:
        epoch = info["epoch"]
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{info['total_epochs']} | "
                f"train_loss={info['train_loss']:.4f} | "
                f"test_loss={info['test_loss']:.4f} | "
                f"test_acc={info['test_acc']:.2f}%"
            )

    result = train_gesture_model(csv_path, device=device, progress_callback=on_progress)
    print(f"\n测试集准确率: {result['test_acc']:.2f}%")
    print(f"最佳测试准确率: {result['best_acc']:.2f}%")
    print(f"类别: {result['classes']}")
    print(f"模型已保存: {result['model_path']}")


if __name__ == "__main__":
    main()
