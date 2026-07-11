"""
手势特征提取（训练与推理共用）
支持单手/双手混合 CSV：左手 h1 + 右手 h2（缺省填 0）+ num_hands 标志。
"""

import numpy as np
import pandas as pd

NUM_LANDMARKS = 21
LANDMARK_DIM = NUM_LANDMARKS * 3  # 63

H1_COLS = [f"h1_lm{i}_{a}" for i in range(NUM_LANDMARKS) for a in ("x", "y", "z")]
H2_COLS = [f"h2_lm{i}_{a}" for i in range(NUM_LANDMARKS) for a in ("x", "y", "z")]

# 126 维关键点 + 1 维手数标志
INPUT_DIM = LANDMARK_DIM * 2 + 1


def _landmarks_flat(landmarks) -> np.ndarray:
    vec = []
    for lm in landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec, dtype=np.float32)


def sort_hands_left_to_right(hands: list) -> list:
    return sorted(hands, key=lambda lm: lm[0].x)


def hands_to_feature_vector(hand_landmarks_list: list) -> np.ndarray:
    """
    MediaPipe 检测结果 → 与训练 CSV 一致的特征向量 (127,)。
    0~62: 左手/唯一手 h1；63~125: 右手 h2（单手时为 0）；126: num_hands/2
    """
    num_hands = len(hand_landmarks_list)
    h1 = np.zeros(LANDMARK_DIM, dtype=np.float32)
    h2 = np.zeros(LANDMARK_DIM, dtype=np.float32)

    if num_hands >= 1:
        sorted_hands = sort_hands_left_to_right(hand_landmarks_list)
        h1 = _landmarks_flat(sorted_hands[0])
        if num_hands >= 2:
            h2 = _landmarks_flat(sorted_hands[1])

    num_flag = np.array([num_hands / 2.0], dtype=np.float32)
    return np.concatenate([h1, h2, num_flag])


def dataframe_to_features(df: pd.DataFrame) -> np.ndarray:
    """从 DataFrame 构建特征矩阵，单手时 h2 空列填 0。"""
    h1 = df[H1_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).values
    h2 = df[H2_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).values
    num_hands = df["num_hands"].fillna(1).astype(np.float32).values.reshape(-1, 1) / 2.0
    return np.hstack([h1, h2, num_hands]).astype(np.float32)
