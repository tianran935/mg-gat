from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_rank = np.argsort(np.argsort(y_true, kind='mergesort'), kind='mergesort').astype(float)
    pred_rank = np.argsort(np.argsort(y_pred, kind='mergesort'), kind='mergesort').astype(float)
    true_rank -= true_rank.mean()
    pred_rank -= pred_rank.mean()
    denom = np.sqrt((true_rank ** 2).sum() * (pred_rank ** 2).sum())
    return 0.0 if denom == 0 else float((true_rank * pred_rank).sum() / denom)
