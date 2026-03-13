from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_rank = np.argsort(np.argsort(y_true, kind="mergesort"), kind="mergesort").astype(float)
    pred_rank = np.argsort(np.argsort(y_pred, kind="mergesort"), kind="mergesort").astype(float)
    true_rank -= true_rank.mean()
    pred_rank -= pred_rank.mean()
    denom = np.sqrt((true_rank ** 2).sum() * (pred_rank ** 2).sum())
    return 0.0 if denom == 0 else float((true_rank * pred_rank).sum() / denom)


def _pairwise_user_scores(y_true: np.ndarray, y_pred: np.ndarray):
    n = len(y_true)
    concordant = 0.0
    discordant = 0.0
    evaluable = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            evaluable += 1.0
            true_sign = 1.0 if y_true[i] > y_true[j] else -1.0
            pred_diff = y_pred[i] - y_pred[j]
            if pred_diff == 0:
                concordant += 0.5
                discordant += 0.5
            elif pred_diff * true_sign > 0:
                concordant += 1.0
            else:
                discordant += 1.0
    return concordant, discordant, evaluable


def fcp_score(user_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total_concordant = 0.0
    total_pairs = 0.0
    for user_id in np.unique(user_ids):
        mask = user_ids == user_id
        if mask.sum() <= 1:
            continue
        concordant, _, evaluable = _pairwise_user_scores(y_true[mask], y_pred[mask])
        total_concordant += concordant
        total_pairs += evaluable
    return 0.0 if total_pairs == 0 else float(total_concordant / total_pairs)


def bpr_score(user_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    user_scores = []
    for user_id in np.unique(user_ids):
        mask = user_ids == user_id
        if mask.sum() <= 1:
            continue
        concordant, _, evaluable = _pairwise_user_scores(y_true[mask], y_pred[mask])
        if evaluable > 0:
            user_scores.append(concordant / evaluable)
    return 0.0 if not user_scores else float(np.mean(user_scores))
