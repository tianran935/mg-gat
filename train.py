from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.build_graph import build_graphs
from src.config import DEFAULT_CONFIG
from src.evaluate import bpr_score, fcp_score, rmse, spearman_corr
from src.model import MGGAT
from src.preprocess import build_preprocessed_dataset


def load_split(path: Path):
    frame = pd.read_csv(path)
    return (
        frame["user_idx"].to_numpy(dtype=np.int64),
        frame["business_idx"].to_numpy(dtype=np.int64),
        frame["stars"].to_numpy(dtype=np.float32),
    )


def merge_business_graphs(data_dir: Path):
    parts = [
        np.load(data_dir / "business_edges_geo.npy"),
        np.load(data_dir / "business_edges_cat.npy"),
        np.load(data_dir / "business_edges_covisit.npy"),
    ]
    edges = []
    edge_types = []
    for edge_type, arr in enumerate(parts):
        if arr.size == 0:
            continue
        edges.append(arr)
        edge_types.append(np.full(arr.shape[0], edge_type, dtype=np.int64))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.concatenate(edges, axis=0), np.concatenate(edge_types, axis=0)


def main():
    cfg = DEFAULT_CONFIG
    data_dir = cfg.data.processed_dir / "default_pa"
    stats = build_preprocessed_dataset(cfg)
    print("preprocess_stats", json.dumps(stats, indent=2))
    build_graphs()

    train_u, train_b, train_y = load_split(data_dir / "train.csv")
    valid_u, valid_b, valid_y = load_split(data_dir / "valid.csv")
    test_u, test_b, test_y = load_split(data_dir / "test.csv")

    global_mean = float(train_y.mean())
    print("global_mean_baseline_valid_rmse", rmse(valid_y, np.full_like(valid_y, global_mean)))
    print("global_mean_baseline_test_rmse", rmse(test_y, np.full_like(test_y, global_mean)))

    device_name = cfg.train.device if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu"
    device = torch.device(device_name)
    print("training_device", device)

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    user_features = torch.tensor(np.load(data_dir / "user_features.npy"), dtype=torch.float32, device=device)
    business_features = torch.tensor(np.load(data_dir / "business_features.npy"), dtype=torch.float32, device=device)
    user_edges = torch.tensor(np.load(data_dir / "user_edges.npy"), dtype=torch.long, device=device)
    business_edges_np, edge_types_np = merge_business_graphs(data_dir)
    business_edges = torch.tensor(business_edges_np, dtype=torch.long, device=device)
    business_edge_types = torch.tensor(edge_types_np, dtype=torch.long, device=device)

    train_u_t = torch.tensor(train_u, dtype=torch.long, device=device)
    train_b_t = torch.tensor(train_b, dtype=torch.long, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.float32, device=device)
    valid_u_t = torch.tensor(valid_u, dtype=torch.long, device=device)
    valid_b_t = torch.tensor(valid_b, dtype=torch.long, device=device)
    test_u_t = torch.tensor(test_u, dtype=torch.long, device=device)
    test_b_t = torch.tensor(test_b, dtype=torch.long, device=device)

    model = MGGAT(
        num_users=user_features.shape[0],
        num_businesses=business_features.shape[0],
        user_feat_dim=user_features.shape[1],
        business_feat_dim=business_features.shape[1],
        hidden_dim=cfg.train.hidden_dim,
        latent_dim=cfg.train.latent_dim,
        num_business_graphs=3,
        interpretable=cfg.train.interpretable,
        dropout=cfg.train.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_rmse = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred, _, _ = model(
            user_features,
            business_features,
            user_edges,
            business_edges,
            business_edge_types,
            train_u_t,
            train_b_t,
        )
        mse = ((pred - train_y_t) ** 2).mean()
        reg = cfg.train.theta1 * model.graph_regularization(user_edges, business_edges, cfg.train.theta2)
        loss = mse + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred, _, _ = model(
                user_features,
                business_features,
                user_edges,
                business_edges,
                business_edge_types,
                valid_u_t,
                valid_b_t,
            )
        val_rmse = rmse(valid_y, val_pred.detach().cpu().numpy())
        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_rmse": val_rmse, "lr": current_lr})
        print(f"epoch={epoch} train_loss={loss.item():.4f} val_rmse={val_rmse:.4f} lr={current_lr:.6f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.patience:
                print("early_stopping_triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    with torch.no_grad():
        test_pred, _, _ = model(
            user_features,
            business_features,
            user_edges,
            business_edges,
            business_edge_types,
            test_u_t,
            test_b_t,
        )
    test_pred_np = test_pred.detach().cpu().numpy()
    result = {
        "best_val_rmse": best_rmse,
        "best_epoch": best_epoch,
        "test_rmse": rmse(test_y, test_pred_np),
        "test_spearman": spearman_corr(test_y, test_pred_np),
        "test_fcp": fcp_score(test_u, test_y, test_pred_np),
        "test_bpr": bpr_score(test_u, test_y, test_pred_np),
        "paper_target_interpretable_rmse": 1.249,
        "paper_target_interpretable_spearman": 0.405,
        "paper_target_interpretable_fcp": 0.602,
        "paper_target_interpretable_bpr": 0.520,
        "paper_target_uninterpretable_rmse": 1.217,
        "paper_target_uninterpretable_spearman": 0.430,
        "paper_target_uninterpretable_fcp": 0.645,
        "paper_target_uninterpretable_bpr": 0.551,
        "history": history,
    }
    cfg.data.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data.artifacts_dir / "train_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
