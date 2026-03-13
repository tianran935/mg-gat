from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.build_graph import build_graphs
from src.evaluate import rmse, spearman_corr
from src.model import MGGAT
from src.preprocess import build_preprocessed_dataset


def load_split(path: Path):
    frame = pd.read_csv(path)
    return frame['user_idx'].to_numpy(dtype=np.int64), frame['business_idx'].to_numpy(dtype=np.int64), frame['stars'].to_numpy(dtype=np.float32)


def merge_business_graphs(data_dir: Path):
    parts = [
        np.load(data_dir / 'business_edges_geo.npy'),
        np.load(data_dir / 'business_edges_cat.npy'),
        np.load(data_dir / 'business_edges_covisit.npy'),
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
    data_dir = Path('/root/autodl-tmp/yelp/data/processed/default_pa')
    if not (data_dir / 'train.csv').exists():
        stats = build_preprocessed_dataset()
        print('preprocess_stats', json.dumps(stats, indent=2))
    else:
        print('reusing existing preprocessed data')
    if not (data_dir / 'user_edges.npy').exists():
        build_graphs()
    else:
        print('reusing existing graph files')

    user_features = torch.tensor(np.load(data_dir / 'user_features.npy'), dtype=torch.float32)
    business_features = torch.tensor(np.load(data_dir / 'business_features.npy'), dtype=torch.float32)
    user_edges = torch.tensor(np.load(data_dir / 'user_edges.npy'), dtype=torch.long)
    business_edges_np, edge_types_np = merge_business_graphs(data_dir)
    business_edges = torch.tensor(business_edges_np, dtype=torch.long)
    business_edge_types = torch.tensor(edge_types_np, dtype=torch.long)

    train_u, train_b, train_y = load_split(data_dir / 'train.csv')
    valid_u, valid_b, valid_y = load_split(data_dir / 'valid.csv')
    test_u, test_b, test_y = load_split(data_dir / 'test.csv')

    device = torch.device('cpu')
    model = MGGAT(
        num_users=user_features.shape[0],
        num_businesses=business_features.shape[0],
        user_feat_dim=user_features.shape[1],
        business_feat_dim=business_features.shape[1],
        hidden_dim=32,
        latent_dim=32,
        num_business_graphs=3,
        interpretable=True,
    ).to(device)
    user_features = user_features.to(device)
    business_features = business_features.to(device)
    user_edges = user_edges.to(device)
    business_edges = business_edges.to(device)
    business_edge_types = business_edge_types.to(device)
    train_u_t = torch.tensor(train_u, dtype=torch.long, device=device)
    train_b_t = torch.tensor(train_b, dtype=torch.long, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.float32, device=device)
    valid_u_t = torch.tensor(valid_u, dtype=torch.long, device=device)
    valid_b_t = torch.tensor(valid_b, dtype=torch.long, device=device)
    test_u_t = torch.tensor(test_u, dtype=torch.long, device=device)
    test_b_t = torch.tensor(test_b, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_rmse = float('inf')
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, 11):
        model.train()
        optimizer.zero_grad()
        pred, _, _ = model(user_features, business_features, user_edges, business_edges, business_edge_types, train_u_t, train_b_t)
        mse = ((pred - train_y_t) ** 2).mean()
        reg = 1e-4 * model.graph_regularization(user_edges, business_edges, 1e-4)
        loss = mse + reg
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred, _, _ = model(user_features, business_features, user_edges, business_edges, business_edge_types, valid_u_t, valid_b_t)
        val_rmse = rmse(valid_y, val_pred.cpu().numpy())
        history.append({'epoch': epoch, 'train_loss': float(loss.item()), 'val_rmse': val_rmse})
        print(f'epoch={epoch} train_loss={loss.item():.4f} val_rmse={val_rmse:.4f}')
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred, _, _ = model(user_features, business_features, user_edges, business_edges, business_edge_types, test_u_t, test_b_t)
    test_pred_np = test_pred.cpu().numpy()
    result = {
        'test_rmse': rmse(test_y, test_pred_np),
        'test_spearman': spearman_corr(test_y, test_pred_np),
        'paper_target_interpretable_rmse': 1.249,
        'paper_target_uninterpretable_rmse': 1.217,
        'history': history,
    }
    Path('/root/autodl-tmp/yelp/artifacts').mkdir(parents=True, exist_ok=True)
    Path('/root/autodl-tmp/yelp/artifacts/train_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
