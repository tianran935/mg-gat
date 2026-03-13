from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from src.config import DEFAULT_CONFIG


def build_graphs():
    data_dir = DEFAULT_CONFIG.data.processed_dir / 'default_pa'
    user_index = pd.read_csv(data_dir / 'user_index.csv')
    business_index = pd.read_csv(data_dir / 'business_index.csv')
    train_df = pd.read_csv(data_dir / 'train.csv')

    user_id_to_idx = dict(zip(user_index['user_id'], user_index['user_idx']))
    user_edges: List[Tuple[int, int]] = []
    for row in user_index.itertuples(index=False):
        src = int(row.user_idx)
        friends = row.friends if isinstance(row.friends, str) else ''
        if not friends or friends == 'None':
            continue
        nbrs = []
        for friend in [x.strip() for x in friends.split(',') if x.strip()]:
            dst = user_id_to_idx.get(friend)
            if dst is not None and dst != src:
                nbrs.append(int(dst))
        nbrs = nbrs[: DEFAULT_CONFIG.graph.max_user_neighbors]
        for dst in nbrs:
            user_edges.append((src, dst))

    coords = business_index[['latitude', 'longitude']].fillna(0.0).to_numpy(dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(DEFAULT_CONFIG.graph.business_k + 1, len(coords)), metric='euclidean')
    nn.fit(coords)
    geo_nbrs = nn.kneighbors(coords, return_distance=False)
    geo_edges = []
    for src, nbrs in enumerate(geo_nbrs):
        for dst in nbrs[1:]:
            geo_edges.append((src, int(dst)))

    category_lists = [sorted(set([c.strip() for c in str(cats).split(',') if c.strip()])) for cats in business_index['categories']]
    all_categories = sorted({c for cats in category_lists for c in cats})
    cat_index = {c: i for i, c in enumerate(all_categories)}
    cat_matrix = np.zeros((len(category_lists), len(all_categories)), dtype=np.float32)
    for i, cats in enumerate(category_lists):
        for c in cats:
            cat_matrix[i, cat_index[c]] = 1.0
    sims = cosine_similarity(cat_matrix)
    np.fill_diagonal(sims, -np.inf)
    cat_edges = []
    topk = min(DEFAULT_CONFIG.graph.business_k, sims.shape[1] - 1)
    for src in range(sims.shape[0]):
        nbrs = np.argpartition(-sims[src], topk)[:topk]
        for dst in nbrs:
            cat_edges.append((src, int(dst)))

    per_user = defaultdict(list)
    for row in train_df.itertuples(index=False):
        per_user[int(row.user_idx)].append(int(row.business_idx))
    covisit_scores = defaultdict(int)
    for businesses in per_user.values():
        unique = sorted(set(businesses))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                covisit_scores[(unique[i], unique[j])] += 1
    covisit_neighbors = defaultdict(list)
    for (a, b), score in covisit_scores.items():
        if score < DEFAULT_CONFIG.graph.covisit_min_shared:
            continue
        covisit_neighbors[a].append((b, score))
        covisit_neighbors[b].append((a, score))
    covisit_edges = []
    for src in range(len(business_index)):
        ranked = sorted(covisit_neighbors.get(src, []), key=lambda x: (-x[1], x[0]))[: DEFAULT_CONFIG.graph.covisit_k]
        for dst, _ in ranked:
            covisit_edges.append((src, dst))

    np.save(data_dir / 'user_edges.npy', np.asarray(user_edges, dtype=np.int64) if user_edges else np.zeros((0, 2), dtype=np.int64))
    np.save(data_dir / 'business_edges_geo.npy', np.asarray(geo_edges, dtype=np.int64))
    np.save(data_dir / 'business_edges_cat.npy', np.asarray(cat_edges, dtype=np.int64))
    np.save(data_dir / 'business_edges_covisit.npy', np.asarray(covisit_edges, dtype=np.int64))


if __name__ == '__main__':
    build_graphs()
    print('saved graph files')
