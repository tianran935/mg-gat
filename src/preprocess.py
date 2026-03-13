from __future__ import annotations

import ast
import json
from datetime import datetime
from collections import Counter, defaultdict, deque
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.config import DEFAULT_CONFIG, ReproductionConfig
from src.utils import ensure_dir, json_lines


def _parse_categories(value: str) -> List[str]:
    if not isinstance(value, str) or not value:
        return []
    return [token.strip() for token in value.split(',') if token.strip()]


def _parse_attributes(value) -> Dict[str, float]:
    if not value or value == 'None':
        return {}
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = ast.literal_eval(value)
        except Exception:
            return {}
    parsed: Dict[str, float] = {}
    for key, item in raw.items():
        if isinstance(item, dict):
            for sub_key, sub_val in item.items():
                parsed[f'{key}:{sub_key}'] = float(_to_numeric(sub_val))
        else:
            parsed[str(key)] = float(_to_numeric(item))
    return parsed


def _to_numeric(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if value in ('True', 'true', 'yes', '1'):
        return 1
    if value in ('False', 'false', 'no', '0', 'None', None):
        return 0
    try:
        return int(float(value))
    except Exception:
        return 1


def _parse_hours(value) -> np.ndarray:
    vec = np.zeros(14, dtype=np.float32)
    if not value or value == 'None':
        return vec
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = ast.literal_eval(value)
        except Exception:
            return vec
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for idx, day in enumerate(days):
        if day not in raw:
            continue
        try:
            start, end = raw[day].split('-')
            vec[2 * idx] = _time_to_hour(start)
            vec[2 * idx + 1] = _time_to_hour(end)
        except Exception:
            continue
    return vec


def _time_to_hour(token: str) -> float:
    hour, minute = token.split(':')
    return float(hour) + float(minute) / 60.0


def _standardize(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (arr - mean) / std


def _load_checkin_vectors(config: ReproductionConfig, business_ids: Set[str]) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    slot_cache: Dict[str, int] = {}
    for row in json_lines(config.data.checkin_json):
        bid = row['business_id']
        if bid not in business_ids:
            continue
        vec = np.zeros(168, dtype=np.float32)
        raw = row.get('date')
        if isinstance(raw, str) and raw:
            for stamp in [x.strip() for x in raw.split(',') if x.strip()]:
                slot = slot_cache.get(stamp)
                if slot is None:
                    dt = datetime.fromisoformat(stamp)
                    slot = dt.weekday() * 24 + dt.hour
                    slot_cache[stamp] = slot
                vec[slot] += 1.0
        vectors[bid] = vec
    return vectors


def build_preprocessed_dataset(config: ReproductionConfig = DEFAULT_CONFIG) -> Dict[str, int]:
    out_dir = config.data.processed_dir / 'default_pa'
    ensure_dir(out_dir)

    businesses: Dict[str, dict] = {}
    for row in json_lines(config.data.business_json):
        categories = row.get('categories') or ''
        if row.get('state') != config.region:
            continue
        if config.business_category_token not in categories:
            continue
        if config.require_hours and not row.get('hours'):
            continue
        if config.require_attributes and not row.get('attributes'):
            continue
        businesses[row['business_id']] = row

    checkin_vectors = _load_checkin_vectors(config, set(businesses))
    if config.require_checkins:
        businesses = {bid: row for bid, row in businesses.items() if bid in checkin_vectors}

    review_rows = []
    candidate_users: Set[str] = set()
    for row in json_lines(config.data.review_json):
        bid = row['business_id']
        if bid not in businesses:
            continue
        year = int(row['date'][:4])
        if config.use_year_window_only and not (2009 <= year <= config.split.test_year):
            continue
        review_rows.append({
            'user_id': row['user_id'],
            'business_id': bid,
            'stars': float(row['stars']),
            'date': row['date'],
            'year': year,
        })
        candidate_users.add(row['user_id'])

    user_records: Dict[str, dict] = {}
    adjacency = defaultdict(set)
    for row in json_lines(config.data.user_json):
        uid = row['user_id']
        if uid not in candidate_users:
            continue
        user_records[uid] = row
        friends = row.get('friends') or ''
        if not friends or friends == 'None':
            continue
        for friend in [x.strip() for x in friends.split(',') if x.strip()]:
            if friend in candidate_users and friend != uid:
                adjacency[uid].add(friend)

    kept_users = set(candidate_users)
    if config.require_friend_in_subset:
        kept_users = {u for u in candidate_users if any(f in candidate_users for f in adjacency.get(u, set()))}

    if config.use_largest_friend_component and kept_users:
        undirected = defaultdict(set)
        for u in kept_users:
            for v in adjacency.get(u, set()):
                if v in kept_users:
                    undirected[u].add(v)
                    undirected[v].add(u)
        visited = set()
        giant = set()
        for u in undirected:
            if u in visited:
                continue
            queue = deque([u])
            visited.add(u)
            comp = []
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nb in undirected[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if len(comp) > len(giant):
                giant = set(comp)
        if giant:
            kept_users = giant

    review_df = pd.DataFrame([r for r in review_rows if r['user_id'] in kept_users]).reset_index(drop=True)
    user_ids = sorted(review_df['user_id'].unique().tolist())
    business_ids = sorted(review_df['business_id'].unique().tolist())

    user_df = pd.DataFrame([user_records[uid] for uid in user_ids]).reset_index(drop=True)
    business_df = pd.DataFrame([businesses[bid] for bid in business_ids]).reset_index(drop=True)

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    business_to_idx = {bid: idx for idx, bid in enumerate(business_ids)}
    review_df['user_idx'] = review_df['user_id'].map(user_to_idx)
    review_df['business_idx'] = review_df['business_id'].map(business_to_idx)

    category_counter = Counter()
    attr_counter = Counter()
    parsed_categories = []
    parsed_attrs = []
    checkin_matrix = np.zeros((len(business_df), 168), dtype=np.float32)
    for idx, row in business_df.iterrows():
        cats = _parse_categories(row.get('categories'))
        attrs = _parse_attributes(row.get('attributes'))
        parsed_categories.append(cats)
        parsed_attrs.append(attrs)
        category_counter.update(cats)
        attr_counter.update(attrs.keys())
        checkin_matrix[idx] = checkin_vectors.get(row['business_id'], np.zeros(168, dtype=np.float32))

    kept_categories = sorted([k for k, v in category_counter.items() if v >= config.min_category_freq])
    kept_attrs = sorted([k for k, v in attr_counter.items() if v >= config.min_attribute_freq])
    cat_index = {name: idx for idx, name in enumerate(kept_categories)}
    attr_index = {name: idx for idx, name in enumerate(kept_attrs)}

    category_matrix = np.zeros((len(business_df), len(kept_categories)), dtype=np.float32)
    attr_matrix = np.zeros((len(business_df), len(kept_attrs)), dtype=np.float32)
    hour_matrix = np.zeros((len(business_df), 14), dtype=np.float32)
    location_matrix = business_df[['latitude', 'longitude']].fillna(0.0).to_numpy(dtype=np.float32)
    for idx, row in enumerate(business_df.itertuples(index=False)):
        for cat in parsed_categories[idx]:
            loc = cat_index.get(cat)
            if loc is not None:
                category_matrix[idx, loc] = 1.0
        for key, value in parsed_attrs[idx].items():
            loc = attr_index.get(key)
            if loc is not None:
                attr_matrix[idx, loc] = value
        hour_matrix[idx] = _parse_hours(getattr(row, 'hours'))

    user_numeric_cols = [
        'review_count', 'average_stars', 'fans', 'useful', 'funny', 'cool',
        'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
        'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
        'compliment_funny', 'compliment_writer', 'compliment_photos',
    ]
    user_feature_matrix = user_df[user_numeric_cols].fillna(0.0).to_numpy(dtype=np.float32)
    elite_count = user_df['elite'].fillna('').apply(lambda x: 0 if not x or x == 'None' else len([v for v in str(x).split(',') if v.strip()]))
    friend_count = user_df['friends'].fillna('').apply(lambda x: 0 if not x or x == 'None' else len([v for v in str(x).split(',') if v.strip()]))
    user_feature_matrix = np.concatenate([
        user_feature_matrix,
        elite_count.to_numpy(dtype=np.float32).reshape(-1, 1),
        friend_count.to_numpy(dtype=np.float32).reshape(-1, 1),
    ], axis=1)

    interaction = sparse.csr_matrix((np.ones(len(review_df), dtype=np.float32), (review_df['user_idx'].to_numpy(), review_df['business_idx'].to_numpy())), shape=(len(user_ids), len(business_ids)), dtype=np.float32)
    n_components = min(config.implicit_dim, min(interaction.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=config.train.seed)
    user_implicit = svd.fit_transform(interaction).astype(np.float32)
    business_implicit = svd.components_.T.astype(np.float32)

    user_features = _standardize(np.concatenate([user_feature_matrix, user_implicit], axis=1)).astype(np.float32)
    business_features = _standardize(np.concatenate([attr_matrix, category_matrix, hour_matrix, location_matrix, checkin_matrix, business_implicit], axis=1)).astype(np.float32)

    split_map = {
        'train': review_df[review_df['year'] <= config.split.train_end_year],
        'valid': review_df[review_df['year'] == config.split.valid_year],
        'test': review_df[review_df['year'] == config.split.test_year],
    }
    for name, frame in split_map.items():
        frame[['user_idx', 'business_idx', 'stars']].to_csv(out_dir / f'{name}.csv', index=False)

    np.save(out_dir / 'user_features.npy', user_features)
    np.save(out_dir / 'business_features.npy', business_features)
    user_df[['user_id', 'friends']].assign(user_idx=np.arange(len(user_df))).to_csv(out_dir / 'user_index.csv', index=False)
    business_df[['business_id', 'latitude', 'longitude', 'categories']].assign(business_idx=np.arange(len(business_df))).to_csv(out_dir / 'business_index.csv', index=False)

    return {
        'users': len(user_ids),
        'businesses': len(business_ids),
        'reviews': len(review_df),
        'train_reviews': len(split_map['train']),
        'valid_reviews': len(split_map['valid']),
        'test_reviews': len(split_map['test']),
    }


if __name__ == '__main__':
    stats = build_preprocessed_dataset(DEFAULT_CONFIG)
    print(json.dumps(stats, indent=2))
