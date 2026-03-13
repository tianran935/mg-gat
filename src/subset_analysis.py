from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Set

from src.config import DEFAULT_CONFIG, ReproductionConfig
from src.utils import ensure_dir, json_lines


def build_business_subset(config: ReproductionConfig) -> Set[str]:
    checkin_ids: Set[str] = set()
    if config.require_checkins:
        for row in json_lines(config.data.checkin_json):
            checkin_ids.add(row['business_id'])

    business_ids: Set[str] = set()
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
        if config.require_checkins and row['business_id'] not in checkin_ids:
            continue
        business_ids.add(row['business_id'])
    return business_ids


def analyze_subset(config: ReproductionConfig) -> Dict:
    business_ids = build_business_subset(config)

    user_ids: Set[str] = set()
    review_count = 0
    yearly_reviews = Counter()
    per_user_review_count = Counter()
    for row in json_lines(config.data.review_json):
        if row['business_id'] not in business_ids:
            continue
        year = int(row['date'][:4])
        yearly_reviews[year] += 1
        if config.use_year_window_only and not (2009 <= year <= config.split.test_year):
            continue
        user_ids.add(row['user_id'])
        review_count += 1
        per_user_review_count[row['user_id']] += 1

    valid_friend_users: Set[str] = set()
    if config.require_friend_in_subset:
        for row in json_lines(config.data.user_json):
            uid = row['user_id']
            if uid not in user_ids:
                continue
            friends = row.get('friends') or ''
            if not friends or friends == 'None':
                continue
            parts = [x.strip() for x in friends.split(',') if x.strip()]
            if any(friend in user_ids for friend in parts):
                valid_friend_users.add(uid)
    else:
        valid_friend_users = set(user_ids)

    filtered_reviews = 0
    filtered_users: Set[str] = set()
    for row in json_lines(config.data.review_json):
        if row['business_id'] not in business_ids:
            continue
        year = int(row['date'][:4])
        if config.use_year_window_only and not (2009 <= year <= config.split.test_year):
            continue
        if row['user_id'] not in valid_friend_users:
            continue
        filtered_reviews += 1
        filtered_users.add(row['user_id'])

    largest_component_size = 0
    if valid_friend_users:
        adjacency = defaultdict(set)
        for row in json_lines(config.data.user_json):
            uid = row['user_id']
            if uid not in valid_friend_users:
                continue
            friends = row.get('friends') or ''
            if not friends or friends == 'None':
                continue
            for friend in [x.strip() for x in friends.split(',') if x.strip()]:
                if friend in valid_friend_users and friend != uid:
                    adjacency[uid].add(friend)
                    adjacency[friend].add(uid)

        visited: Set[str] = set()
        for uid in adjacency:
            if uid in visited:
                continue
            queue = deque([uid])
            visited.add(uid)
            size = 0
            while queue:
                cur = queue.popleft()
                size += 1
                for nb in adjacency[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            largest_component_size = max(largest_component_size, size)

    result = {
        'config': asdict(config),
        'business_count': len(business_ids),
        'user_count_before_friend_filter': len(user_ids),
        'review_count_before_friend_filter': review_count,
        'user_count_after_friend_filter': len(filtered_users),
        'review_count_after_friend_filter': filtered_reviews,
        'largest_friend_component_size': largest_component_size,
        'yearly_review_counts': dict(sorted(yearly_reviews.items())),
        'paper_target_table_d1': {
            'business_count': 10966,
            'user_count': 76865,
            'rating_count': 260350,
        },
    }
    return result


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    config = DEFAULT_CONFIG
    report = _jsonable(analyze_subset(config))
    ensure_dir(config.data.artifacts_dir)
    output_path = config.data.artifacts_dir / 'subset_analysis.json'
    output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
