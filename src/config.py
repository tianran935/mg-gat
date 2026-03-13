from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class DataConfig:
    raw_dir: Path = Path('/root/autodl-tmp/mg-gat/data/raw/Yelp JSON')
    business_json: Path = Path('/root/autodl-tmp/mg-gat/data/raw/Yelp JSON/yelp_academic_dataset_business.json')
    user_json: Path = Path('/root/autodl-tmp/mg-gat/data/raw/Yelp JSON/yelp_academic_dataset_user.json')
    review_json: Path = Path('/root/autodl-tmp/mg-gat/data/raw/Yelp JSON/yelp_academic_dataset_review.json')
    checkin_json: Path = Path('/root/autodl-tmp/mg-gat/data/raw/Yelp JSON/yelp_academic_dataset_checkin.json')
    processed_dir: Path = Path('/root/autodl-tmp/yelp/data/processed')
    artifacts_dir: Path = Path('/root/autodl-tmp/yelp/artifacts')


@dataclass
class SplitConfig:
    train_end_year: int = 2016
    valid_year: int = 2017
    test_year: int = 2018


@dataclass
class ReproductionConfig:
    region: str = 'PA'
    business_category_token: str = 'Restaurants'
    require_hours: bool = True
    require_attributes: bool = True
    require_checkins: bool = False
    require_friend_in_subset: bool = False
    use_year_window_only: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    assumptions: Dict[str, str] = field(default_factory=lambda: {
        'paper_scope': 'Only Stage 1 MG-GAT is implemented; all LLM-based explanation components are excluded.',
        'dataset_warning': 'The current Yelp snapshot appears materially larger than the paper\'s PA dataset, so exact paper subset construction is not directly observable.',
        'default_subset': 'The default conservative starting point is PA restaurants with hours and attributes present.',
    })


DEFAULT_CONFIG = ReproductionConfig()
