from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class DataConfig:
    raw_dir: Path = Path("/root/autodl-tmp/mg-gat/data/raw/Yelp JSON")
    business_json: Path = Path("/root/autodl-tmp/mg-gat/data/raw/Yelp JSON/yelp_academic_dataset_business.json")
    user_json: Path = Path("/root/autodl-tmp/mg-gat/data/raw/Yelp JSON/yelp_academic_dataset_user.json")
    review_json: Path = Path("/root/autodl-tmp/mg-gat/data/raw/Yelp JSON/yelp_academic_dataset_review.json")
    checkin_json: Path = Path("/root/autodl-tmp/mg-gat/data/raw/Yelp JSON/yelp_academic_dataset_checkin.json")
    processed_dir: Path = Path("/root/autodl-tmp/yelp/data/processed")
    artifacts_dir: Path = Path("/root/autodl-tmp/yelp/artifacts")


@dataclass
class SplitConfig:
    train_end_year: int = 2016
    valid_year: int = 2017
    test_year: int = 2018


@dataclass
class GraphConfig:
    business_k: int = 10
    covisit_k: int = 10
    covisit_min_shared: int = 2
    max_user_neighbors: int = 50


@dataclass
class TrainConfig:
    device: str = "cuda"
    hidden_dim: int = 64
    latent_dim: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    theta1: float = 1e-4
    theta2: float = 1e-4
    epochs: int = 70
    patience: int = 12
    seed: int = 42
    interpretable: bool = True


@dataclass
class ReproductionConfig:
    region: str = "PA"
    business_category_token: str = "Restaurants"
    require_hours: bool = True
    require_attributes: bool = True
    require_checkins: bool = False
    require_friend_in_subset: bool = True
    use_largest_friend_component: bool = True
    use_year_window_only: bool = True
    min_category_freq: int = 20
    min_attribute_freq: int = 20
    implicit_dim: int = 32
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    assumptions: Dict[str, str] = field(default_factory=lambda: {
        "paper_scope": "Only Stage 1 MG-GAT is implemented; all LLM-based explanation components are excluded.",
        "dataset_warning": "The current Yelp snapshot appears materially larger than the paper's PA dataset, so exact paper subset construction is not directly observable.",
        "default_subset": "PA restaurants with hours and attributes, restricted to users connected through the friendship graph giant component.",
        "best_known_setting": "The current default training schedule reflects the strongest interpretable configuration found so far on the available Yelp snapshot.",
    })


DEFAULT_CONFIG = ReproductionConfig()
