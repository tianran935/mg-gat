from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator


def json_lines(path: Path) -> Iterator[Dict]:
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
