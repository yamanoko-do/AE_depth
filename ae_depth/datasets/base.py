from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SampleSpec:
    dataset: str
    name: str
    scene_type: str
    primary_path: Path
    calibration_path: Path | None = None
    mask_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthSample:
    dataset: str
    name: str
    scene_type: str
    depth_m: np.ndarray
    source_valid: np.ndarray
    evaluation_mask: np.ndarray
    metadata: dict[str, Any]

