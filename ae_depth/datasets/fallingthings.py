from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ae_depth.datasets.base import DepthSample, SampleSpec
from ae_depth.io import evenly_spaced, read_split_file


class FallingThingsAdapter:
    name = "FallingThings"

    def __init__(
        self,
        root: Path,
        max_depth_m: float,
        per_scene: int = 10,
        split_file: Path | None = None,
    ) -> None:
        self.root = root
        self.max_depth_m = max_depth_m
        self.per_scene = per_scene
        self.split_file = split_file

    def discover(self) -> list[SampleSpec]:
        by_scene: dict[Path, list[Path]] = defaultdict(list)
        if self.split_file is not None:
            paths = [self.root / row[2] for row in read_split_file(self.split_file)]
        else:
            paths = sorted((self.root / "fat/mixed").glob("*/*.left.depth.png"))
        for path in paths:
            by_scene[path.parent].append(path)
        specs = []
        for scene, paths in sorted(by_scene.items()):
            scene_id = str(scene.relative_to(self.root / "fat")).replace("/", "_")
            for path in evenly_spaced(paths, self.per_scene):
                specs.append(
                    SampleSpec(
                        dataset=self.name,
                        name=f"{scene_id}_{path.name.split('.')[0]}",
                        scene_type="synthetic_indoor",
                        primary_path=path,
                        metadata={"depth_scale_m": 1e-4, "scene": scene_id},
                    )
                )
        return specs

    def load(self, spec: SampleSpec) -> DepthSample:
        raw = np.asarray(Image.open(spec.primary_path))
        if raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError(f"Unexpected FallingThings depth format: {spec.primary_path}")
        scale = float(spec.metadata["depth_scale_m"])
        depth = raw.astype(np.float32) * scale
        valid = np.isfinite(depth) & (raw > 0)
        evaluation = valid & (depth <= self.max_depth_m)
        return DepthSample(
            self.name, spec.name, spec.scene_type, depth, valid, evaluation,
            dict(spec.metadata),
        )

