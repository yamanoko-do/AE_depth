from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from ae_depth.datasets.base import DepthSample, SampleSpec
from ae_depth.io import read_split_file, resize_depth_nearest


def read_booster_calibration(path: Path) -> tuple[float, float]:
    root = ElementTree.parse(path).getroot()
    projection_text = root.findtext("proj_matL/data")
    baseline_text = root.findtext("baselineLR")
    if projection_text is None or baseline_text is None:
        raise ValueError(f"Incomplete Booster calibration: {path}")
    projection = np.fromstring(projection_text, sep=" ").reshape(3, 4)
    return float(projection[0, 0]), float(baseline_text) / 1000.0


class BoosterAdapter:
    name = "Booster"

    def __init__(
        self,
        root: Path,
        max_depth_m: float,
        scale: int = 4,
        split_file: Path | None = None,
    ) -> None:
        self.root = root
        self.max_depth_m = max_depth_m
        self.scale = scale
        self.split_file = split_file

    def discover(self) -> list[SampleSpec]:
        if self.split_file is not None:
            paths = [self.root / row[2] for row in read_split_file(self.split_file)]
        else:
            paths = sorted((self.root / "train").glob("*/*/disp_00.npy"))
        specs = []
        for index, path in enumerate(paths):
            relative = path.parent.relative_to(self.root / "train")
            specs.append(
                SampleSpec(
                    dataset=self.name,
                    name=f"{index:06d}_{str(relative).replace('/', '_')}",
                    scene_type="indoor",
                    primary_path=path,
                    calibration_path=path.parent / "calib_00-02.xml",
                    metadata={"scale": self.scale},
                )
            )
        return specs

    def load(self, spec: SampleSpec) -> DepthSample:
        disparity = np.load(spec.primary_path).astype(np.float32)
        fx, baseline = read_booster_calibration(spec.calibration_path)
        valid = np.isfinite(disparity) & (disparity > 0)
        depth = np.zeros_like(disparity, dtype=np.float32)
        depth[valid] = fx * baseline / disparity[valid]
        depth, valid = resize_depth_nearest(depth, valid, self.scale)
        evaluation = valid & (depth <= self.max_depth_m)
        return DepthSample(
            self.name, spec.name, spec.scene_type, depth, valid, evaluation,
            {
                "fx_px_original": fx,
                "baseline_m": baseline,
                "evaluation_scale": self.scale,
            },
        )

