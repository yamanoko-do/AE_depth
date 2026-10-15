from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ae_depth.datasets.base import DepthSample, SampleSpec
from ae_depth.io import parse_middlebury_calibration, read_pfm, read_split_file


class MiddleburyAdapter:
    name = "Middlebury"

    def __init__(
        self,
        root: Path,
        max_depth_m: float,
        resolution: str = "H",
        split_file: Path | None = None,
    ) -> None:
        self.root = root
        self.max_depth_m = max_depth_m
        self.resolution = resolution.upper()
        self.split_file = split_file

    def discover(self) -> list[SampleSpec]:
        if self.split_file is not None:
            paths = [self.root / row[2] for row in read_split_file(self.split_file)]
        else:
            base = self.root / "MiddEval3" / f"training{self.resolution}"
            paths = sorted(base.glob("*/disp0GT.pfm"))
        return [
            SampleSpec(
                dataset=self.name,
                name=path.parent.name,
                scene_type="indoor",
                primary_path=path,
                calibration_path=path.parent / "calib.txt",
                mask_path=path.parent / "mask0nocc.png",
                metadata={"resolution": self.resolution},
            )
            for path in paths
        ]

    def load(self, spec: SampleSpec) -> DepthSample:
        disparity = read_pfm(spec.primary_path)
        fx, baseline_mm, doffs = parse_middlebury_calibration(spec.calibration_path)
        denominator = disparity + doffs
        source_valid = np.isfinite(disparity) & (disparity > 0) & (denominator > 0)
        depth = np.zeros_like(disparity, dtype=np.float32)
        depth[source_valid] = fx * (baseline_mm / 1000.0) / denominator[source_valid]
        mask = np.asarray(Image.open(spec.mask_path)) == 255
        evaluation = source_valid & mask & (depth <= self.max_depth_m)
        return DepthSample(
            self.name, spec.name, spec.scene_type, depth, source_valid, evaluation,
            {
                "fx_px": fx,
                "baseline_m": baseline_mm / 1000.0,
                "doffs_px": doffs,
                **spec.metadata,
            },
        )

