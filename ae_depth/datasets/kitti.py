from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ae_depth.datasets.base import DepthSample, SampleSpec
from ae_depth.io import parse_projection_file, projection_fb, read_split_file


class KITTI12Adapter:
    name = "KITTI12"

    def __init__(
        self,
        root: Path,
        max_depth_m: float,
        split_file: Path | None = None,
        use_noc: bool = True,
    ) -> None:
        self.root = root
        self.max_depth_m = max_depth_m
        self.split_file = split_file
        self.use_noc = use_noc

    def discover(self) -> list[SampleSpec]:
        if self.split_file is not None:
            relative_paths = [row[2] for row in read_split_file(self.split_file)]
            if self.use_noc:
                relative_paths = [p.replace("disp_occ", "disp_noc") for p in relative_paths]
            paths = [self.root / p for p in relative_paths]
        else:
            directory = "disp_noc" if self.use_noc else "disp_occ"
            paths = sorted((self.root / "training" / directory).glob("*_10.png"))
        specs = []
        for path in paths:
            stem = path.name.split("_")[0]
            specs.append(
                SampleSpec(
                    dataset=self.name,
                    name=stem,
                    scene_type="outdoor",
                    primary_path=path,
                    calibration_path=self.root / "training/calib" / f"{stem}.txt",
                )
            )
        return specs

    def load(self, spec: SampleSpec) -> DepthSample:
        disparity = np.asarray(Image.open(spec.primary_path), dtype=np.float32) / 256.0
        matrices = parse_projection_file(spec.calibration_path)
        fb, fx, baseline = projection_fb(matrices["P0"], matrices["P1"])
        valid = np.isfinite(disparity) & (disparity > 0)
        depth = np.zeros_like(disparity, dtype=np.float32)
        depth[valid] = fb / disparity[valid]
        evaluation = valid & (depth <= self.max_depth_m)
        return DepthSample(
            self.name, spec.name, spec.scene_type, depth, valid, evaluation,
            {"fx_px": fx, "baseline_m": baseline, "projection_pair": "P0/P1"},
        )


class KITTI15Adapter:
    name = "KITTI15"

    def __init__(
        self,
        root: Path,
        max_depth_m: float,
        split_file: Path | None = None,
        use_noc: bool = True,
    ) -> None:
        self.root = root
        self.max_depth_m = max_depth_m
        self.split_file = split_file
        self.use_noc = use_noc

    def discover(self) -> list[SampleSpec]:
        if self.split_file is not None:
            relative_paths = [row[2] for row in read_split_file(self.split_file)]
            if self.use_noc:
                relative_paths = [p.replace("disp_occ_0", "disp_noc_0") for p in relative_paths]
            paths = [self.root / p for p in relative_paths]
        else:
            directory = "disp_noc_0" if self.use_noc else "disp_occ_0"
            paths = sorted((self.root / "training" / directory).glob("*_10.png"))
        specs = []
        for path in paths:
            stem = path.name.split("_")[0]
            specs.append(
                SampleSpec(
                    dataset=self.name,
                    name=stem,
                    scene_type="outdoor",
                    primary_path=path,
                    calibration_path=(
                        self.root / "training/calib_cam_to_cam" / f"{stem}.txt"
                    ),
                )
            )
        return specs

    def load(self, spec: SampleSpec) -> DepthSample:
        disparity = np.asarray(Image.open(spec.primary_path), dtype=np.float32) / 256.0
        matrices = parse_projection_file(spec.calibration_path)
        fb, fx, baseline = projection_fb(
            matrices["P_rect_02"], matrices["P_rect_03"]
        )
        valid = np.isfinite(disparity) & (disparity > 0)
        depth = np.zeros_like(disparity, dtype=np.float32)
        depth[valid] = fb / disparity[valid]
        evaluation = valid & (depth <= self.max_depth_m)
        return DepthSample(
            self.name, spec.name, spec.scene_type, depth, valid, evaluation,
            {
                "fx_px": fx,
                "baseline_m": baseline,
                "projection_pair": "P_rect_02/P_rect_03",
            },
        )

