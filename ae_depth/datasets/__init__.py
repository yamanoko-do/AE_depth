from __future__ import annotations

from pathlib import Path

from ae_depth.datasets.booster import BoosterAdapter
from ae_depth.datasets.eth3d import ETH3DAdapter
from ae_depth.datasets.fallingthings import FallingThingsAdapter
from ae_depth.datasets.kitti import KITTI12Adapter, KITTI15Adapter
from ae_depth.datasets.middlebury import MiddleburyAdapter


def build_adapters(config: dict, max_depth_m: float):
    if "DATA_CONFIG" in config:
        entries = config["DATA_CONFIG"]["DATA_INFOS"]
        datasets = {}
        enabled = []
        for entry in entries:
            name = entry["DATASET"]
            enabled.append(name)
            datasets[name] = {
                **entry,
                "ROOT": entry["DATA_PATH"],
                "SPLIT_FILE": entry.get("DATA_SPLIT", {}).get("EVALUATING"),
            }
    else:
        datasets = config["DATASETS"]
        enabled = datasets["ENABLED"]

    def split(cfg):
        value = cfg.get("SPLIT_FILE")
        return Path(value) if value else None

    builders = {
        "ETH3D": lambda cfg: ETH3DAdapter(
            Path(cfg["ROOT"]), max_depth_m, split(cfg)
        ),
        "KITTI12": lambda cfg: KITTI12Adapter(
            Path(cfg["ROOT"]), max_depth_m, split(cfg), bool(cfg.get("USE_NOC", True))
        ),
        "KITTI15": lambda cfg: KITTI15Adapter(
            Path(cfg["ROOT"]), max_depth_m, split(cfg), bool(cfg.get("USE_NOC", True))
        ),
        "Middlebury": lambda cfg: MiddleburyAdapter(
            Path(cfg["ROOT"]),
            max_depth_m,
            cfg.get("RESOLUTION", cfg.get("MIDDLEBURY_RESO", "H")),
            split(cfg),
        ),
        "Booster": lambda cfg: BoosterAdapter(
            Path(cfg["ROOT"]),
            max_depth_m,
            int(
                cfg.get(
                    "SCALE",
                    {"F": 1, "H": 2, "Q": 4}.get(cfg.get("BOOSTER_RESO", "Q"), 4),
                )
            ),
            split(cfg),
        ),
        "FallingThings": lambda cfg: FallingThingsAdapter(
            Path(cfg["ROOT"]),
            max_depth_m,
            int(cfg.get("PER_SCENE", cfg.get("MAX_PER_SCENE", 10))),
            split(cfg),
        ),
    }
    return [builders[name](datasets[name]) for name in enabled]


__all__ = ["build_adapters"]

