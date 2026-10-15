from __future__ import annotations

import math

import numpy as np


class LinearMinMaxRGB:
    name = "linear_minmax_rgb"
    display_name = "线性Min-Max复制RGB"

    def encode(self, depth_m: np.ndarray):
        minimum = float(depth_m.min())
        maximum = float(depth_m.max())
        span = max(maximum - minimum, np.finfo(np.float32).eps)
        normalized = np.clip((depth_m - minimum) / span, 0.0, 1.0)
        return np.repeat(normalized[..., None], 3, axis=2), {
            "minimum_m": minimum,
            "maximum_m": maximum,
        }

    def decode(self, decoded_rgb: np.ndarray, metadata: dict):
        normalized = decoded_rgb.mean(axis=2)
        return (
            normalized * (metadata["maximum_m"] - metadata["minimum_m"])
            + metadata["minimum_m"]
        ).astype(np.float32)


class LogMinMaxRGB:
    name = "log_minmax_rgb"
    display_name = "Log-Min-Max复制RGB"

    def encode(self, depth_m: np.ndarray):
        log_depth = np.log1p(depth_m)
        minimum = float(log_depth.min())
        maximum = float(log_depth.max())
        span = max(maximum - minimum, np.finfo(np.float32).eps)
        normalized = np.clip((log_depth - minimum) / span, 0.0, 1.0)
        return np.repeat(normalized[..., None], 3, axis=2), {
            "minimum_log_depth": minimum,
            "maximum_log_depth": maximum,
        }

    def decode(self, decoded_rgb: np.ndarray, metadata: dict):
        normalized = decoded_rgb.mean(axis=2)
        log_depth = normalized * (
            metadata["maximum_log_depth"] - metadata["minimum_log_depth"]
        ) + metadata["minimum_log_depth"]
        return np.expm1(log_depth).astype(np.float32)


class DeFMThreeChannel:
    name = "defm_three_channel"
    display_name = "DeFM三通道"

    def __init__(self, max_depth_c1: float = 100.0, max_depth_c2: float = 9.0):
        self.max_depth_c1 = max_depth_c1
        self.max_depth_c2 = max_depth_c2

    def encode(self, depth_m: np.ndarray):
        metric = np.clip(
            np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            self.max_depth_c1,
        )
        log_depth = np.log1p(metric)
        minimum = float(log_depth.min())
        maximum = float(log_depth.max())
        c1 = log_depth / math.log1p(self.max_depth_c1)
        c2 = np.clip(log_depth / math.log1p(self.max_depth_c2), 0.0, 1.0)
        c3 = (
            (log_depth - minimum) / (maximum - minimum)
            if maximum > minimum
            else np.zeros_like(log_depth)
        )
        return np.stack((c1, c2, c3), axis=2).astype(np.float32), {
            "max_depth_c1_m": self.max_depth_c1,
            "max_depth_c2_m": self.max_depth_c2,
            "minimum_log_depth": minimum,
            "maximum_log_depth": maximum,
        }

    def decode(self, decoded_rgb: np.ndarray, metadata: dict):
        rgb = np.clip(decoded_rgb, 0.0, 1.0)
        log_range = metadata["maximum_log_depth"] - metadata["minimum_log_depth"]
        log_c1 = rgb[..., 0] * math.log1p(metadata["max_depth_c1_m"])
        log_c2 = rgb[..., 1] * math.log1p(metadata["max_depth_c2_m"])
        log_c3 = rgb[..., 2] * log_range + metadata["minimum_log_depth"]
        candidates = np.stack((log_c1, log_c2, log_c3), axis=2)
        slopes = np.array(
            [
                1.0 / math.log1p(metadata["max_depth_c1_m"]),
                1.0 / math.log1p(metadata["max_depth_c2_m"]),
                0.0 if log_range <= 0.0 else 1.0 / log_range,
            ],
            dtype=np.float32,
        )
        weights = np.broadcast_to(slopes**2, candidates.shape).copy()
        weights[..., 1] *= log_c3 < math.log1p(metadata["max_depth_c2_m"])
        denominator = np.maximum(
            np.sum(weights, axis=2), np.finfo(np.float32).eps
        )
        fused_log = np.sum(candidates * weights, axis=2) / denominator
        return np.expm1(fused_log).astype(np.float32)


def build_encodings(names: list[str]):
    available = {
        LinearMinMaxRGB.name: LinearMinMaxRGB,
        LogMinMaxRGB.name: LogMinMaxRGB,
        DeFMThreeChannel.name: DeFMThreeChannel,
    }
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown depth encodings: {unknown}")
    return [available[name]() for name in names]

