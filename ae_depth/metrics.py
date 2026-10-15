from __future__ import annotations

import math
import statistics
from collections import defaultdict

import numpy as np


def image_metrics(reference: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray):
    ref = reference[mask].astype(np.float64)
    rec = reconstruction[mask].astype(np.float64)
    if ref.size == 0:
        raise ValueError("Metric mask contains no valid pixels")
    error = rec - ref
    absolute = np.abs(error)
    return {
        "valid_pixels": int(ref.size),
        "sum_abs_error_m": float(absolute.sum()),
        "sum_sq_error_m2": float(np.square(error).sum()),
        "sum_abs_rel": float((absolute / np.maximum(ref, 1e-8)).sum()),
        "sum_signed_error_m": float(error.sum()),
        "epe_m_per_pixel": float(absolute.mean()),
        "rmse_m": float(np.sqrt(np.square(error).mean())),
        "abs_rel": float((absolute / np.maximum(ref, 1e-8)).mean()),
        "bias_m": float(error.mean()),
    }


class AggregateMetrics:
    def __init__(self):
        self.data = defaultdict(
            lambda: {
                "pixels": 0,
                "sum_abs": 0.0,
                "sum_sq": 0.0,
                "sum_abs_rel": 0.0,
                "sum_signed": 0.0,
                "image_epe": [],
                "encode_ms": [],
                "decode_ms": [],
                "latent_bytes": [],
                "raw_bytes": [],
            }
        )

    def update(self, key: tuple, row: dict):
        target = self.data[key]
        target["pixels"] += row["valid_pixels"]
        target["sum_abs"] += row["sum_abs_error_m"]
        target["sum_sq"] += row["sum_sq_error_m2"]
        target["sum_abs_rel"] += row["sum_abs_rel"]
        target["sum_signed"] += row["sum_signed_error_m"]
        target["image_epe"].append(row["epe_m_per_pixel"])
        target["encode_ms"].append(row["encode_ms"])
        target["decode_ms"].append(row["decode_ms"])
        target["latent_bytes"].append(row["latent_bytes"])
        target["raw_bytes"].append(row["raw_depth_uint16_bytes"])

    def summaries(self):
        output = {}
        for key, value in self.data.items():
            pixels = value["pixels"]
            output[key] = {
                "images": len(value["image_epe"]),
                "valid_pixels": pixels,
                "epe_m_per_pixel": value["sum_abs"] / pixels,
                "rmse_m": math.sqrt(value["sum_sq"] / pixels),
                "abs_rel": value["sum_abs_rel"] / pixels,
                "bias_m": value["sum_signed"] / pixels,
                "macro_image_epe_m": statistics.mean(value["image_epe"]),
                "encode_median_ms": statistics.median(value["encode_ms"]),
                "decode_median_ms": statistics.median(value["decode_ms"]),
                "total_median_ms": statistics.median(
                    [a + b for a, b in zip(value["encode_ms"], value["decode_ms"])]
                ),
                "compression_vs_uint16": sum(value["raw_bytes"])
                / sum(value["latent_bytes"]),
            }
        return output

