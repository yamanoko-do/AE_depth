from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"Unsupported PFM header {header!r}: {path}")
        width, height = map(int, handle.readline().decode("ascii").split())
        scale = float(handle.readline().decode("ascii").strip())
        dtype = "<f4" if scale < 0 else ">f4"
        channels = 3 if header == "PF" else 1
        data = np.fromfile(handle, dtype=dtype)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"PFM payload has {data.size} values, expected {expected}: {path}")
    shape = (height, width, channels) if channels == 3 else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float32)


def parse_middlebury_calibration(path: Path) -> tuple[float, float, float]:
    text = path.read_text(encoding="utf-8")
    camera = re.search(r"cam0=\[([^;]+)", text)
    baseline = re.search(r"baseline=([^\n]+)", text)
    doffs = re.search(r"doffs=([^\n]+)", text)
    if not camera or not baseline or not doffs:
        raise ValueError(f"Incomplete Middlebury-style calibration: {path}")
    fx = float(camera.group(1).split()[0])
    return fx, float(baseline.group(1)), float(doffs.group(1))


def parse_projection_file(path: Path) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        array = np.fromstring(values, sep=" ", dtype=np.float64)
        if array.size == 12:
            matrices[key] = array.reshape(3, 4)
    return matrices


def projection_fb(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    fx = float(left[0, 0])
    left_center_x = -float(left[0, 3]) / float(left[0, 0])
    right_center_x = -float(right[0, 3]) / float(right[0, 0])
    baseline_m = abs(right_center_x - left_center_x)
    return fx * baseline_m, fx, baseline_m


def nearest_fill(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        raise ValueError("Cannot fill a depth map with no valid pixels")
    if np.all(valid):
        return depth_m.astype(np.float32, copy=True)
    indices = distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    return depth_m[tuple(indices)].astype(np.float32)


def resize_depth_nearest(
    depth_m: np.ndarray, valid: np.ndarray, scale: int
) -> tuple[np.ndarray, np.ndarray]:
    if scale == 1:
        return depth_m, valid
    height, width = depth_m.shape
    size = (width // scale, height // scale)
    depth = cv2.resize(depth_m, size, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return depth.astype(np.float32), mask


def evenly_spaced(items: list, count: int) -> list:
    if count <= 0 or len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[int(index)] for index in indices]


def read_split_file(path: Path) -> list[list[str]]:
    """Read an OpenStereo-style whitespace separated dataset list."""
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) < 3:
            raise ValueError(
                f"Expected at least 3 columns in {path}:{line_number}, got {fields}"
            )
        rows.append(fields)
    return rows

