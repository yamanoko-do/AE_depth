from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CodecResult:
    decoded_rgb: np.ndarray
    latent_shape: list[int]
    latent_elements: int
    latent_bytes: int
    encode_ms: float
    decode_ms: float
    padded_shape_hw: list[int]


@dataclass
class BatchCodecResult:
    decoded_rgb: np.ndarray
    latent_shape_per_image: list[int]
    latent_elements_per_image: int
    latent_bytes_per_image: int
    encode_batch_ms: float
    decode_batch_ms: float
    padded_shape_hw: list[int]
    batch_size: int


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise TypeError(f"Cannot extract a tensor from {type(value).__name__}")


def _pad_rgb(image: np.ndarray, multiple: int = 16):
    height, width = image.shape[:2]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    padded = np.pad(
        image, ((top, bottom), (left, right), (0, 0)), mode="constant"
    )
    return padded, (top, left, top + height, left + width)


class CosmosCI16x16Codec:
    def __init__(
        self,
        weights: Path,
        cosmos_tokenizer_root: Path,
        dtype: str = "bfloat16",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Cosmos CI16x16 benchmark requires CUDA")
        if not weights.is_file():
            raise FileNotFoundError(f"PyTorch weights not found: {weights}")
        package_root = str(cosmos_tokenizer_root.resolve())
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        from cosmos_tokenizer.networks.continuous_image import (
            ContinuousImageTokenizer,
        )

        self.dtype = getattr(torch, dtype)
        payload = torch.load(weights, map_location="cpu", weights_only=False)
        if payload.get("format") != "cosmos_ci16x16_state_dict":
            raise ValueError(f"Unsupported Cosmos weight format: {weights}")
        model = ContinuousImageTokenizer(**payload["model_config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        model = model.eval().to(device="cuda", dtype=self.dtype)
        self.model = model
        self.encoder = model.encoder_jit().eval()
        self.decoder = model.decoder_jit().eval()

    @staticmethod
    def _time(fn, repeats: int):
        values = []
        result = None
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = fn()
            torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000.0)
        return result, statistics.median(values)

    def reconstruct(
        self, rgb_01: np.ndarray, repeats: int = 1, warmup: int = 0
    ) -> CodecResult:
        if rgb_01.ndim != 3 or rgb_01.shape[2] != 3:
            raise ValueError("Codec input must be HxWx3")
        result = self.reconstruct_batch(
            rgb_01[None], repeats=repeats, warmup=warmup
        )
        return CodecResult(
            decoded_rgb=result.decoded_rgb[0],
            latent_shape=result.latent_shape_per_image,
            latent_elements=result.latent_elements_per_image,
            latent_bytes=result.latent_bytes_per_image,
            encode_ms=result.encode_batch_ms,
            decode_ms=result.decode_batch_ms,
            padded_shape_hw=result.padded_shape_hw,
        )

    def reconstruct_batch(
        self, rgb_01: np.ndarray, repeats: int = 1, warmup: int = 0
    ) -> BatchCodecResult:
        if rgb_01.ndim != 4 or rgb_01.shape[-1] != 3:
            raise ValueError("Batch codec input must be BxHxWx3")
        if rgb_01.shape[0] < 1:
            raise ValueError("Batch codec input cannot be empty")
        first_padded, crop = _pad_rgb(rgb_01[0].astype(np.float32, copy=False))
        padded_images = [first_padded]
        for image in rgb_01[1:]:
            padded, current_crop = _pad_rgb(image.astype(np.float32, copy=False))
            if padded.shape != first_padded.shape or current_crop != crop:
                raise ValueError("All images in a codec batch must have the same shape")
            padded_images.append(padded)
        padded_batch = np.stack(padded_images)
        tensor = torch.from_numpy(padded_batch).permute(0, 3, 1, 2)
        tensor = tensor.cuda().to(self.dtype).mul(2.0).sub(1.0)
        with torch.inference_mode():
            for _ in range(warmup):
                latent = _first_tensor(self.encoder(tensor))
                _ = self.decoder(latent)
            latent, encode_ms = self._time(
                lambda: _first_tensor(self.encoder(tensor)), repeats
            )
            decoded, decode_ms = self._time(
                lambda: _first_tensor(self.decoder(latent)), repeats
            )
        decoded = decoded.float().add(1.0).div(2.0).clamp(0.0, 1.0)
        top, left, bottom, right = crop
        decoded = decoded[:, :, top:bottom, left:right]
        decoded_rgb = decoded.permute(0, 2, 3, 1).cpu().numpy()
        element_size = torch.tensor([], dtype=latent.dtype).element_size()
        elements_per_image = int(latent[0].numel())
        return BatchCodecResult(
            decoded_rgb=decoded_rgb,
            latent_shape_per_image=[1, *list(latent.shape[1:])],
            latent_elements_per_image=elements_per_image,
            latent_bytes_per_image=elements_per_image * element_size,
            encode_batch_ms=encode_ms,
            decode_batch_ms=decode_ms,
            padded_shape_hw=[int(first_padded.shape[0]), int(first_padded.shape[1])],
            batch_size=int(rgb_01.shape[0]),
        )

