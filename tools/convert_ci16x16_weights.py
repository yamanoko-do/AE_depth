#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise TypeError(f"Cannot extract tensor from {type(value).__name__}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Cosmos CI16x16 JIT files to a native PyTorch state_dict."
    )
    parser.add_argument(
        "--jit_dir",
        type=Path,
        default=Path("/root/host_share/AE_coder/checkpoints/Cosmos-CI16x16"),
    )
    parser.add_argument(
        "--cosmos_tokenizer_root",
        type=Path,
        default=Path("/root/host_share/AE_coder/Cosmos-Tokenizer"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "weight/cosmos_ci16x16.pth",
    )
    parser.add_argument("--skip_verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder_path = args.jit_dir / "encoder.jit"
    decoder_path = args.jit_dir / "decoder.jit"
    for path in (encoder_path, decoder_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(args.cosmos_tokenizer_root.resolve()))
    from cosmos_tokenizer.networks.configs import continuous_image
    from cosmos_tokenizer.networks.continuous_image import ContinuousImageTokenizer

    encoder_jit = torch.jit.load(str(encoder_path), map_location="cpu").eval()
    decoder_jit = torch.jit.load(str(decoder_path), map_location="cpu").eval()
    native_model = ContinuousImageTokenizer(**continuous_image)
    native_keys = set(native_model.state_dict())
    merged = {
        key: value.detach().cpu()
        for key, value in {
            **encoder_jit.state_dict(),
            **decoder_jit.state_dict(),
        }.items()
        if key in native_keys
    }
    missing = sorted(native_keys - set(merged))
    if missing:
        raise RuntimeError(f"JIT files do not cover native model keys: {missing}")
    native_model.load_state_dict(merged, strict=True)

    metadata = {
        "format": "cosmos_ci16x16_state_dict",
        "format_version": 1,
        "model_name": "Cosmos-0.1-Tokenizer-CI16x16",
        "model_config": dict(continuous_image),
        "state_dict": merged,
        "source": {
            "encoder_jit": str(encoder_path.resolve()),
            "encoder_sha256": sha256(encoder_path),
            "decoder_jit": str(decoder_path.resolve()),
            "decoder_sha256": sha256(decoder_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(metadata, temporary)
    os.replace(temporary, args.output)

    verification = {}
    if not args.skip_verify:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for numerical verification")
        device = torch.device("cuda")
        dtype = torch.bfloat16
        encoder_jit = encoder_jit.to(device=device, dtype=dtype)
        decoder_jit = decoder_jit.to(device=device, dtype=dtype)
        native_model = native_model.to(device=device, dtype=dtype).eval()
        native_encoder = native_model.encoder_jit().eval()
        native_decoder = native_model.decoder_jit().eval()
        generator = torch.Generator(device=device).manual_seed(0)
        image = torch.randn(
            1, 3, 64, 64, generator=generator, device=device, dtype=dtype
        )
        with torch.inference_mode():
            reference_latent = first_tensor(encoder_jit(image))
            converted_latent = first_tensor(native_encoder(image))
            reference_image = decoder_jit(reference_latent)
            converted_image = native_decoder(converted_latent)
        latent_difference = (reference_latent.float() - converted_latent.float()).abs()
        image_difference = (reference_image.float() - converted_image.float()).abs()
        verification = {
            "latent_max_abs": float(latent_difference.max()),
            "latent_mean_abs": float(latent_difference.mean()),
            "image_max_abs": float(image_difference.max()),
            "image_mean_abs": float(image_difference.mean()),
        }
        if verification["latent_max_abs"] > 1e-5 or verification["image_max_abs"] > 1e-5:
            raise RuntimeError(f"Numerical verification failed: {verification}")

    summary = {
        "output": str(args.output.resolve()),
        "size_bytes": args.output.stat().st_size,
        "state_tensors": len(merged),
        "parameters": sum(value.numel() for value in merged.values()),
        "verification": verification,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
