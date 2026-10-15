from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ae_depth.codecs import CosmosCI16x16Codec, build_encodings
from ae_depth.datasets import build_adapters
from ae_depth.io import evenly_spaced, nearest_fill
from ae_depth.metrics import AggregateMetrics, image_metrics


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _colorize(values: np.ndarray, maximum: float) -> np.ndarray:
    scaled = np.clip(values / max(maximum, 1e-8), 0.0, 1.0)
    image = np.round(scaled * 255.0).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def _label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    bar_height = 36
    output = cv2.copyMakeBorder(
        panel, bar_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
    )
    cv2.putText(
        output,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def save_example(
    output_dir: Path,
    dataset: str,
    sample_name: str,
    reference: np.ndarray,
    mask: np.ndarray,
    reconstructions: dict[str, np.ndarray],
) -> None:
    example_dir = output_dir / "examples" / dataset / sample_name
    example_dir.mkdir(parents=True, exist_ok=True)
    np.save(example_dir / "reference_depth_m.npy", reference.astype(np.float32))
    display_limit = float(np.percentile(reference[mask], 99.0))
    panels = [_label_panel(_colorize(reference, display_limit), "GT depth")]
    for method, reconstruction in reconstructions.items():
        np.save(
            example_dir / f"{method}_reconstructed_depth_m.npy",
            reconstruction.astype(np.float32),
        )
        absolute_error = np.abs(reconstruction - reference)
        error_limit = max(float(np.percentile(absolute_error[mask], 99.0)), 1e-4)
        panels.append(
            _label_panel(
                _colorize(reconstruction, display_limit), f"{method}: reconstructed"
            )
        )
        panels.append(
            _label_panel(
                _colorize(absolute_error, error_limit),
                f"{method}: abs error, p99={error_limit:.4f}m",
            )
        )
    target_width = min(1280, max(panel.shape[1] for panel in panels))
    resized = []
    for panel in panels:
        if panel.shape[1] != target_width:
            height = round(panel.shape[0] * target_width / panel.shape[1])
            panel = cv2.resize(panel, (target_width, height), interpolation=cv2.INTER_AREA)
        resized.append(panel)
    cv2.imwrite(str(example_dir / "comparison.png"), np.vstack(resized))


def _summary_rows(aggregates: AggregateMetrics, scope: str) -> list[dict]:
    rows = []
    for key, values in aggregates.summaries().items():
        dataset, method = key
        rows.append(
            {
                "scope": scope,
                "dataset": dataset,
                "method": method,
                "depth_bin_m": "all",
                **values,
            }
        )
    return rows


def _bin_label(lower: float, upper: float) -> str:
    return f"[{lower:g},{upper:g})"


def _build_report(
    config: dict,
    dataset_rows: list[dict],
    bin_rows: list[dict],
    failures: list[dict],
) -> str:
    methods = config["BENCHMARK"]["METHODS"]
    lines = [
        "# Cosmos CI16x16 深度图压缩 Benchmark",
        "",
        "## 测试设置",
        "",
        f"- 最大评估深度：`{config['BENCHMARK']['MAX_DEPTH_M']:.1f} m`。",
        f"- 配置 batch size：`{config['BENCHMARK'].get('BATCH_SIZE', 1)}`；不同分辨率分别组 batch。",
        "- 编解码耗时为 batch CUDA 同步计时除以实际 batch size 后的单图摊销耗时。",
        "- 样本集合来自 OpenStereo 的数据列表；官方测试集无公开 GT 时，使用其带 GT 的训练/评估列表。",
        "- KITTI 使用非遮挡视差，ETH3D/Middlebury 只统计 `mask0nocc == 255`。",
        "- 稀疏深度先做最近邻填充后输入编码器，指标只在原始有效 GT 像素上计算。",
        "- EPE 即每个有效像素的平均绝对深度误差，单位为米/像素，越低越好。",
        f"- 编码方法：`{'`, `'.join(methods)}`。",
        "",
        "## 按数据集汇总",
        "",
        "| 数据集 | 编码方式 | 图像数 | EPE (m/pixel) | RMSE (m) | AbsRel | Bias (m) | 编码 (ms/image) | 解码 (ms/image) | latent/uint16 压缩率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        dataset_rows,
        key=lambda row: (row["dataset"] != "ALL", row["dataset"], row["epe_m_per_pixel"]),
    )
    for row in ordered:
        lines.append(
            "| {dataset} | {method} | {images} | {epe_m_per_pixel:.6f} | "
            "{rmse_m:.6f} | {abs_rel:.6f} | {bias_m:.6f} | "
            "{encode_median_ms:.3f} | {decode_median_ms:.3f} | "
            "{compression_vs_uint16:.2f}× |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 按深度区间汇总",
            "",
            "| 数据集 | 深度区间 (m) | 编码方式 | 有效像素 | EPE (m/pixel) | RMSE (m) | AbsRel |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        bin_rows, key=lambda item: (item["dataset"], item["depth_bin_m"], item["epe_m_per_pixel"])
    ):
        lines.append(
            "| {dataset} | {depth_bin_m} | {method} | {valid_pixels} | "
            "{epe_m_per_pixel:.6f} | {rmse_m:.6f} | {abs_rel:.6f} |".format(**row)
        )
    if failures:
        lines.extend(["", "## 失败样本", ""])
        for failure in failures:
            lines.append(
                f"- `{failure['dataset']}/{failure['sample']}`：{failure['error']}"
            )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `summary.csv`：数据集和深度区间聚合指标。",
            "- `per_image_results.csv`：逐图指标、耗时和 latent 大小。",
            "- `results.json`：完整机器可读结果与配置。",
            "- `examples/`：GT、重建深度、误差热力图和 `.npy` 深度。",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(
    config: dict,
    limit_per_dataset: int = 0,
    batch_size_override: int | None = None,
    logger=None,
) -> dict:
    benchmark_cfg = config["BENCHMARK"]
    output_dir = Path(benchmark_cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(
        batch_size_override
        if batch_size_override is not None
        else benchmark_cfg.get("BATCH_SIZE", 1)
    )
    if batch_size < 1:
        raise ValueError("BATCH_SIZE must be at least 1")
    log_interval = max(1, int(benchmark_cfg.get("LOG_INTERVAL", 10)))
    max_depth_m = float(benchmark_cfg["MAX_DEPTH_M"])
    repeats = int(benchmark_cfg.get("REPEATS", 1))
    warmup = int(benchmark_cfg.get("WARMUP", 0))
    save_count = int(benchmark_cfg.get("SAVE_EXAMPLES_PER_DATASET", 1))
    depth_bins = [tuple(map(float, values)) for values in benchmark_cfg["DEPTH_BINS_M"]]
    encodings = build_encodings(benchmark_cfg["METHODS"])
    codec = CosmosCI16x16Codec(
        Path(config["MODEL"]["WEIGHTS"]),
        Path(config["MODEL"]["COSMOS_TOKENIZER_ROOT"]),
        config["MODEL"].get("DTYPE", "bfloat16"),
    )
    adapters = build_adapters(config, max_depth_m)
    aggregates = AggregateMetrics()
    bin_aggregates = AggregateMetrics()
    per_image_rows: list[dict] = []
    failures: list[dict] = []
    warmed_shapes: set[tuple[int, int, int]] = set()

    def log(message: str) -> None:
        if logger is None:
            print(message, flush=True)
        else:
            logger.info(message)

    log(f"Model weights: {config['MODEL']['WEIGHTS']}")
    log(f"Output directory: {output_dir}")
    log(f"Batch size: {batch_size}")
    log(f"Depth encodings: {', '.join(benchmark_cfg['METHODS'])}")

    def record_failure(dataset: str, sample_name: str, error: Exception) -> None:
        failure = {
            "dataset": dataset,
            "sample": sample_name,
            "error": f"{type(error).__name__}: {error}",
        }
        failures.append(failure)
        log(f"FAILED [{dataset}] {sample_name}: {failure['error']}")

    for adapter in adapters:
        specs = adapter.discover()
        if limit_per_dataset > 0:
            specs = evenly_spaced(specs, limit_per_dataset)
        log(f">>> Start evaluating [{adapter.name}]")
        log(f"Total samples for [{adapter.name}]: {len(specs)}")
        examples_saved = 0
        processed = 0
        batch_index = 0
        shape_buckets: dict[tuple[int, int], list] = defaultdict(list)

        def evaluate_batch(samples: list) -> None:
            nonlocal examples_saved, processed, batch_index
            if not samples:
                return
            current_batch_size = len(samples)
            shape = tuple(samples[0].depth_m.shape)
            signature = (current_batch_size, *shape)
            codec_depths = []
            for sample in samples:
                codec_depth = nearest_fill(sample.depth_m, sample.source_valid)
                codec_depths.append(np.clip(codec_depth, 0.0, max_depth_m))
            reconstructions = [dict() for _ in samples]
            method_batch_times = []
            for encoding in encodings:
                encoded = [encoding.encode(depth) for depth in codec_depths]
                rgb_batch = np.stack([item[0] for item in encoded])
                result = codec.reconstruct_batch(
                    rgb_batch,
                    repeats=repeats,
                    warmup=warmup if signature not in warmed_shapes else 0,
                )
                warmed_shapes.add(signature)
                encode_per_image_ms = result.encode_batch_ms / current_batch_size
                decode_per_image_ms = result.decode_batch_ms / current_batch_size
                method_batch_times.append(result.encode_batch_ms + result.decode_batch_ms)
                for item_index, sample in enumerate(samples):
                    reconstruction = encoding.decode(
                        result.decoded_rgb[item_index], encoded[item_index][1]
                    )
                    reconstruction = np.clip(reconstruction, 0.0, max_depth_m)
                    reconstructions[item_index][encoding.name] = reconstruction
                    metrics = image_metrics(
                        sample.depth_m, reconstruction, sample.evaluation_mask
                    )
                    raw_bytes = int(
                        sample.depth_m.size * np.dtype(np.uint16).itemsize
                    )
                    row = {
                        "dataset": sample.dataset,
                        "sample": sample.name,
                        "scene_type": sample.scene_type,
                        "method": encoding.name,
                        "height": int(sample.depth_m.shape[0]),
                        "width": int(sample.depth_m.shape[1]),
                        "batch_size": current_batch_size,
                        "latent_shape": json.dumps(result.latent_shape_per_image),
                        "latent_elements": result.latent_elements_per_image,
                        "latent_bytes": result.latent_bytes_per_image,
                        "raw_depth_uint16_bytes": raw_bytes,
                        "compression_vs_uint16": (
                            raw_bytes / result.latent_bytes_per_image
                        ),
                        "encode_batch_ms": result.encode_batch_ms,
                        "decode_batch_ms": result.decode_batch_ms,
                        "encode_ms": encode_per_image_ms,
                        "decode_ms": decode_per_image_ms,
                        "total_ms": encode_per_image_ms + decode_per_image_ms,
                        "padded_shape_hw": json.dumps(result.padded_shape_hw),
                        **metrics,
                    }
                    per_image_rows.append(row)
                    aggregates.update((sample.dataset, encoding.name), row)
                    aggregates.update(("ALL", encoding.name), row)
                    for lower, upper in depth_bins:
                        bin_mask = (
                            sample.evaluation_mask
                            & (sample.depth_m >= lower)
                            & (sample.depth_m < upper)
                        )
                        if not np.any(bin_mask):
                            continue
                        bin_row = {
                            **row,
                            **image_metrics(sample.depth_m, reconstruction, bin_mask),
                        }
                        label = _bin_label(lower, upper)
                        bin_aggregates.update(
                            (sample.dataset, f"{encoding.name}|{label}"), bin_row
                        )
            for sample, sample_reconstructions in zip(samples, reconstructions):
                if examples_saved >= save_count:
                    break
                save_example(
                    output_dir,
                    sample.dataset,
                    sample.name,
                    sample.depth_m,
                    sample.evaluation_mask,
                    sample_reconstructions,
                )
                examples_saved += 1
            processed += current_batch_size
            batch_index += 1
            if batch_index % log_interval == 0 or processed == len(specs):
                mean_method_time = sum(method_batch_times) / len(method_batch_times)
                log(
                    f"Evaluating {adapter.name} [{processed}/{len(specs)}] "
                    f"Iter:{batch_index:>4d} BatchSize:{current_batch_size} "
                    f"InferTime:{mean_method_time:.2f}ms/batch "
                    f"({mean_method_time/current_batch_size:.2f}ms/image)"
                )

        def evaluate_with_oom_fallback(samples: list) -> None:
            try:
                evaluate_batch(samples)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                if len(samples) == 1:
                    record_failure(adapter.name, samples[0].name, error)
                    return
                midpoint = len(samples) // 2
                log(
                    f"CUDA OOM at batch size {len(samples)}; retrying as "
                    f"{midpoint}+{len(samples)-midpoint}."
                )
                evaluate_with_oom_fallback(samples[:midpoint])
                evaluate_with_oom_fallback(samples[midpoint:])
            except Exception as error:
                for sample in samples:
                    record_failure(adapter.name, sample.name, error)

        for spec in specs:
            try:
                sample = adapter.load(spec)
                if not np.any(sample.evaluation_mask):
                    raise ValueError("no valid evaluation pixels")
                shape = tuple(sample.depth_m.shape)
                shape_buckets[shape].append(sample)
                if len(shape_buckets[shape]) >= batch_size:
                    evaluate_with_oom_fallback(shape_buckets.pop(shape))
            except Exception as error:
                record_failure(adapter.name, spec.name, error)
        for samples in shape_buckets.values():
            evaluate_with_oom_fallback(samples)
        dataset_summary = aggregates.summaries()
        for encoding in encodings:
            values = dataset_summary.get((adapter.name, encoding.name))
            if values is not None:
                log(
                    f"[{adapter.name}] {encoding.name}: "
                    f"EPE={values['epe_m_per_pixel']:.6f}m "
                    f"RMSE={values['rmse_m']:.6f}m "
                    f"Encode={values['encode_median_ms']:.3f}ms/image "
                    f"Decode={values['decode_median_ms']:.3f}ms/image"
                )

    dataset_rows = _summary_rows(aggregates, "dataset")
    bin_rows = []
    for row in _summary_rows(bin_aggregates, "depth_bin"):
        method, label = row["method"].split("|", 1)
        row["method"] = method
        row["depth_bin_m"] = label
        bin_rows.append(row)
    summary_rows = dataset_rows + bin_rows
    _write_csv(output_dir / "per_image_results.csv", per_image_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    payload = {
        "config": config,
        "summary": summary_rows,
        "per_image_results": per_image_rows,
        "failures": failures,
    }
    (output_dir / "results.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        _build_report(config, dataset_rows, bin_rows, failures), encoding="utf-8"
    )
    log(f"Evaluation finished. Results saved to: {output_dir}")
    log(f"Report: {output_dir / 'REPORT.md'}")
    return payload
