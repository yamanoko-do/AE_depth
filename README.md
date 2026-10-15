# AE_depth

`AE_depth` 用于评估图像 Autoencoder 对度量深度图的压缩与重建能力。目前实现
Cosmos Tokenizer `CI16x16`，比较三种单通道深度到 RGB 的编码方式，并以每个有效
像素的平均绝对深度误差 `m/pixel` 为核心指标。

项目的数据列表和 YAML 风格参考：
`/root/host_share/OpenStereo/cfgs/mixdataset_cfg_stage2.yaml`。

## 当前支持范围

- 模型：Cosmos `CI16x16`，连续 BF16 latent，无 codebook。
- 数据集：Middlebury、ETH3D、KITTI 2012、KITTI 2015、Booster、FallingThings。
- 编码方式：
  - `linear_minmax_rgb`：每张图线性 Min-Max 归一化，复制到三个通道。
  - `log_minmax_rgb`：对深度执行 `log1p` 后进行每图 Min-Max，再复制到三个通道。
  - `defm_three_channel`：参考 DeFM，分别编码 100 m 全局 log、9 m 中尺度 log、每图局部 log。
- 指标：EPE、RMSE、AbsRel、Bias、编解码耗时、latent 大小和相对 uint16 深度图的压缩率。
- 深度区间：`0–5 m`、`5–20 m`、`20–50 m`、`50–100 m`。

## 评估文件列表

配置文件为 [cfgs/cosmos_ci16x16.yaml](cfgs/cosmos_ci16x16.yaml)，其中的
`DATA_CONFIG.DATA_INFOS` 直接使用 OpenStereo 的列表：

| 数据集 | 评估列表 | 规则 |
|---|---|---|
| Middlebury | `OpenStereo/data/Middlebury/MiddEval3_train_h.txt` | `trainingH`，非遮挡 GT |
| ETH3D | `OpenStereo/data/ETH3D/ETH3D_train.txt` | two-view training，非遮挡 GT |
| KITTI12 | `OpenStereo/data/KITTI12/kitti12_train194.txt` | 使用 `disp_noc` |
| KITTI15 | `OpenStereo/data/KITTI15/kitti15_train200.txt` | 使用 `disp_noc_0` |
| Booster | `OpenStereo/data/Booster/booster_balanced_train.txt` | Q 分辨率，即宽高缩小 4 倍 |
| FallingThings | `OpenStereo/data/FallingThings/fat_test.txt` | 默认每个场景均匀抽 10 张 |

官方 test split 通常不提供视差 GT，因此这里的“评估集合”是 OpenStereo 配置中本地
可计算定量指标的带 GT 列表，而不是无 GT 的官方提交测试集。

## 视差转换为深度

Middlebury 和 ETH3D：

```text
depth_m = fx_px * baseline_mm / 1000 / (disparity_px + doffs_px)
```

KITTI 2012 使用每张图标定文件中的 `P0/P1`；KITTI 2015 使用
`training/calib_cam_to_cam` 中对应文件的 `P_rect_02/P_rect_03`：

```text
baseline_m = abs(-P_right[0,3] / P_right[0,0]
                 + P_left[0,3] / P_left[0,0])
depth_m = fx_px * baseline_m / disparity_px
```

KITTI PNG 视差先除以 256。Booster 从 XML 读取 `proj_matL` 和 `baselineLR`。
FallingThings 的 uint16 深度单位为 `1e-4 m`，直接转换为米。

KITTI 的 GT 较稀疏。送入 codec 前使用最近有效像素填充空洞，但所有指标仍只在原始
有效 GT 像素上计算。默认只统计 `(0, 100] m` 范围。

## 环境与模型

默认使用已有环境、Cosmos Tokenizer 源码和已经转换的原生 PyTorch 权重：

```text
Python: /root/host_share/miniconda3/envs/sam3d-objects/bin/python
Cosmos源码: /root/host_share/AE_coder/Cosmos-Tokenizer
PyTorch权重: /root/host_share/AE_depth/weight/cosmos_ci16x16.pth
```

运行时实例化开源的 `ContinuousImageTokenizer`，再加载上述 `state_dict`，不会读取
JIT 文件。权重由 Git LFS 跟踪。运行需要 CUDA，当前模型按 BF16 推理。

如需从 NVIDIA 原始 `encoder.jit` 和 `decoder.jit` 重新转换：

```bash
cd /root/host_share/AE_depth
/root/host_share/miniconda3/envs/sam3d-objects/bin/python \
  tools/convert_ci16x16_weights.py
```

转换器会检查两份 JIT 是否完整覆盖原生模型的全部参数，并在 CUDA 上比较转换前后的
latent 和重建输出。当前权重包含 `81,509,088` 个参数，保存格式为 BF16
PyTorch `state_dict`。

## 运行

先对每个数据集抽 1 张做冒烟测试：

```bash
cd /root/host_share
/root/host_share/miniconda3/envs/sam3d-objects/bin/python \
  AE_depth/tools/eval.py \
  --limit_per_dataset 1 \
  --output_dir /root/host_share/outputs/cosmos_depth_smoke
```

运行完整评估列表：

```bash
cd /root/host_share
/root/host_share/miniconda3/envs/sam3d-objects/bin/python \
  AE_depth/tools/eval.py \
  --cfg_file AE_depth/cfgs/cosmos_ci16x16.yaml
```

batch size 由 `BENCHMARK.BATCH_SIZE` 控制。相同分辨率的样本会组成 batch，分辨率不同的样本
分别处理；如果发生 CUDA OOM，程序自动将当前 batch 二分重试。命令行可以覆盖：

```bash
/root/host_share/miniconda3/envs/sam3d-objects/bin/python \
  AE_depth/tools/eval.py \
  --cfg_file AE_depth/cfgs/cosmos_ci16x16.yaml \
  --batch_size 8
```

报告中的编解码耗时是 batch CUDA 同步耗时除以实际 batch size 后的单图摊销值，
`per_image_results.csv` 另外保留 `encode_batch_ms`、`decode_batch_ms` 和实际
`batch_size`。`--limit_per_dataset N` 会在每个数据集列表中均匀选择最多 N 个样本，
`--output_dir` 可覆盖 YAML 中的结果目录。短横线参数写法也兼容。

## 输出

默认输出到 `/root/host_share/outputs/cosmos_depth_datasets`：

```text
REPORT.md                  中文汇总报告
summary.csv                数据集级和深度区间级聚合结果
per_image_results.csv      每张图、每种编码方式的指标与耗时
results.json               配置和完整机器可读结果
examples/<dataset>/<id>/
  comparison.png           GT、重建深度和绝对误差热力图
  reference_depth_m.npy
  *_reconstructed_depth_m.npy
```

EPE 的聚合方式是将所有有效像素的绝对误差相加，再除以有效像素总数，因此单位严格为
`m/pixel`，不是逐图 EPE 的简单平均。报告同时给出 `macro_image_epe_m` 供参考。

## 目录结构

```text
AE_depth/
├── ae_depth/
│   ├── benchmark.py       benchmark 执行、聚合、报告和可视化
│   ├── codecs/            Cosmos codec 和深度 RGB 编码
│   ├── datasets/          各数据集深度与标定适配器
│   ├── io.py              PFM、标定、列表和稀疏深度处理
│   └── metrics.py         逐图与全局指标
├── cfgs/
│   └── cosmos_ci16x16.yaml
├── weight/
│   └── cosmos_ci16x16.pth  Git LFS 跟踪的原生 PyTorch state_dict
└── tools/
    ├── convert_ci16x16_weights.py
    └── eval.py            OpenStereo 风格的评估入口与日志
```

## 后续扩展

后续可在 `ae_depth/codecs` 中增加其他 AE，也可基于这些数据集训练或微调 Cosmos
CI16x16。训练时应保留独立验证划分，避免使用同一深度图同时训练和报告最终指标；模型
接口建议继续统一为 `encode -> latent -> decode`，以便复用当前指标与报告模块。
