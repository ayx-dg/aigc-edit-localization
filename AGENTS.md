# AGENTS.md — 仓库说明 / Repository guide

本文件面向在本仓库中工作的 agent 与贡献者：说明**本仓库中这些数据和代码是什么，使用方式**，以及必须遵守的格式约定。

## 1. 仓库目标

研究 Kaggle 竞赛 `ai-edited-image-localization-cs-5489` 的逐像素 AI 编辑区域定位。
核心结论是：性能主要由**原生像素保真度**与**和官方全图指标对齐的验证/选模**决定，
而不是编码器本身。历史原生裁剪 U-Net 系列达到 0.90–0.92；SparseViT 全局 512 缩放
只有 0.65764；改用原生 patch 后 0.77279；双视图加权集成后 0.80870。
报告在 `report/report.pdf`，源码在 `report/`。

## 2. 本仓库中这些数据和代码是什么，使用方式

### 2.1 目录树

```
code/
  historical_aigc/     历史 0.92 模型代码（native.py / core.py / models.py / distillation.py）
  sparsevit/           当前 SparseViT 训练、推理、数据处理、评估脚本
  sparsevit/SparseViT/ 官方 AAAI2025 SparseViT 源码（CC BY 4.0）
  common/rle.py        经验证的 RLE 编解码（列优先、1-based）
data/
  metadata/            数据集划分元信息（meta.csv / test.csv / sample_submission.csv）
  analysis/            验证/测试错误统计 JSON 与最差图片 ID 列表
  error_images/        错误分类的逐像素叠加图（PNG）
report/                最终英文报告（PDF + LaTeX 源码 + 图 + 数据表）
submissions/           历史 0.92213 与当前 0.80870 两份可提交 CSV
experiments/           最终实验的历史曲线/混淆矩阵/最佳信息
scripts/               可复用的分析、作图、提交脚本
.github/workflows/     固定 CSV 更新并 push 后自动提交 Kaggle 的 GitHub Action
```

### 2.2 数据是什么

| 路径 | 内容 | 用途 |
|---|---|---|
| `data/metadata/meta.csv` | 每张图的 `ID, orig_h, orig_w, cache_h, cache_w, area, split`（4500 train / 500 val / 500 test） | 解码 RLE、按图统计面积分层 |
| `data/metadata/test.csv`, `sample_submission.csv` | 测试集 ID 与官方样例格式 | 参考；实际提交必须用 2 列 |
| `data/analysis/error_analysis_val.json` | 500 张验证图的逐图 Score、最差 10 张、面积分层、连通域统计 | 报告第 3、4 节的数据依据 |
| `data/analysis/disagreement_test.json` | 历史 0.92 与 v5 在 500 张测试图上的掩码分歧、面积比、最差 12 张 | 测试集无 GT 时的高概率错分代理 |
| `data/analysis/worst_*.csv` | 最差验证 ID / 最大测试分歧 ID 列表 | 直接定位错分图片 |
| `data/error_images/*.png` | 验证错误叠加图、测试分歧图、面积–Score 散点、权衡图 | 报告中错误分析的视觉证据 |
| `report/*.json`, `experiments/*.json` | Kaggle 提交分数、滑动窗口阈值扫描、最终曲线/混淆矩阵 | `report/make_report.py` 的输入 |
| `submissions/best_submission.csv` | 固定命名的当前最佳提交（v5 副本，500 行、2 列） | GitHub Action 监视此文件；更新并 push 才会触发 Kaggle 提交 |
| `submissions/historical_best_0.92213.csv`, `submissions/sparsevit_v5_0.80870.csv` | 归档的两份最佳提交 | 复现/对比，Action 默认不提交它们 |

**未入库的数据**：原始 `webp` 图像、512×512 缓存图/掩码、604 MB 官方 SparseViT
checkpoint、`.npz` 概率图。它们体积大且受竞赛条款约束；需要时按 `data/README.md`
重新生成或从原始竞赛下载。

### 2.3 代码是什么

- `code/historical_aigc/native.py`：历史高分管线的核心。原生像素 512 裁剪、编辑中心采样、
  全分辨率分块推理与 **官方 Score** 验证/阈值扫描。报告中的 `aigc/native.py:81-108`
  即指此文件（仓库内路径为 `code/historical_aigc/native.py`）。
- `code/historical_aigc/core.py` / `models.py`：小型 GroupNorm U-Net、BCE+Dice 损失、
  EfficientNet-B0 U-Net 配置。历史 0.90638 提交就是后者 + 原生 512 + 单视图。
- `code/historical_aigc/distillation.py`：教师–学生蒸馏支持（可选）。
- `code/sparsevit/train.py`：缓存模式与原生 patch 模式训练；每 epoch 记录 loss/Dice/mIoU/
  mRecall/Score/阈值，保存 `last.pt` 与按 Score 选的 `best.pt`。
- `code/sparsevit/predict.py`：缓存验证推理 + RLE 提交 + 概率缓存。
- `code/sparsevit/sliding_infer.py`：全分辨率滑动推理，`--mode global|native|both`，Hann 融合。
- `code/sparsevit/dataset.py` / `dataset_patch.py` / `make_patches.py`：数据加载与无损 PNG patch 构建。
- `code/sparsevit/model.py` / `sparsevit_model.py` / `SparseViT/`：模型定义与官方源码。
- `code/sparsevit/metrics.py` / `sweep_probs.py` / `make_submission_from_probs.py`：
  指标、阈值扫描、uint8 概率图到提交的转换。
- `scripts/analyze_errors.py`：定位最差验证图与最大测试分歧图，生成 `data/analysis/*.json`
  与 `report/figures/fig_*.pdf`。
- `scripts/make_report_figs.py`：从实验记录生成训练曲线、权衡图、混淆矩阵等。
- `report/make_report.py`：从 JSON + 图生成 `report/report.tex`。
- `scripts/submit_kaggle.py`：校验 2 列格式并提交；GitHub Action 调用它。

### 2.4 使用方式

**只读/复现报告**（不需要大型数据）：

```bash
cd report
python make_report.py
pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex
```

**重新定位错分图片**（需要外部实验产物）：

```bash
export AIGC_WORKDIR=/path/to/original/experiment   # 含 cache/ 与 pipeline_results/
export AIGC_TEST_IMAGES=/path/to/test/images       # 原始 test/*.webp
python scripts/analyze_errors.py
```

可选环境变量：`AIGC_CACHE_DIR`、`AIGC_GLOBAL_PROBS`、`AIGC_PATCH_PROBS`、
`AIGC_OLD_SUBMISSION`、`AIGC_NEW_SUBMISSION`、`AIGC_REPORT_DIR`。默认值均指向本仓库
或 `$AIGC_WORKDIR` 下的常规位置。生成 PDF 后可用 `pdftoppm -png -r 140` 转成
`data/error_images/` 中的 PNG。

**重新生成报告图**（需要实验记录）：

```bash
AIGC_WORKDIR=/path/to/original/experiment python scripts/make_report_figs.py
```

**训练/推理**（Windows 远程机 `ac1`，Python 路径仅作示例）：

```bash
PY=C:/Users/Lenovo/.conda/envs/BME6145/python.exe
CKPT=C:/Users/Lenovo/unet_full/SparseViT/checkpoint/checkpoint.pth
$PY code/sparsevit/train.py --cache-dir .../cache --out-dir .../out_final_4500 \
  --arch sparsevit --encoder-weights $CKPT --size 512 --batch-size 8 --epochs 10 \
  --lr 1e-4 --weight-decay 1e-4 --drop-path 0.2 --seed 42 --workers 2
$PY code/sparsevit/train.py --patch-dir .../patches --out-dir .../out_patch \
  --arch sparsevit --encoder-weights $CKPT --size 512 --batch-size 8 --epochs 5
$PY code/sparsevit/sliding_infer.py --webp-dir .../test_webp --out .../sw_patch_test_native \
  --ckpt .../out_patch/best.pt --mode native --overlap 0.5 --weight hann --save-probs
```

**提交 Kaggle**：

```bash
python scripts/submit_kaggle.py --dry-run                    # 只校验固定文件
KAGGLE_USERNAME=xxx KAGGLE_KEY=yyy python scripts/submit_kaggle.py
```

自动提交通道：**固定文件名是 `submissions/best_submission.csv`**。只有该文件发生
变更并 push 到 `main`，`.github/workflows/kaggle_submit.yml` 才会触发提交；
其他文件（代码、报告、文档）的 push 不会提交 Kaggle。要提交新模型，先用新的
2 列 `ID,Predicted` CSV 覆盖该固定文件再 push。需要先设置仓库 secrets
`KAGGLE_USERNAME` 与 `KAGGLE_KEY`，并在 Kaggle 网页接受竞赛规则；secrets 缺失时
workflow 只告警并跳过，不会使 push 失败。

## 3. 必须遵守的约定与已验证的坑

1. 提交文件必须是 **2 列** `ID,Predicted`；5 列样例格式会被 Kaggle 判 `SubmissionStatus.ERROR`。
2. RLE 为**列优先（Fortran）、1-based `(start, length)`**，空掩码为空字符串。
3. `*_probs_512.npz` 中是 `uint8 0–255`，阈值化前必须除以 255。
4. `Score = 0.7*mIoU + 0.3*mRecall`，逐图计算后平均；两空掩码记 1，一空一非空记 0。
5. SparseViT 只支持 `--size 512`；其他尺寸会触发官方 `alter_sparse` 断言。
6. `predict.py` / `sliding_infer.py` 必须把 `ROOT/SparseViT` 放在 `sys.path` 的 `ROOT` 之前，
   否则 `from SparseViT import SparseViT` 会绑定到包而不是类。
7. Windows 下预测需 `--workers 0`（`TestDS` 定义在 `main()` 内，spawn 无法 pickle）；
   `TestDS.__getitem__` 必须返回 `(image, dummy_label)`。
8. ac1 的 torch 轮子 cuDNN 损坏，训练保持默认 `--cudnn` 关闭，不要“修好”它。
9. patch 缓存必须是无损 PNG/uint8，不能 JPEG；压缩伪影本身就是信号。
10. 本仓库 git 作者固定为 `user.name=martin`、`user.email=""`（空邮箱，按要求）。

## 4. GitHub Actions 说明

`.github/workflows/kaggle_submit.yml`：

- 触发：`push` 到 `main` 且路径包含 `submissions/best_submission.csv`，或手动
  `workflow_dispatch`（可覆盖 CSV 路径与提交信息）。
- 步骤：checkout → setup-python → `submit_kaggle.py --dry-run` 校验 → 检查 secrets →
  `pip install kaggle` → 提交并打印最近提交列表。
- 默认提交固定文件 `submissions/best_submission.csv`（当前内容为 v5，Kaggle 0.80870）。
- 想停止自动提交：删除 workflow 或修改 `on.push.paths`。
