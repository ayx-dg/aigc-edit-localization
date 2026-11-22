# AGENTS.md — 仓库说明 / Repository guide

本文件面向在本仓库中工作的 agent 与贡献者：说明**本仓库中这些数据和代码是什么，使用方式**，以及必须遵守的格式与协作约定。

## 1. 仓库目标

研究 Kaggle 竞赛 `ai-edited-image-localization-cs-5489` 的逐像素 AI 编辑区域定位。
核心结论：性能主要由**原生像素保真度**与**和官方全图指标对齐的验证/选模**决定，
而不是编码器本身。历史原生裁剪 U-Net 系列达到 0.90–0.92；SparseViT 全局 512 缩放
0.65764；原生 patch 0.77279；双视图加权集成 0.80870。报告在 `report/report.pdf`。

本仓库面向三类读者：持有完整本地产物的维护者、没有 Kaggle 账号/未加入 repo/使用
Codex 的组员，以及只想从头复现的助教/教授。所有命令使用相对路径或 `AIGC_*`
环境变量，不依赖任何个人电脑、Windows 盘符或特定远程主机。

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
submissions/
  best_submission.csv  固定交接文件；更新它才会触发本仓库的提交步骤
  candidates/          组员 PR 中提议的候选 CSV
  *.csv                归档的最佳提交
experiments/           训练历史、调参/稀疏度记录、混淆矩阵
scripts/               分析、作图、集成、校验与交接脚本
.github/workflows/     Kaggle 交接自动化
```

### 2.2 数据是什么

| 路径 | 内容 | 用途 |
|---|---|---|
| `data/metadata/meta.csv` | `ID, orig_h, orig_w, cache_h, cache_w, area, split`（4500/500/500） | 解码 RLE、按面积分层 |
| `data/metadata/test.csv`, `sample_submission.csv` | 测试 ID 与官方样例格式 | 参考；实际提交必须 2 列 |
| `data/analysis/error_analysis_val.json` | 逐图 Score、最差 10 张、面积分层、连通域统计 | 报告证据 |
| `data/analysis/disagreement_test.json` | 历史 0.92 与 v5 在测试集上的掩码分歧与最差 12 张 | 测试无 GT 时的错分代理 |
| `data/analysis/worst_*.csv` | 最差验证 ID / 最大测试分歧 ID | 直接定位错分图片 |
| `data/error_images/*.png` | 错误叠加图与分歧图 | 报告视觉证据 |
| `report/*.json`, `experiments/*` | Kaggle 分数、阈值扫描、训练曲线/混淆矩阵 | `report/make_report.py` 输入 |
| `submissions/best_submission.csv` | 固定命名的当前最佳提交（v5） | 交接点；维护者更新后推送 |

原始大图像、缓存图/掩码、604 MB SparseViT checkpoint、`.npz` 概率图未入库；
它们应由课程/竞赛提供，或从原始数据本地生成。仓库只保留复现报告所需的元信息、
统计和图像。

### 2.3 代码是什么

- `code/historical_aigc/native.py`：历史高分管线的原生像素裁剪、全分辨率分块验证和官方
  Score 阈值扫描。报告中的 `aigc/native.py:81-108` 对应此文件。
- `code/historical_aigc/core.py` / `models.py`：小型 GroupNorm U-Net、BCE+Dice、EfficientNet-B0 U-Net。
- `code/historical_aigc/distillation.py`：可选蒸馏支持。
- `code/sparsevit/train.py`：缓存或原生 patch 模式训练，按 Score 保存 `best.pt`。
- `code/sparsevit/predict.py`：缓存验证推理、RLE 提交、概率缓存。
- `code/sparsevit/sliding_infer.py`：全分辨率滑动推理，`--mode global|native|both`。
- `code/sparsevit/make_patches.py` / `dataset*.py`：无损 patch 与数据加载。
- `code/sparsevit/model.py` / `sparsevit_model.py` / `SparseViT/`：模型定义与官方源码。
- `code/sparsevit/metrics.py` / `sweep_probs.py` / `make_submission_from_probs.py`：
  指标、阈值扫描、uint8 概率图转提交。
- `scripts/analyze_errors.py`：定位最差验证图与最大测试分歧图。
- `scripts/make_report_figs.py`：生成训练曲线、权衡图、混淆矩阵。
- `scripts/make_ensemble_probs.py`：按 0.4/0.6 合并 global/native 概率图。
- `scripts/submit_kaggle.py`：校验 2 列格式并做交接；自动化步骤调用它。
- `report/make_report.py`：从 JSON + 图生成 `report.tex`。

### 2.4 使用方式

#### 新机器环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # Linux/macOS
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/ayx-dg/aigc-edit-localization.git
cd aigc-edit-localization
uv venv --python 3.11 .venv
source .venv/bin/activate                                # Windows: .venv\Scripts\activate
uv pip install -r requirements-report.txt                # 报告/图（CPU）
uv pip install -r requirements.txt                       # 完整训练/推理
```

重建 PDF 还需 `pdflatex`（TeX Live / MiKTeX）。完整训练需要 CUDA GPU 和已提供的
数据文件；仅复现报告/错误图不需要它们。

#### 复现报告与错误证据

```bash
python report/make_report.py
cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex && cd ..
```

如需重跑错误分析，用环境变量指向外部产物：

```bash
export AIGC_WORKDIR=/path/to/artifacts
export AIGC_CACHE_DIR="$AIGC_WORKDIR/cache"
export AIGC_TEST_IMAGES=/path/to/provided/dataset/test/images
python scripts/analyze_errors.py
```

#### 从头训练/推理（假设数据文件已存在）

数据布局：cache 模式为 `$AIGC_CACHE_DIR/{meta.csv,imgs,masks}`；patch 模式为
`$AIGC_DATA_DIR/{train.csv,train/images,test/images}`。

```bash
export AIGC_DATA_DIR=/path/to/provided/dataset
export AIGC_WORKDIR="$PWD/work"
export AIGC_CACHE_DIR="$AIGC_DATA_DIR/cache"
export AIGC_PATCH_DIR="$AIGC_WORKDIR/patches"
export AIGC_SPARSE_CKPT=/path/to/official_sparsevit_checkpoint.pth

python code/sparsevit/make_patches.py --data-dir "$AIGC_DATA_DIR" --out-dir "$AIGC_PATCH_DIR"
python code/sparsevit/train.py --cache-dir "$AIGC_CACHE_DIR" --out-dir "$AIGC_WORKDIR/out_global" \
  --arch sparsevit --encoder-weights "$AIGC_SPARSE_CKPT" --size 512 \
  --batch-size 8 --epochs 10 --lr 1e-4 --weight-decay 1e-4 --drop-path 0.2 --seed 42
python code/sparsevit/train.py --patch-dir "$AIGC_PATCH_DIR" --out-dir "$AIGC_WORKDIR/out_patch" \
  --arch sparsevit --encoder-weights "$AIGC_SPARSE_CKPT" --size 512 \
  --batch-size 8 --epochs 5 --lr 1e-4 --weight-decay 1e-4 --drop-path 0.2 --seed 42
python code/sparsevit/sliding_infer.py --webp-dir "$AIGC_DATA_DIR/test/images" \
  --out "$AIGC_WORKDIR/sw_global" --ckpt "$AIGC_WORKDIR/out_global/best.pt" \
  --mode global --overlap 0.5 --weight hann --save-probs
python code/sparsevit/sliding_infer.py --webp-dir "$AIGC_DATA_DIR/test/images" \
  --out "$AIGC_WORKDIR/sw_native" --ckpt "$AIGC_WORKDIR/out_patch/best.pt" \
  --mode native --overlap 0.5 --weight hann --save-probs
python scripts/make_ensemble_probs.py \
  "$AIGC_WORKDIR/sw_global/test_probs_512.npz" \
  "$AIGC_WORKDIR/sw_native/test_probs_512.npz" \
  "$AIGC_WORKDIR/ensemble.npz" --w-global 0.4 --w-native 0.6
python code/sparsevit/make_submission_from_probs.py "$AIGC_WORKDIR/ensemble.npz" 0.59 \
  "$AIGC_WORKDIR/candidate.csv"
python scripts/submit_kaggle.py --dry-run --submission "$AIGC_WORKDIR/candidate.csv"
```

阈值用 `code/sparsevit/sweep_probs.py` 在验证集上确定。

#### 协作与交接（组员 / Codex 用户）

1. 无需 Kaggle 账号和 repo 写权限；fork 公开仓库即可开始。
2. 按上面方式建环境，使用仓库已提交的元信息/错误图/报告进行工作。
3. 候选提交 CSV 放到 `submissions/candidates/`，或改代码/文档，在分支上开 PR。
4. PR 描述里写清改动和验证 Score；维护者审核合并后，把选中的候选复制为
   `submissions/best_submission.csv` 并推送。

自动化步骤只把固定文件 `submissions/best_submission.csv` 当作交接点；其他文件的
push 不会启动它。贡献者不需要接触任何凭据。

### 2.5 必须遵守的约定与已验证的坑

1. 提交必须 2 列 `ID,Predicted`；5 列样例格式会被 Kaggle 判 `SubmissionStatus.ERROR`。
2. RLE 为列优先（Fortran）、1-based `(start, length)`；空掩码为空字符串。
3. `*_probs_512.npz` 为 `uint8 0–255`，阈值化前除以 255。
4. `Score = 0.7*mIoU + 0.3*mRecall`，逐图计算后平均；两空记 1，一空一非空记 0。
5. SparseViT 只支持 `--size 512`；其他尺寸会触发官方 `alter_sparse` 断言。
6. `predict.py` / `sliding_infer.py` 把 `SparseViT/` 放在 `sys.path` 的脚本目录之前，
   否则 `from SparseViT import SparseViT` 会绑定到包而不是类。
7. Windows 下预测需 `--workers 0`；`TestDS.__getitem__` 返回 `(image, dummy_label)`。
8. 部分 torch 轮子的 cuDNN 有问题；默认关闭，只有确认可用才加 `--cudnn`。
9. patch 缓存必须无损 PNG/uint8，不能 JPEG。
10. 所有外部路径通过 `AIGC_*` 环境变量传入；不要把个人路径写回仓库。

## 3. 自动化说明

`.github/workflows/kaggle_submit.yml` 仅在 push 包含
`submissions/best_submission.csv` 变更时运行（或手动 dispatch）。**不是每个新提交都直接
当成最终成绩**：`scripts/gate_best_submission.py` 在运行时查询本人在 Kaggle 的提交历史
（离线兜底 `0.92213`，ref `58415875`）作为历史最高，与候选 sidecar
`submissions/best_submission.score.json` 中的成绩比较：

- 候选严格高于历史最高 → 归档该 CSV、更新 `submissions/score_registry.json`，再提交 Kaggle。
- 候选不高于历史最高 → 不提交；按 registry 把固定 CSV 回退为归档的最高分 CSV，并重写
  sidecar，保证仓库与竞赛成绩一致。

sidecar 的 `sha256` 必须与 CSV 相符，否则闸门直接失败（fail closed）。手动 dry-run 可无副作用
验证两条分支：`gh workflow run kaggle_submit.yml -f dry_run=true -f score=0.99`（SUBMIT）与
`-f score=0.50`（KEEP），再用 `gh run view --log` 看 `decision=`。凭据只由维护者配置；贡献者
无需接触，也不要在 PR 中加入任何密钥或 token。
