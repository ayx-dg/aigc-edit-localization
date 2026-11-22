# AGENTS.md — AIGC 编辑区域定位复现与报告

本文件面向在本仓库中工作的 agent 与贡献者：说明本仓库中数据与代码是什么、如何从零使用，以及必须遵守的格式与协作约定。人读的入口是 [README.md](README.md)；两份文档服务不同读者。

## Scope / 范围

本目录是一个独立、自包含的 Kaggle 竞赛 `ai-edited-image-localization-cs-5489` 复现与报告。只在本仓库内工作。所有命令使用相对路径或 `AIGC_*` 环境变量，不依赖任何个人电脑、盘符、远程主机或私有环境。仓库根目录就是项目目录：`clone` 后直接进入 `aigc-edit-localization/`，没有再嵌一层的子目录。

核心结论：性能主要由**原生像素保真度**与**和官方全图指标对齐的验证/选模**决定，而不是编码器本身。历史原生裁剪 U-Net 系列公开成绩 0.90–0.92；SparseViT 全局 512 缩放 0.65764；原生 patch 0.77279；双视图加权集成 0.80870。报告在 `report/report.pdf`（13 页）。

## Team Handoff and Working with Codex / 协作与交接

公开仓库对贡献者只读；所有修改通过 fork + Pull Request 进入主仓库。贡献者不需要 Kaggle 账号或仓库写权限，也不需要接触任何凭据。

```bash
# 先在 GitHub 上 fork https://github.com/ayx-dg/aigc-edit-localization
git clone https://github.com/<your-username>/aigc-edit-localization.git
cd aigc-edit-localization
git remote add upstream https://github.com/ayx-dg/aigc-edit-localization.git

# 每次开始工作前同步上游
git checkout main
git pull --rebase upstream main
git checkout -b codex/<your-topic>
# 例：git checkout -b codex/error-analysis-fix

git add -A
git commit -m "Describe the concrete change"
git push -u origin codex/<your-topic>
# 然后在 GitHub 上从你的 fork 向 ayx-dg/main 开 PR，等待维护者审核合并
```

给 agent 的交接规则：

1. 改动前先读 `README.md` 与 `AGENTS.md`。本文件是规则、数据清单、种子、阈值、已知限制与「不要做什么」的权威来源。
2. 工作目录是 clone 后的仓库根目录，不要去找任何不存在的嵌套目录。
3. 新协作者使用自己的 Git 身份提交；不要改写历史提交。
4. 任何源码改动都必须完成本文件「Reproduction contract」一节规定的完整流程：重跑受影响的管线、重建图与报告、更新报告中的数字、重新编译 PDF。即使只改图或报告文字，也要重建 PDF。
5. 按「从全新机器复现」的说明建环境。`requirements*.txt` 固定依赖；不要提交 `.venv/` 或临时大文件。
6. 不要写死个人路径、盘符或远程主机名；外部路径一律通过 `AIGC_*` 环境变量传入。

## Source of truth / 任务、数据与指标的权威来源

- 任务：预测每个像素是否为 AI 编辑区域。提交为 RLE 字符串；RLE 是**列优先（Fortran）、1-based `(start, length)`**，空掩码为空字符串。实现见 `code/common/rle.py`，已验证 round-trip。
- 指标：`Score = 0.7 * mIoU + 0.3 * mRecall`，先逐图计算再平均；两图皆空时两者记 1，仅一图为空时两者记 0。权威实现是 `code/sparsevit/metrics.py`，不要另写一套口径。
- 提交格式：必须是两列 `ID,Predicted`。5 列 `sample_submission.csv` 样例格式会被 Kaggle 判 `SubmissionStatus.ERROR`。
- 概率缓存：`*_probs_512.npz` 存的是 `uint8` `[0,255]`，阈值化前必须除以 255。
- 数据划分：4500 训练 / 500 验证 / 500 测试，种子 42；全局训练 10 epoch，patch 训练 5 epoch；最佳集成权重 `w_global=0.4`、`w_patch=0.6`，阈值 0.59。
- 公开成绩：历史最高 `0.92213`（ref `58415875`）；SparseViT v5 集成 `0.80870`（ref `58688905`）。不要把 v5 误写成历史最高；闸门以运行时 Kaggle 提交历史为准，离线兜底见 `submissions/score_registry.json`。

### 仓库中数据的清单

| 路径 | 内容 | 用途 |
|---|---|---|
| `data/metadata/meta.csv` | `ID, orig_h, orig_w, cache_h, cache_w, area, split`（4500/500/500） | 解码 RLE、按面积分层 |
| `data/metadata/test.csv`, `sample_submission.csv` | 测试 ID 与官方样例格式 | 参考；实际提交必须两列 |
| `data/analysis/error_analysis_val.json` | 逐图 Score、最差 10 张、面积分层、连通域统计 | 报告证据 |
| `data/analysis/disagreement_test.json` | 历史 0.92 与 v5 在测试集上的掩码分歧与最差 12 张 | 测试无 GT 时的错分代理 |
| `data/analysis/worst_*.csv` | 最差验证 ID / 最大测试分歧 ID | 直接定位错分图片 |
| `data/error_images/*.png` | 错误分类的逐像素叠加图 | 报告视觉证据 |
| `report/*.json`, `experiments/*` | Kaggle 分数、阈值扫描、训练曲线/混淆矩阵 | `report/make_report.py` 输入 |
| `submissions/best_submission.csv` | 固定命名的当前交接文件 | 维护者更新后推送才触发自动化 |

原始大图、缓存图/掩码、604 MB SparseViT checkpoint、`.npz` 概率图未入库；它们应由课程/竞赛提供，或从原始数据按下面的命令本地生成。

## Reproduction contract / 复现契约

Python 3.11，依赖由 `requirements.txt` 与 `requirements-report.txt` 给出；首次安装需要联网，之后用已提交的 JSON/PNG 复现报告不需要联网。仅重建报告用 CPU 即可，完整训练/推理需要 CUDA GPU 与已提供的数据文件。所有路径相对仓库根目录或通过 `AIGC_*` 传入。

### 全新机器：环境与报告

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # Linux/macOS
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/ayx-dg/aigc-edit-localization.git
cd aigc-edit-localization
uv venv --python 3.11 .venv
source .venv/bin/activate                                # Windows: .venv\Scripts\activate
uv pip install -r requirements-report.txt                # 报告/图（CPU）
uv pip install -r requirements.txt                       # 完整训练/推理

python report/make_report.py
cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex && cd ..
```

期望产物：13 页 `report/report.pdf`，数字与 `report/*.json`、`experiments/*` 一致。

如需重跑错误分析，用环境变量指向外部产物：

```bash
export AIGC_WORKDIR=/path/to/artifacts
export AIGC_CACHE_DIR="$AIGC_WORKDIR/cache"
export AIGC_TEST_IMAGES=/path/to/provided/dataset/test/images
python scripts/analyze_errors.py
```

### 全新机器：从头训练与推理

数据布局：cache 模式为 `$AIGC_CACHE_DIR/{meta.csv,imgs,masks}`；patch 模式为 `$AIGC_DATA_DIR/{train.csv,train/images,test/images}`。官方 SparseViT checkpoint 由课程/竞赛提供。

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

阈值用 `code/sparsevit/sweep_probs.py` 在验证集上确定；上面是报告中的最佳配置。每次改动源码后，重跑受影响阶段 → `scripts/make_report_figs.py` → `report/make_report.py` → 更新 `report.tex` 与 JSON 中的数字 → 重新编译 PDF 两次 → 若提交 CSV 变化，更新 `submissions/best_submission.score.json` 的 `score` 与 `sha256`。

## Technical notes / 代码与数据清单、已验证的坑

### 代码是什么

- `code/historical_aigc/native.py`：历史高分管线的原生像素裁剪、全分辨率分块验证与官方 Score 阈值扫描。报告中的 `aigc/native.py:81-108` 对应此文件。
- `code/historical_aigc/core.py` / `models.py`：小型 GroupNorm U-Net、BCE+Dice、EfficientNet-B0 U-Net；`distillation.py` 是可选蒸馏支持。
- `code/sparsevit/train.py`：cache 或原生 patch 模式训练，按 Score 保存 `best.pt`。
- `code/sparsevit/predict.py`：cache 验证推理、RLE 提交、概率缓存。
- `code/sparsevit/sliding_infer.py`：全分辨率滑动推理，`--mode global|native|both`。
- `code/sparsevit/make_patches.py` / `dataset*.py`：无损 patch 与数据加载。
- `code/sparsevit/model.py` / `sparsevit_model.py` / `SparseViT/`：模型定义与官方源码（CC BY 4.0）。
- `code/sparsevit/metrics.py` / `sweep_probs.py` / `make_submission_from_probs.py`：指标、阈值扫描、uint8 概率图转提交。
- `scripts/analyze_errors.py`：定位最差验证图与最大测试分歧图。
- `scripts/make_report_figs.py`：生成训练曲线、mIoU-vs-Recall 权衡图、混淆矩阵图。
- `scripts/make_ensemble_probs.py`：按 0.4/0.6 合并 global/native 概率图。
- `scripts/submit_kaggle.py`：校验两列格式并交接；自动化步骤调用它。
- `report/make_report.py`：从 JSON + 图生成 `report.tex`。

### 已验证的坑（不要 silently 改动）

1. 提交必须两列 `ID,Predicted`；5 列样例格式会被 Kaggle 判 `SubmissionStatus.ERROR`。
2. RLE 为列优先（Fortran）、1-based `(start, length)`；空掩码为空字符串。
3. `*_probs_512.npz` 为 `uint8 0–255`，阈值化前除以 255。
4. `Score = 0.7*mIoU + 0.3*mRecall`，逐图计算后平均；两空记 1，一空一非空记 0。
5. SparseViT 只支持 `--size 512`；其他尺寸会触发官方 `alter_sparse` 断言。
6. `predict.py` / `sliding_infer.py` 必须把 `SparseViT/` 放在 `sys.path` 中脚本目录之前；否则 `from SparseViT import SparseViT` 会绑定到包而不是类，推理报 `TypeError: 'module' object is not callable`。
7. 在 Windows / spawn 环境下预测需 `--workers 0`；`TestDS.__getitem__` 返回 `(image, dummy_label)`，因为 `predict_probs` 从每个 batch 解包两个值。
8. 部分 torch 轮子的 cuDNN 有问题；默认关闭，只有确认本地轮子可用才加 `--cudnn`，不要反过来「修好」它。
9. patch 缓存必须无损 PNG/uint8，不能 JPEG：压缩伪影本身就是信号。
10. 全局滑动把整图缩放到方形 512×512，与训练时的保宽高比 padding 不同；原生 sliding 用全局模型效果差，属于尺度不匹配。报告中的集成使用 global + native-patch 两视图。
11. 所有外部路径通过 `AIGC_*` 环境变量传入；不要把个人路径写回仓库。`scripts/analyze_errors.py` 使用的完整变量清单见 `data/README.md`。

## Automated submission / 自动化提交与评审证据

`.github/workflows/kaggle_submit.yml` 仅在 push 包含 `submissions/best_submission.csv` 变更时运行（或手动 dispatch）。**不是每个新提交都直接当成最终成绩**：`scripts/gate_best_submission.py` 在运行时查询本人 Kaggle 提交历史（离线兜底 `0.92213`，ref `58415875`）作为历史最高，与候选 sidecar `submissions/best_submission.score.json` 中的成绩比较：

- 候选严格高于历史最高 → 归档该 CSV、更新 `submissions/score_registry.json`，再提交到 Kaggle。
- 候选不高于历史最高 → 不提交；按 registry 把固定 CSV 回退为归档的最高分 CSV，并重写 sidecar，保证仓库与竞赛成绩一致。

sidecar 的 `sha256` 必须与 CSV 相符，否则闸门直接失败（fail closed）。手动 dry-run 可无副作用验证两条分支：

```bash
gh workflow run kaggle_submit.yml -f dry_run=true -f score=0.99   # 期望 decision=SUBMIT
gh workflow run kaggle_submit.yml -f dry_run=true -f score=0.50   # 期望 decision=KEEP
gh run view --log
```

凭据只由维护者配置在 GitHub Secrets 中；贡献者无需接触，也不要在 PR 中加入任何密钥或 token。可选的人工评审证据（如 paper-judge 草稿）只保留在本地，不随公开仓库分发。

## Things NOT to do / 禁止事项

- 不要写死个人路径、盘符、远程主机名或任何机器专属环境；外部路径一律用 `AIGC_*`。
- 不要提交 `.venv/`、`work/` 下的大文件或临时产物。
- 不要提交 5 列样例格式的 CSV；提交必须两列 `ID,Predicted`。
- 不要把 `0.80870` 当成历史最高；历史最高是 `0.92213`，闸门口径以运行时 Kaggle 历史为准。
- 不要修改 RLE 约定，也不要忘记概率图是 uint8、阈值前要除以 255。
- 不要静默混用历史原生管线与 SparseViT 的结果；两者在报告中都保留并分别标注。
- 不要在未确认本地 torch 轮子可用时启用 `--cudnn`；默认关闭。
- 不要对 SparseViT 使用非 512 尺寸。
- 不要改名或删除报告身份信息，不要提交与 JSON/LaTeX 数字不一致的 PDF。
- 不要在 PR 中加入 Kaggle token、API key 或任何凭据。
