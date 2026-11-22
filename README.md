# AIGC 编辑区域定位 — SparseViT 与历史原生管线

本仓库是 Kaggle 竞赛 `ai-edited-image-localization-cs-5489`（逐像素 AI 编辑区域定位）的公开配套仓库。它包含历史约 0.92 的原生分辨率管线、当前 SparseViT 管线、课程报告（源码 + PDF）、错误定位数据与图片、参考提交，以及仓库自有的 Kaggle 交接自动化。它同时服务四类读者：

| 读者 | 你应该先看 |
| --- | --- |
| 本地维护者 / 接手的组员 | [仓库结构与约定](#仓库结构与约定) |
| Codex、Claude Code 等 agent | [AGENTS.md](AGENTS.md) |
| 评分老师 / 助教 | [从全新机器复现](#从全新机器复现)、[报告与仓库的对应](#报告与仓库的对应) |
| 只想看结果的人 | 直接打开 `report/report.pdf`，不需要运行任何命令 |

**仓库根目录就是本项目目录**：`git clone` 之后直接进入 `aigc-edit-localization/`，里面就是 `README.md`、`AGENTS.md`、`report/`、`code/` 等文件，不存在再嵌一层的子目录。

---

## 仓库结构与约定

```
aigc-edit-localization/       # clone 后的根目录 = 本目录
├── AGENTS.md                 # agent / 贡献者必读的规则、数据清单与复现契约
├── README.md                 # 本文件：人读的入口
├── requirements.txt          # 完整训练/推理依赖（包含报告依赖）
├── requirements-report.txt   # 仅报告 + 作图所需依赖（CPU）
├── .github/workflows/
│   └── kaggle_submit.yml     # 固定 CSV 的交接步骤：闸门通过后提交到 Kaggle
├── code/
│   ├── common/rle.py         # 已验证的 RLE 编解码（列优先、1-based）
│   ├── historical_aigc/      # 历史 0.90–0.92 原生裁剪管线
│   └── sparsevit/            # 当前 SparseViT 训练、推理、数据处理、评估脚本
│       └── SparseViT/        # 上游 AAAI2025 源码（CC BY 4.0）
├── data/
│   ├── metadata/             # 划分元信息、测试 ID、样例提交
│   ├── analysis/             # 错误分析 JSON 与最差图片 ID 列表
│   └── error_images/         # 逐像素错误/分歧叠加图（PNG）
├── report/                   # report.pdf / report.tex / 生成器 / 图 / 分数 JSON
├── experiments/              # 最终训练记录、调参与稀疏度实验
├── scripts/                  # 分析、作图、集成、闸门与交接工具
├── submissions/
│   ├── best_submission.csv            # 固定交接文件；更新它才会触发提交步骤
│   ├── best_submission.score.json     # 候选成绩与 CSV 的 sha256（必须匹配）
│   ├── candidates/                    # PR 中提议的候选 CSV
│   ├── historical_best_0.92213.csv    # 归档的历史最佳
│   └── sparsevit_v5_0.80870.csv       # 归档的 v5 集成提交
└── work/                     # 本地生成的大文件（已被 git 忽略）
```

原始 WebP 图片、缓存图/掩码、无损 patch 缓存、官方 604 MB SparseViT checkpoint 与 `.npz` 概率图**不在仓库中**。它们由竞赛/课程提供，或按下面的命令从原始数据本地生成。仓库只保留复现报告所需的元信息、统计与图像。

---

## 从全新机器复现

这一节写给没有见过我们终端历史的人：按下面顺序执行即可得到本报告的全部数字和 PDF。命令只使用相对路径与 `AIGC_*` 环境变量，不依赖任何个人电脑、Windows 盘符或特定远程主机。

### 1. 前置依赖

需要 `git`、`uv`；重建 PDF 还需要 `pdflatex`；完整训练/推理需要 CUDA GPU 与已提供的数据文件。仅查看报告不需要 GPU 或 API key。

```bash
# 安装 uv（若已安装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell): powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

若只重建报告，安装 `requirements-report.txt` 即可；若要跑完整管线，安装 `requirements.txt`。首次安装需要联网，之后的实验阶段使用仓库中已提交的 JSON/PNG，不再联网。

Debian/Ubuntu 的 LaTeX 依赖可用：

```bash
sudo apt-get install -y texlive-latex-base texlive-latex-extra \
                        texlive-fonts-recommended texlive-pictures
```

### 2. 拉取代码

```bash
git clone https://github.com/ayx-dg/aigc-edit-localization.git
cd aigc-edit-localization
uv venv --python 3.11 .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -r requirements-report.txt   # 报告 + 图（CPU）
uv pip install -r requirements.txt          # 完整管线（额外加入 torch/timm/smp/kaggle）
```

### 3. 复现报告与错误证据（已提交数据）

```bash
python report/make_report.py
cd report && pdflatex -interaction=nonstopmode report.tex \
          && pdflatex -interaction=nonstopmode report.tex && cd ..
```

期望得到 13 页的 `report/report.pdf`。若要重跑错误分析本身，用环境变量指向外部产物：

```bash
export AIGC_WORKDIR=/path/to/artifacts
export AIGC_CACHE_DIR="$AIGC_WORKDIR/cache"
export AIGC_TEST_IMAGES=/path/to/provided/dataset/test/images
python scripts/analyze_errors.py
```

### 4. 从头训练与推理（数据集与 checkpoint 已由课程/竞赛提供）

数据支持两种布局：cache 模式要求 `$AIGC_CACHE_DIR` 下有 `meta.csv`、`imgs/`、`masks/`；patch 模式要求 `$AIGC_DATA_DIR` 下有 `train.csv`、`train/images/*.webp`、`test/images/*.webp`。

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

阈值用 `code/sparsevit/sweep_probs.py` 在验证集上确定；上面给出的数值复现最佳报告配置。

### 5. 产物核对

| 产物 | 内容 / 期望值 |
| --- | --- |
| `report/report.pdf` | 13 页；数字来自 `report/*.json`、`experiments/*`、`data/analysis/*` |
| `data/analysis/` | 验证集误差分层、最差图片 ID；测试集两模型分歧 |
| `data/error_images/` | 报告第 5 节使用的错误叠加图 |
| `submissions/best_submission.csv` | 固定交接文件；其 `score.json` 的 `sha256` 必须与 CSV 一致 |
| `submissions/candidates/` | PR 中提议的两列候选提交 |

若修改代码或数据，需重跑相应管线、重新生成图与报告，并保持 PDF、JSON、LaTeX 表格中的数字一致。协作与闸门规则见 [AGENTS.md](AGENTS.md)。

---

## For Codex

If you are an agent, see [AGENTS.md](AGENTS.md) for contributing rules.

## 报告与仓库的对应

- `report/report.tex` 是报告源，`report/report.pdf` 是提交稿；图由 `scripts/make_report_figs.py` 生成。
- 报告中的每个数字都能在 `report/*.json`、`experiments/*`、`data/analysis/*` 中找到；提交成绩在 `submissions/score_registry.json` 与 `report/kaggle_score*.json`。
- `submissions/best_submission.csv` 是固定交接文件；候选 CSV 放在 `submissions/candidates/`，由维护者在审核后复制到固定文件并更新 sidecar。
- `data/error_images/` 中的 PNG 是报告中的视觉证据；大图、checkpoint 和概率图不随仓库分发。
- 自动化步骤只把固定文件名 `submissions/best_submission.csv` 当作交接点；其他文件的 push 不会启动它，贡献者不需要接触任何凭据。

如果只想核对结论，直接打开 `report/report.pdf`：性能主要由像素保真度与官方全图指标对齐的验证决定；历史原生 U-Net 系列约 0.92，SparseViT 全局 512 缩放 0.65764，原生 patch 0.77279，双视图加权集成 0.80870；剩余错误集中在小面积、低对比度编辑。

---

## 关键事实

- 接受的提交格式是两列 `ID,Predicted`；5 列样例格式会被 Kaggle 判 `SubmissionStatus.ERROR`。
- RLE 为列优先（Fortran）、1-based `(start, length)`，空掩码为空字符串。
- `*_probs_512.npz` 中的概率是 `uint8` `[0,255]`，阈值化前必须除以 255。
- `Score = 0.7 * mIoU + 0.3 * mRecall`，逐图计算后平均；两图皆空记 1，仅一图为空记 0。
- 历史最高公开成绩为 `0.92213`（ref `58415875`）；v5 集成的 `0.80870` 不是历史最高。

上游 SparseViT 源码以 CC BY 4.0 保留在 `code/sparsevit/SparseViT/LICENSE`。
