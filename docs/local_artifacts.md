# 本地数据与研究产物整理

`data/` 和 `factor_report/` 都是不纳入 Git 的本地资产，但用途不同。整理时不能只按
文件大小删除；行情输入、时点上下文、模型窗口和运行清单都可能是不可廉价重建的研究证据。

## 固定目录

以下目录保留在固定位置，正式命令会直接读取或写入：

```text
data/raw/                 原始日线行情与抓取 manifest
data/context/             历史时点上下文、基准与风格因子
data/candidates/          自定义策略候选
data/kline/               点时候选图表
data/review/              Gemini 复评产物
data/backtest/            当前标准信号收益输出
data/portfolio_backtest/  当前标准组合回测输出
data/ml/<experiment>/     ML 数据集、逐窗模型、预测与统一信号
data/reports/             综合报告
data/runs/<run-id>/       正式 CLI 的运行清单和日志
factor_report/            单因子、批处理和相关性结果
```

`data/ml/` 体积通常最大，但其中的 `features.csv`、逐窗预测、模型文件和 manifest 是完整
walk-forward 实验的一部分。除非已确认实验不再需要，否则不应把它当作缓存清理。

## 历史归档

非标准名称的 `data/portfolio_backtest_*` 和 `data/backtest_*` 属于一次性实验，整理工具会
在连续 7 天没有更新后移动到：

```text
data/archive/portfolio_backtests/
data/archive/signal_backtests/
```

旧版单体 `factor-run-all` 的产物移动到：

```text
factor_report/archive/legacy_workflows/factor_run_all/
```

归档只移动、不删除研究文件，并在 `data/archive/archive_index.json` 记录原路径、归档路径、
体积、原因和时间。旧 manifest 中的绝对输出路径是历史证据，不会重写；若要从断点继续，
应把整目录移回索引记录的原路径。

## 使用整理工具

先只读预览：

```bash
python scripts/organize_workspace.py
```

确认清单后执行：

```bash
python scripts/organize_workspace.py --apply
```

执行时会归档上述历史输出，并删除 `.DS_Store`、工具缓存、根目录 `build/`、
`*.egg-info/` 和空输出目录。macOS 上部分源码可能是 dataless 文件，Python 字节码缓存
可能是离线启动所需的唯一可读副本，因此整理工具会保留 `__pycache__/`。它不会删除 `data/raw/`、`data/context/`、
`data/ml/`、标准输出目录或任何非空研究产物。归档目标已存在时会拒绝覆盖并返回非零。
可用 `--minimum-age-days N` 调整一次性 `data/` 实验的静置期；旧版
`factor-run-all` 只有在其 manifest 全部为 `complete` 时才会归档。

建议每轮一次性实验使用有含义的 `--output` 名称，实验结束后再运行一次预览。需要长期
保留且准备继续断点运行的实验，应留在原路径；仅在实验结束后归档。
