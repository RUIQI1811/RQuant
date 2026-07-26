# Polars 迁移边界

RQuant 的表格数据迁移以“不改变研究语义”为前提。新的 long-format 内部契约是
`polars.DataFrame`；CSV 读写、分组、排序、去重、横截面变换和信号筛选应优先使用
Polars 表达式。

## 已迁移

- `domain/tabular.py`：集中的 Polars 表边界、metadata CSV 序列化和原子写入。
- `labels/make_forward_return.py`：`forward_return_Nd` 和 `next_open_return_Nd`。
- `signals/schema.py`、`signals/factor_adapters.py` 和
  `signals/strategy_adapters.py`：统一信号长表。
- `training/predict_score.py`：每日分数排名、Top-N/Top-quantile 和等权信号。
- `backtest/signal_portfolio.py`：信号 CSV 读取、source/日期筛选、去重与权重契约校验。
- `training/build_dataset.py`：因子宽表进入长表后的 join、横截面 rank/z-score、日期筛选和输出。
- `factors/filter_rank.py`、`factors/ensemble.py` 和
  `training/train_walk_forward.py`：信号产物与序列化已接受 Polars 契约。

`domain.tabular.to_polars()` 只用于输入边界的过渡兼容。它可接收旧 dataframe-like
输入，但不在新业务逻辑中导入 Pandas，也不将 Polars 结果默默转回 Pandas。

## 保留的显式兼容边界

Alpha101 和 GTJA191 现有算子使用“日期索引 × 六位股票代码列”的宽表语义，
并依赖 rolling/correlation/rank 的对齐行为。这一部分暂时保留 Pandas，只在
`training.build_dataset._panel_to_long()` 转成 Polars 长表。策略预处理、组合回测和可视化中
仍使用 Pandas 索引的模块也属于后续迁移范围。

不允许为了删除 import 而用往返转换伪装迁移。宽表核心只有在以下条件都满足后
才能移除 Pandas 依赖：

1. Alpha101/GTJA191/custom 因子在同一构造数据上逐元素一致，NaN 位置一致。
2. `shift(1)`、rolling 窗口、截面 rank 和分组的边界行为有回归测试。
3. 组合回测的信号日/次日开盘时点、费用、T+1、涨跌停和严格 cohort 语义不变。
4. 输出 CSV/JSON 的字段、六位 `symbol` 和排序稳定性不变。

## 验证

```bash
python -m unittest tests.test_labels tests.test_signal_schema tests.test_factor_signals
python -m unittest tests.test_factor_filter_rank tests.test_factor_ensemble
python -m unittest tests.test_signal_portfolio_backtest tests.test_ml_dataset
python -m unittest discover -s tests -p 'test_*.py'
```
