# 因子策略清单

## 用途和口径

本文档整理当前仓库中已实现、已有回测结果的因子，回答三个问题：

1. 哪些因子可以继续做独立策略研究。
2. 哪些因子只适合做组合的确认项或组件。
3. 哪些因子当前不应继续跑。

当前分类来自 2026-07-10 对
`factor_report/alpha101_batch/latest/leaderboard.csv` 的逐因子复核，并固化在
`config/factors.yaml`。评估中的所有因子值都先按股票 `shift(1)`，再进入
IC、分组和净值计算。仓库不再根据综合分或分数阈值自动修改生命周期。

`active`、`watch` 和 `disabled` 都是研究生命周期，不是实盘收益承诺。
当前因子线已有计算、检验、排名信号适配器，以及可配置“过滤因子 + 排序因子”的
`factor-backtest` 两阶段组合回测。`factor-ensemble-select` 和
`factor-ensemble-backtest` 已支持显式权重的 Alpha101 横截面排名组合；投票和模型
集成仍未实现。所有组合建议仍是研究方案，不是可直接下单的交易策略。

## 当前可用层级

| YAML 状态 | 研究结论 | 因子 |
| --- | --- | --- |
| `active` | 核心保留 | `alpha_013`、`alpha_016`、`alpha_040`、`alpha_044` |
| `watch` | 组件保留 | `alpha_003`、`alpha_004`、`alpha_006`、`alpha_015`、`alpha_026`、`alpha_027`、`alpha_029`、`alpha_050`、`alpha_055`、`alpha_069`、`alpha_073`、`alpha_077`、`alpha_094` |
| `watch` | 组件观察 | `alpha_042`、`alpha_065` |
| `watch` | 低优先观察 | `alpha_014`、`alpha_018`、`alpha_019`、`alpha_024`、`alpha_037`、`alpha_052`、`alpha_058`、`alpha_059`、`alpha_061`、`alpha_067`、`alpha_074`、`alpha_075`、`alpha_076`、`alpha_080`、`alpha_081`、`alpha_087`、`alpha_088` |

其余 65 个 Alpha101 因子为 `disabled`。其中 `alpha_020` 是反向重测候选，
但原方向仍停用；如需验证，应新建显式反向因子并完整重跑，不能直接把收益取反。
`alpha_056` 缺少历史时点市值，`alpha_096` 数据覆盖不足，`alpha_097` 至
`alpha_101` 尚无完整结果，因此也保持停用。

A 股多头主口径查看 `tradable_top_quantile.csv` 和 `tradable_top_n.csv`；
`stat_long_short.csv` 与 `tradable_long_short.csv` 仅保留为统计诊断和兼容输出。

## 建议的组合方式

### 1. 单因子基线

- 只用 `alpha_040`。
- 优先检验 20 日，10 日作为次周期。
- 每日使用 T-1 日因子值做横截面排名，因子值越高排名越靠前。
- 这是后续组合比较的基准，不先叠加其他价量相关因子。

### 2. 价量反转确认组

- 主因子：`alpha_040`。
- 确认项：`alpha_013`、`alpha_016`、`alpha_044` 三选一，或使用投票而不是简单求和。
- 三个确认项都是高价/收盘价与成交量关系的变体，经济含义高度重叠；
  把它们当成三份独立 alpha 会虚增同一类暴露。
- 持有期只使用 20 日，10 日的可交易结果没有达标。

### 3. 分散化研究组

- 基准：`alpha_040`。
- 补充：`alpha_069` 和 `alpha_042` 或 `alpha_077` 中的一个。
- `alpha_069` 包含行业中性处理，比再加一个价量相关变体更有分散意义；
  但它样本外 20 日可交易结果接近持平，目前只应作小权重观察项。
- 组件应先按日做横截面 rank 或 z-score，再组合；不直接相加原始因子值。

### 4. 显式权重的排名组合

多因子组合先把每个因子按日转换为横截面百分位，再计算加权平均，不直接相加原始值。
建议先验证一个克制的分散化假设：`alpha_040` 权重 0.6，`alpha_069` 和
`alpha_077` 各 0.2。默认要求三个因子全部可用；不能因为某只股票缺少弱势因子就
无条件提高主因子权重。若研究需要放宽覆盖率，应显式设置 `--min-factor-coverage`，
输出会记录每只股票实际参与计算的因子和有效权重覆盖率。

权重、方向和覆盖阈值都必须在样本外验证前固定。该组合只是可运行的研究起点，
不是当前最优组合结论。

### 5. Alpha077 过滤 + Alpha040 排序

这是两阶段选股，不把两个因子的原始值直接相加：

1. 每日使用 T-1 日 `alpha_077` 做横截面过滤，默认保留前 50%。
2. 只在通过过滤的股票中，按 T-1 日 `alpha_040` 从高到低排序。
3. 默认选前 10 只，每日候选之间等权。
4. 股票上市样本至少 60 个交易日；如果有历史 `is_st` 则排除 ST，
   缺失时在 `filter_status.csv` 明确标记，不伪造过滤。

`alpha_077` 是价格相对 VWAP 的衰减偏离与价量相关性衰减排名两者的
较小值；用它做门槛，要求两个子条件不能有明显短板。`alpha_040` 是当前
主因子，保留排序决定权。全样本日横截面 Spearman 矩阵中，两者相关性约为
0.091，因此这个门控不是对 `alpha_040` 的重复排名。

默认 50% 是当前使用参数，不是已证明最优参数。后续应事先固定
`20% / 30% / 50%` 三档做样本外比较，不应每次根据全样本结果调到最好。

## 当前不纳入可用清单的因子

- Alpha101 中除 4 个 `active` 和 32 个 `watch` 之外的 65 个因子均为 `disabled`。
- `alpha_056`：缺少历史时点市值数据，不能伪造当前市值回填。
- `brick`：平均覆盖率约 3.1%，10/20 日统计和可交易结果不支持作为独立横截面因子。
  它仍保留在自定义买点策略线中。
- `brick_growth`：覆盖率高，但 1/5/10/20 日 Rank IC 全为负，原方向不可用。
- `momentum_20d`：当前报告中 10 日 Rank IC 为 -0.0616，20 日为 -0.0706。
  这说明样本更像中期反转，不支持按正向动量使用。如需反向使用，应新建明确的
  `reversal_20d` 因子并重跑完整的成本、约束和样本外评估，不应只在下单时临时取反。

## 常用命令

查看 Alpha101 当前研究状态：

```bash
/opt/miniconda3/envs/stocktrade/bin/python \
  scripts/quant_cli.py factor-batch --list-factor-status
```

生成 Alpha077 过滤、Alpha040 排序的最新信号：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/quant_cli.py factor-select
```

生成显式权重的多因子排名信号：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/quant_cli.py \
  factor-ensemble-select \
  --factors alpha_040 alpha_069 alpha_077 \
  --weights 0.6 0.2 0.2 \
  --min-factor-coverage 1.0 \
  --top-n 10 \
  --output data/factor_signals/ensemble_040_069_077
```

将同一组合送入 20 个固定资金槽位的组合回测：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/quant_cli.py \
  factor-ensemble-backtest \
  --factors alpha_040 alpha_069 alpha_077 \
  --weights 0.6 0.2 0.2 \
  --min-factor-coverage 1.0 \
  --top-n 10 \
  --start 2025-01-01 \
  --end 2026-06-23 \
  --hold-days 20 \
  --initial-cash 10000000 \
  --output data/portfolio_backtest_factor_ensemble_040_069_077
```

生成历史区间信号：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/quant_cli.py factor-select \
  --start 2025-01-01 \
  --end 2026-06-23 \
  --filter-top-quantile 0.5 \
  --top-n 10
```

生成信号并运行 20 个交易日持有期的组合回测：

该命令默认保留 `alpha_077` 前 80%，按 `alpha_040` 选择前 500 只，
初始资金为 1000 万元。

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/quant_cli.py factor-backtest \
  --start 2025-01-01 \
  --end 2026-06-23 \
  --hold-days 20
```

如需研究过滤后 `alpha_040` 排名的中间区间，使用从 1 开始的闭区间：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/quant_cli.py factor-backtest \
  --start 2025-01-01 \
  --end 2026-06-23 \
  --rank-start 200 \
  --rank-end 500 \
  --filter-top-quantile 0.8 \
  --output data/portfolio_backtest_alpha077_alpha040_rank200_500
```

这个示例选取第 200 至第 500 名，共 301 个候选；它不是全股票池的绝对名次，
而是 `alpha_077` 前 50% 过滤后的 `alpha_040` 名次。

信号按 `alpha_040` 分数保留优先级，引擎在信号日后一个交易日开盘尝试买入。
20 日持有期对应 20 个独立资金槽位，每日只调度其中一个；一个槽位必须在
所有到期持仓可成交卖出后才能复用。被阻止的退出不会导致新槽位累积，
因此同时活跃槽位数永远不超过 20。

组合回测会扣除佣金、印花税和过户费，并记录涨停不买、跌停不卖、停牌、
整手和单槽位现金不足。截止日尚未卖出的仓位使用当日收盘价
计入总权益，同时单独写入 `open_positions.csv`。

只复查主因子：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/quant_cli.py factor-test \
  --factor alpha_040 \
  --data data/raw \
  --metadata config/stocklist.csv \
  --windows 10 20 \
  --groups 10
```

复查当前核心因子：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/quant_cli.py factor-batch \
  --factors alpha_013 alpha_016 alpha_040 alpha_044 \
  --windows 10 20 \
  --groups 10
```

新数据复核完成后，人工更新 `config/factors.yaml` 的三档状态，并同步更新本文档。
批处理不会根据报告分数自动改写配置。
