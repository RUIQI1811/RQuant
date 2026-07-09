
# AgentTrader

一个面向 A 股的半自动选股项目：

- 使用 Tushare 拉取股票日线数据
- 用量化规则做初选（支持 B1、砖型图、mBDSR 与 BDSR/MACD/OBV 共振策略，按配置启用）
- 用信号收益与组合回测验证策略表现
- 导出候选股票 K 线图
- 调用 Gemini 对图表进行 AI 复评打分
- 生成研究报告，汇总候选、回测和复评结果

项目现在保留因子、自定义买点和机器学习三条互不混淆的研究方向：

- 因子方向：用 `factors/`、`reports/factor_tester.py` 和 `signals/` 做因子检验、因子排序信号
- 自定义买入策略方向：保留 B1、砖型图、mBDSR、BDSR/MACD/OBV 等规则买点，用 `strategies/` 适配到统一信号格式
- 机器学习方向：用 `labels/` 生成 forward return，`models/` 封装模型，`training/` 做 walk-forward 验证和分数输出

整理后的架构说明见 [docs/architecture.md](docs/architecture.md)。

---

## 更新说明

- 推翻了旧版选股模式（各式各样的B1太麻烦了）
- 新加入了AI看图打分精选功能（是的，不用再自己看图了）
- 新增 CLI 研究闭环：初选、信号收益、组合回测、研究报告都可通过 `python -m pipeline.cli ...` 运行
- 新增独立因子研究流程：单因子检验、Alpha101 批处理、综合评分与生命周期管理

---

## 1. 项目流程

项目共用同一份行情数据，但保留两条独立的研究路线。

### 1.1 共用数据准备

1. 从 Tushare 抓取日线行情：`python -m market.fetch_kline`。
2. 原始行情按股票保存到 `data/raw/`。
3. 因子研究和自定义买点都从这一层读取数据，但不共用评价逻辑。

### 1.2 路线 A：因子研究

1. 计算 `momentum_Nd`、BrickChart 派生因子、Alpha101 或外部 long-format 因子。
2. 通过 `scripts/test_factor.py` 检验 IC、Rank IC、分组收益、中性化后 IC 和样本外表现。
3. 通过 `scripts/test_alpha101_batch.py` 批量运行 Alpha101，逐因子落盘并支持断点续跑。
4. 通过 `python -m pipeline.factor_scoring --update-config` 生成综合评分，同步 `active / watch / disabled` 研究状态。
5. 按 [因子策略清单](docs/factor_strategies.md) 复查主因子、观察因子和组合组件。

因子线当前停在“因子检验、筛选和排名信号适配”阶段，尚未与
`portfolio-backtest` 完整接通为多因子纯多组合 CLI。

### 1.3 路线 B：自定义买入策略

1. `preselect` 运行 B1、brick、mBDSR、`bdsr_macd_obv` 等明确买点规则，生成候选股。
2. `signal-returns` 统计信号后 N 日收益，用于快速判断信号质量。
3. `portfolio-backtest` 在现金、整手、仓位、费用、涨跌停、停牌和 T+1 约束下回测组合。
4. 可选导出 K 线图并运行 Gemini 复评。
5. `research-report` 汇总候选、信号收益、组合回测和 AI 复评结果。

### 1.4 日常选股快捷流程

[run_all.py](run_all.py) 只编排自定义买入策略的日常选股流程，不运行因子批处理或因子评分：

1. 下载 K 线数据。
2. 量化初选。
3. 导出候选图表。
4. Gemini 复评。
5. 打印推荐结果。

输出主链路：

- data/raw：原始日线 CSV
- factor_report：因子检验、Alpha101 批处理与排行榜结果
- data/candidates：初选候选列表
- data/kline/日期：候选图表
- data/review/日期：AI 单股评分与汇总建议
- data/backtest：信号收益明细与汇总
- data/portfolio_backtest：组合回测交易、权益曲线与绩效摘要
- data/reports：研究报告 JSON/HTML

---

## 2. 目录说明

- [market](market)：行情抓取、清洗、股票池和可交易状态。
- [factors](factors)：Alpha101、GTJA191、BrickChart 派生因子、因子注册、评分和相关性。
- [labels](labels)：forward return 和机器学习标签。
- [models](models)：Ridge、ElasticNet、LightGBM、MLP 等模型封装。
- [training](training)：walk-forward 切分、验证和预测分数生成。
- [signals](signals)：统一信号结构和因子/模型/策略信号适配。
- [strategies](strategies)：B1、brick、mBDSR、BDSR/MACD/OBV 等自定义买点。
- [backtest](backtest)：组合构建、交易成本、绩效和基准比较。
- [reports](reports)：IC、分层收益、批处理、信号收益、组合回测和研究报告。
- [scripts](scripts)：可重复执行的命令行入口。
- [agent](agent)：LLM 评审逻辑（Gemini）。
- [dashboard](dashboard)：看盘界面与图表导出。
- [config](config)：抓取、初选、因子生命周期、模型和 Gemini 复评配置。
- [data](data)：本地行情、标签和研究输出。

---

## 3. 快速开始（日常选股流程）

### 3.1 Clone 项目

```bash
git clone https://github.com/SebastienZh/StockTradebyZ
cd StockTradebyZ
```

### 3.2 安装依赖

```bash
pip install -r requirements.txt
```

### 3.3 设置环境变量

Windows PowerShell（永久写入）：

```powershell
[Environment]::SetEnvironmentVariable("TUSHARE_TOKEN", "你的Tushare Token", "User")
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "你的Gemini API Key", "User")
```

写入后请重开终端，环境变量才会在新会话中生效。

### 3.4 运行日常选股脚本

在项目根目录执行：

```bash
python run_all.py
```

常用参数：

```bash
python run_all.py --skip-fetch
python run_all.py --start-from 3
```

参数说明：

- --skip-fetch：跳过数据下载，直接进入初选
- --start-from N：从第 N 步开始执行（1 到 4）

---

## 4. 完整使用教程

这一节按真实使用顺序写。你可以只跑其中一条线：

- 因子线：用于验证“某个变量是否长期有效”
- 自定义策略线：用于验证“某个明确买点是否能赚钱”

### 4.1 准备环境

```bash
pip install -r requirements.txt
```

如果要重新抓取行情，需要设置 Tushare Token：

```bash
export TUSHARE_TOKEN="你的Tushare Token"
```

Windows PowerShell：

```powershell
[Environment]::SetEnvironmentVariable("TUSHARE_TOKEN", "你的Tushare Token", "User")
```

如果只使用已有 `data/raw` 数据，可以先不设置 Token。

### 4.2 准备日线数据

方式一：从 Tushare 抓取：

```bash
python -m pipeline.fetch_kline
```

程序会请求 `start` 到 `end` 的完整区间，并用新结果覆盖本地 CSV 中同一区间的旧行；
区间外的历史数据保持不变。接口返回空数据时保留原文件。如果检测到 qfq 复权历史变化，
则仅对该股做全量刷新，避免前复权价格断层。

配置文件：

```text
config/fetch_kline.yaml
```

输出：

```text
data/raw/000001.csv
data/raw/600000.csv
...
```

每个 CSV 至少需要这些列：

```text
date, open, close, high, low, volume
```

如果数据里有 `pre_close, pct_chg, amount`，系统会在清洗和回测中优先使用这些字段。

### 4.3 路线 A：做因子测试

用途：判断一个因子有没有统计预测力。

内置示例：测试 20 日动量因子。

```bash
python scripts/test_factor.py \
  --factor momentum_20d \
  --windows 1 5 10 20 \
  --groups 10
```

输出：

```text
factor_report/momentum_20d/
  summary.csv
  coverage.csv
  distribution.csv
  ic.csv
  rank_ic.csv
  group_return.csv
  long_short.csv
  stat_long_short.csv
  tradable_long_short.csv
  turnover.csv
  exposure.csv
  neutralized_ic.csv
  annual_performance.csv
  sample_performance.csv
  universe_filter.csv
  filter_status.csv
```

重点看：

- `summary.csv`：总体摘要
- `ic_summary.csv`：IC、Rank IC、ICIR、胜率
- `group_return.csv`：分组收益是否单调
- `stat_long_short.csv`：基于 forward return 的统计净值 `stat_cum_nav`；`long_short.csv` 是为兼容旧路径保留的同内容别名
- `tradable_long_short.csv`：基于每日持仓和交易约束的 `tradable_cum_nav`
- `turnover.csv`：Top 组换手率和排名自相关
- `neutralized_ic.csv`：行业/对数市值中性化后的 IC 和 Rank IC
- `annual_performance.csv`：分年度统计与可交易绩效
- `sample_performance.csv`：时间顺序的样本内/样本外结果
- `filter_status.csv`：ST、市值等字段是否可用，避免缺字段时静默伪造过滤

因子值在所有评价中统一按股票 `shift(1)`，即 T 日选股只使用 T-1 日已完整形成的因子。`forward_return_Nd` 只用于 IC、分组收益和明确标记的 `stat_cum_nav`；统计年化收益使用 `252/(N*h)`，统计 Sharpe 使用 `sqrt(252/h)`。

`tradable_long_short.csv` 每日根据滞后因子新建一批 Top-Bottom 等权持仓，每批使用 `1/h` 资金并持有 h 个交易日。净值只使用 `daily_return`，扣除佣金、印花税和滑点，并处理停牌、涨停不买、跌停不卖及对应的空头方向约束。到期无法退出的持仓会留在队列中继续尝试，不会强制按不可成交价平仓。该净值的年化收益、回撤和 Sharpe 全部基于实际日收益序列。

测试你自己的因子文件：

```bash
python scripts/test_factor.py \
  --factor my_factor \
  --factor-file data/factors/my_factor.csv \
  --date-col date \
  --symbol-col symbol \
  --factor-col factor_value \
  --close-col close \
  --windows 1 5 10 20 \
  --groups 10 \
  --winsorize \
  --zscore
```

你的因子文件建议使用 long format：

```text
date, symbol, factor_value, close
2026-06-23, 600519, 0.83, 1688.00
2026-06-23, 000001, -0.12, 10.25
```

如果已经有未来收益列，也可以扩展为：

```text
date, symbol, factor_value, close, forward_return_1d, forward_return_5d
```

### 4.4 路线 B：跑自定义买入策略

用途：验证 B1、砖型图、mBDSR、BDSR/MACD/OBV 共振或你自己定义的买入规则。

先运行初选：

```bash
python -m pipeline.cli preselect
```

输出：

```text
data/candidates/candidates_latest.json
data/candidates/candidates_YYYY-MM-DD.json
```

调整策略开关和参数：

```text
config/rules_preselect.yaml
```

常用配置：

```yaml
b1:
  enabled: false

brick:
  enabled: true

mbdsr:
  enabled: true
  use_next_confirm: false

bdsr_macd_obv:
  enabled: true
  bdsr_fast_window: 9
  bdsr_slow_window: 26
  macd_fast_period: 12
  macd_slow_period: 26
  macd_signal_period: 9
  obv_ma_window: 20
  obv_trend_lookback: 3
```

`mBDSR_buy_signal` 要求长期 RCI 与 MA60 趋势未破坏、RCI9 从超卖区拐头、
价格靠近 MA20/MA60，同时通过缩量、OBV、ATR 和阳线止跌过滤。具体指标列和
七个子条件都保留在 `strategies/mbdsr.py` 返回的 DataFrame 中，方便排查信号。

如果要用更保守的“次日收盘突破信号日最高价”确认：

```yaml
mbdsr:
  enabled: true
  use_next_confirm: true
```

此时策略名为 `mbdsr_confirm`；普通版策略名为 `mbdsr`。回测单一版本时使用：

```bash
python -m pipeline.cli signal-returns \
  --strategies mbdsr \
  --horizons 1,5,10,20 \
  --buy-mode next_open

python -m pipeline.cli portfolio-backtest \
  --strategy mbdsr \
  --buy-mode next_open \
  --hold-days 5 \
  --initial-cash 100000
```

注意：当 `use_next_confirm: true` 时，上述命令中的策略名需改为 `mbdsr_confirm`。

`bdsr_macd_obv` 是独立的三条件同日共振策略：

- BDSR 金叉：当前项目明确定义为快线 RCI9 当日上穿慢线 RCI26。
- MACD 水上金叉：DIF 当日上穿 DEA，且 DIF、DEA 都大于 0。
- OBV 趋势向上：OBV 位于 20 日 OBV 均线上方，且均线高于 3 日前。

三个条件必须在同一根日 K 线上成立，不使用未来数据。特征列和信号列保留在
`strategies/bdsr_macd_obv.py` 返回的 DataFrame 中。由于信号需要当日收盘价和成交量，
回测建议使用下一交易日开盘买入：

```bash
python -m pipeline.cli signal-returns \
  --strategies bdsr_macd_obv \
  --horizons 1,5,10,20 \
  --buy-mode next_open

python -m pipeline.cli portfolio-backtest \
  --strategy bdsr_macd_obv \
  --buy-mode next_open \
  --hold-days 5 \
  --initial-cash 100000
```

查看候选股票代码：

```bash
python print_candidates.py
```

### 4.5 验证策略信号收益

用途：先看信号后 N 天的统计收益，不考虑真实仓位和资金占用。

```bash
python -m pipeline.cli signal-returns \
  --start 2024-10-20 \
  --end 2026-06-05 \
  --strategies brick \
  --horizons 1,5,10,20 \
  --buy-mode next_open \
  --output data/backtest_manual
```

输出：

```text
data/backtest_manual/signal_returns.csv
data/backtest_manual/signal_summary.csv
data/backtest_manual/signal_summary.json
```

同时回测多个策略时，可以显式启用基础行情预处理复用：

```bash
python -m pipeline.cli signal-returns \
  --strategies brick,mbdsr,bdsr_macd_obv \
  --reuse-base-preparation
```

该开关默认关闭；不传时仍按原方式为每个策略独立准备数据。Python 调用方可使用
`run_signal_returns(..., reuse_base_preparation=True)` 启用同一接口。

重点看：

- 信号数量是否足够
- 不同持有期平均收益
- 胜率
- 中位数收益

### 4.6 做组合回测

用途：用更接近真实交易的方式验证策略。

```bash
python -m pipeline.cli portfolio-backtest \
  --start 2024-10-20 \
  --end 2026-06-05 \
  --strategy brick \
  --buy-mode next_open \
  --hold-days 1 \
  --initial-cash 100000 \
  --commission-wan 0.8 \
  --max-positions 10 \
  --position-pct 0.1 \
  --output data/portfolio_backtest_manual
```

当前组合回测已经处理：

- 现金和持仓
- 整手买入
- 最大持仓数
- 单票目标仓位
- 手续费、印花税、过户费
- 涨停不买
- 跌停不卖
- 停牌不交易
- T+1 不当日卖出

输出：

```text
data/portfolio_backtest_manual/portfolio_summary.json
data/portfolio_backtest_manual/equity_curve.csv
data/portfolio_backtest_manual/equity_curve.html
data/portfolio_backtest_manual/portfolio_trades.csv
data/portfolio_backtest_manual/daily_trade_plan.csv
data/portfolio_backtest_manual/daily_trade_plan.json
data/portfolio_backtest_manual/open_positions.csv
```

重点看：

- `portfolio_summary.json`：总收益、最大回撤、Sharpe
- `equity_curve.html`：权益曲线
- `daily_trade_plan.csv`：每天计划买卖、是否成交、未成交原因
- `portfolio_trades.csv`：已平仓交易明细

### 4.7 导出图表和 AI 复评

导出候选 K 线图：

```bash
python dashboard/export_kline_charts.py
```

运行 Gemini 复评：

```bash
python agent/gemini_review.py
```

如果不需要 AI 看图，这一步可以跳过。

### 4.8 生成研究报告

汇总候选、信号收益、组合回测、AI 复评：

```bash
python -m pipeline.cli research-report \
  --signal-dir data/backtest_manual \
  --portfolio-dir data/portfolio_backtest_manual \
  --candidates data/candidates/candidates_latest.json \
  --review data/review/2026-06-23/suggestion.json \
  --output data/reports
```

如果没有 AI 复评结果，去掉 `--review`：

```bash
python -m pipeline.cli research-report \
  --signal-dir data/backtest_manual \
  --portfolio-dir data/portfolio_backtest_manual \
  --candidates data/candidates/candidates_latest.json \
  --output data/reports
```

输出：

```text
data/reports/research_report.html
data/reports/research_report.json
```

### 4.9 推荐的日常工作流

研究一个新想法时：

1. 如果是因子，先跑 `scripts/test_factor.py`
2. 如果是明确买点，先跑 `pipeline.cli preselect`
3. 用 `signal-returns` 看信号统计收益
4. 用 `portfolio-backtest` 看真实组合表现
5. 看 `daily_trade_plan.csv` 检查交易约束影响
6. 用 `research-report` 汇总结果

---

## 5. 分步运行攻略

### 步骤 1：拉取 K 线

```bash
python -m pipeline.fetch_kline
```

配置见 [config/fetch_kline.yaml](config/fetch_kline.yaml)：

- start、end：抓取区间
- stocklist：股票池文件
- exclude_boards：排除板块（gem、star、bj）
- out：输出目录（默认 data/raw）
- workers：并发线程数

任务结束时，日志会统计新建、增量更新、重叠日覆盖、qfq 全量刷新和失败的股票数量。

### 步骤 2：量化初选

```bash
python -m pipeline.cli preselect
```

可选参数示例：

```bash
python -m pipeline.cli preselect --date 2026-03-13
python -m pipeline.cli preselect --config config/rules_preselect.yaml --data data/raw
```

规则配置见 [config/rules_preselect.yaml](config/rules_preselect.yaml)。

### 步骤 2.1：信号收益验证

```bash
python -m pipeline.cli signal-returns \
  --start 2024-10-20 \
  --end 2026-06-05 \
  --strategies brick \
  --horizons 1,5,10,30 \
  --buy-mode signal_close \
  --output data/backtest_manual
```

输出：

- data/backtest_manual/signal_returns.csv：逐个信号的未来收益
- data/backtest_manual/signal_summary.json：均值、中位数、胜率等指标
- data/backtest_manual/signal_summary.csv：汇总 CSV

### 步骤 2.2：组合级回测

```bash
python -m pipeline.cli portfolio-backtest \
  --start 2024-10-20 \
  --end 2026-06-05 \
  --strategy brick \
  --buy-mode next_open \
  --hold-days 1 \
  --initial-cash 100000 \
  --commission-wan 0.8 \
  --max-positions 10 \
  --position-pct 0.1 \
  --output data/portfolio_backtest_manual
```

输出：

- data/portfolio_backtest_manual/portfolio_trades.csv：真实成交后的平仓交易明细
- data/portfolio_backtest_manual/daily_trade_plan.csv：每日买卖计划、成交状态与阻止原因
- data/portfolio_backtest_manual/daily_trade_plan.json：每日交易计划 JSON
- data/portfolio_backtest_manual/open_positions.csv：回测结束时仍持有的仓位
- data/portfolio_backtest_manual/equity_curve.csv：权益曲线数据
- data/portfolio_backtest_manual/equity_curve.html：权益曲线图
- data/portfolio_backtest_manual/portfolio_summary.json：收益、回撤、波动、Sharpe 等摘要

### 步骤 3：导出候选图表

```bash
python dashboard/export_kline_charts.py
```

输出到 data/kline/选股日期，图像命名为 代码_day.jpg。

### 步骤 4：Gemini 图表复评

```bash
python agent/gemini_review.py
```

可选参数示例：

```bash
python agent/gemini_review.py --config config/gemini_review.yaml
```

配置见 [config/gemini_review.yaml](config/gemini_review.yaml)。

读取候选与图表后，输出：

- data/review/日期/代码.json
- data/review/日期/suggestion.json

### 步骤 5：生成研究报告

```bash
python -m pipeline.cli research-report \
  --signal-dir data/backtest_manual \
  --portfolio-dir data/portfolio_backtest_manual \
  --candidates data/candidates/candidates_latest.json \
  --review data/review/2026-06-23/suggestion.json \
  --output data/reports
```

`--review` 是可选参数；不运行 Gemini 时也可以生成报告。

输出：

- data/reports/research_report.json
- data/reports/research_report.html

---

## 6. 关键配置建议

### 6.1 抓取层

- 首次全量抓取建议 workers 设小一些（如 4 到 8）
- 若遇到频率限制，降低并发并重试

### 6.2 初选层

- top_m 决定流动性股票池大小
- b1.enabled、brick.enabled、mbdsr.enabled、bdsr_macd_obv.enabled 控制策略开关
- mbdsr.use_next_confirm 控制使用原始信号还是次日突破确认信号
- bdsr_macd_obv 下的 BDSR、MACD 和 OBV 窗口参数控制三条件口径
- 可先只开一个策略做回放验证

### 6.3 复评层

在 [config/gemini_review.yaml](config/gemini_review.yaml) 中可调整：

- model：模型名称
- request_delay：调用间隔（防限流）
- skip_existing：是否断点续跑
- suggest_min_score：推荐分数门槛

### 6.4 回测层

- `signal-returns` 用于判断策略信号本身是否有统计优势
- `portfolio-backtest` 使用逐日组合撮合：现金、持仓、订单和权益曲线独立记账
- 买入信号默认在下一交易日执行，支持 `next_open` 或执行日收盘价
- 已处理交易成本、整手买入、最大持仓数、单票目标仓位、涨停不买、跌停不卖、停牌不交易、T+1 不当日卖出
- 当前仍未处理滑点、分红送转现金流、盘中成交队列和止盈止损

### 6.5 股票池层

`config/rules_preselect.yaml` 的 `stock_pool` 控制回测股票池过滤：

- top_m：每日按滚动成交额保留前 N 只
- min_price：最低价格门槛
- min_turnover：最低滚动成交额门槛
- exclude_boards：排除 gem/star/bj
- require_tradeable：排除停牌或不可交易日

---

## 7. 输出结果解读

### 候选文件

[data/candidates/candidates_latest.json](data/candidates/candidates_latest.json)

- pick_date：选股日期
- candidates：候选列表（含 code、strategy、close 等）

### 复评汇总

data/review/日期/suggestion.json

- recommendations：最终推荐（按分数排序）
- excluded：未达门槛代码
- min_score_threshold：推荐门槛

### 研究报告

data/reports/research_report.html

- Run Overview：候选数量、信号数量、组合收益、最大回撤、Sharpe
- Signal Returns：不同持有期的均值、中位数、胜率
- Portfolio Summary：初始资金、最终资金、交易次数等
- Review Recommendations：可选 Gemini 推荐结果

### 每日交易计划

data/portfolio_backtest_manual/daily_trade_plan.csv

- date：计划执行日期
- code：股票代码
- side：buy/sell
- status：filled/skipped/blocked
- reason：signal、hold_days、limit_up、limit_down、suspended、max_positions、insufficient_cash 等
- signal_date：信号来源日期
- price、shares、cash_delta：成交价、股数、现金变化

---

## 8. 测试

项目测试使用 Python 标准库 `unittest`：

```bash
python -m unittest discover -s tests -q
```

测试依赖项目运行依赖，至少需要先安装：

```bash
pip install -r requirements.txt
```

如果当前解释器缺少 `pandas`、`numpy` 等依赖，回测相关测试会在导入阶段失败。

---

## 9. FactorTester 因子测试

当前因子的主因子、观察因子、组合组件和停用原因见
[docs/factor_strategies.md](docs/factor_strategies.md)。该清单是研究分类，不代表可直接下单或保证收益。

当前项目结构对应关系：

- 数据读取：`strategies/preselect.py::load_raw_data` 读取 `data/raw/*.csv`
- 指标/策略计算：`strategies/selector.py` 的 `prepare_df()` 生成策略指标列，如 KDJ、知行线、砖型图、`_vec_pick`
- 信号收益评估：`reports/signal_returns.py`
- 组合回测入口：`backtest/portfolio.py` 与兼容命令 `python -m pipeline.cli portfolio-backtest`
- 当前行情格式：每只股票一个 CSV，至少包含 `date, open, close, high, low, volume`，新增抓取会尽量保留 `pre_close, change, pct_chg, amount`
- 当前因子测试标准格式：long format，默认列为 `date, symbol, factor_value, close`，未来收益列可由 `close` 计算为 `forward_return_Nd`

运行内置动量因子测试：

```bash
python scripts/test_factor.py --factor momentum_20d --windows 1 5 10 20 --groups 10
```

### 9.1 BrickChart 因子测试

BrickChart 有两个互补的因子研究入口：

- `brick`：复用 `config/rules_preselect.yaml` 的完整 `BrickChartSelector` 门控；只有 `_vec_pick=true` 的日期保留 `brick_growth`，用于检验现有策略命中后的横截面强弱。独立的流动性股票池仍由 FactorTester 的新股/流动性参数控制。
- `brick_growth`：只计算连续砖型增长倍数，不套知行线、周线多头和买点门控，适合观察 IC、Rank IC 与分组收益是否单调。

先测试完整 Brick 策略因子：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_factor.py \
  --factor brick \
  --strategy-config config/rules_preselect.yaml \
  --data data/raw \
  --windows 1 5 10 20 \
  --groups 5 \
  --winsorize
```

再测试不带策略门控的连续砖型强度：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_factor.py \
  --factor brick_growth \
  --strategy-config config/rules_preselect.yaml \
  --data data/raw \
  --windows 1 5 10 20 \
  --groups 10 \
  --winsorize \
  --zscore
```

`brick.enabled` 只控制日常初选，不会阻止显式因子测试。输出分别位于 `factor_report/brick/` 和 `factor_report/brick_growth/`。两者的原始值都先落在信号日，再由 FactorTester 统一滞后一个交易日后参与 IC、分组和净值计算；`brick` 通常较稀疏，若多数日期命中股票不足，应优先看覆盖率并使用 5 组，而不是直接解释 10 组结果。这里检验的是信号横截面统计，不替代 `signal-returns` 和 `portfolio-backtest` 的策略收益验证。

### 9.2 Alpha101 因子库

项目已实现 [WorldQuant 101 Alphas 公式页](https://ycjq.95358.com/data/dict/alpha101#alpha001) 中的 `alpha_001` 至 `alpha_101`，包括原页面因行业中性化而标注“未实现”的因子。查看完整编号：

```bash
python scripts/test_factor.py --list-factors
```

直接计算并测试某个 Alpha101 因子：

```bash
python scripts/test_factor.py \
  --factor alpha_001 \
  --data data/raw \
  --metadata pipeline/stocklist.csv \
  --windows 1 5 10 20 \
  --groups 10
```

批量回测 Alpha101（推荐使用专用批处理入口，不要用 shell 循环重复加载 101 次行情）：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_alpha101_batch.py \
  --data data/raw \
  --metadata pipeline/stocklist.csv \
  --output factor_report/alpha101_batch \
  --factors all \
  --exclude 56 \
  --windows 1 5 10 20 \
  --groups 10
```

批处理默认读取 `config/factors.yaml`。每个因子可设为三种状态：

- `active`：正常运行，优先执行并优先显示在同周期排行榜中
- `watch`：继续运行，但排在所有 `active` 因子之后
- `disabled`：默认不参与批处理

未在 `factors` 中列出的因子使用 `default_status`。例如：

```yaml
default_status: active

factors:
  alpha_001: watch
  alpha_002: disabled
```

查看当前全部状态：

```bash
python scripts/test_alpha101_batch.py --list-factor-status
```

需要临时复查已停用因子时，不必改配置；显式指定并忽略状态开关：

```bash
python scripts/test_alpha101_batch.py \
  --factors alpha_002 \
  --ignore-factor-config \
  --max-symbols 50 \
  --output /tmp/alpha002_recheck
```

`scripts/test_factor.py --factor alpha_002` 是单因子显式研究入口，不受批量状态开关影响。

当前基础 K 线没有市值，因此示例用 `--exclude 56` 跳过必须依赖市值的 `alpha_056`。补齐逐日 `cap`、`market_cap` 或 `total_mv` 后，可以去掉该参数。

批处理特性：

- 行情、Alpha101 宽面板和未来收益只构建一次，因子按顺序逐个计算，避免 101 个结果同时占用内存
- 每完成一个因子就更新 `batch_status.csv` 和 `leaderboard.csv`
- 默认自动续跑；只有数据、参数和实现代码指纹一致时才跳过已有结果
- 单因子失败会写入 `logs/因子名.log`，默认继续运行其他因子
- `--force` 强制重算；`--fail-fast` 在第一个失败因子处停止
- `--factors 1-20 30 alpha_101` 可测试指定编号或区间；`--exclude` 使用相同语法
- `--factor-config` 可指定另一份状态配置；`--ignore-factor-config` 用于一次性覆盖
- `--start-date` 与 `--end-date` 只限制评价期，仍保留更早历史作为因子预热数据
- 所有因子默认滞后 1 日；`--oos-start-date` 可指定样本外起点，否则按时间 70%/30% 切分
- `--min-listing-days`、`--min-liquidity`、`--liquidity-lookback-days` 控制新股和流动性过滤
- `--commission-rate`、`--slippage-rate`、`--stamp-tax-rate` 控制可交易多空净值的成本假设

正式全量运行前可以先做小规模冒烟测试：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_alpha101_batch.py \
  --factors 1-3 \
  --exclude 56 \
  --windows 1 5 \
  --groups 5 \
  --max-symbols 50 \
  --output /tmp/alpha101_smoke
```

批处理总览输出：

- `factor_report/alpha101_batch/batch_manifest.json`：本次数据签名、参数、因子范围和运行状态
- `factor_report/alpha101_batch/batch_status.csv`：每个因子的成功、跳过或失败状态及耗时
- `factor_report/alpha101_batch/leaderboard.csv`：按周期汇总因子状态、Rank IC、ICIR、分组收益、覆盖率和换手率
- `factor_report/alpha101_batch/alpha_001/`：单因子的完整 FactorTester 报告
- `factor_report/alpha101_batch/logs/`：失败堆栈，便于修复后直接续跑
- `factor_report/archive/`：不再参与续跑、排行榜或因子评分的历史报告；仅用于人工追溯

`leaderboard.csv` 默认先按持有周期、再按 `active/watch` 状态和 `abs_rank_icir` 排序。`direction=negative` 表示该因子应反向使用，不能仅因原始 Rank IC 为负就直接淘汰。

这个入口是横截面因子有效性回测，同时输出统计多空净值和受交易状态/成本约束的分批持仓净值。但 A 股个股空头仍假设可借券，未处理融券券源和借券费；也不替代包含现金、整手和仓位上限的 `portfolio-backtest`。批量筛选后，还应把候选因子转成统一信号，再接入 `portfolio-backtest`做纯多头实盘约束验证。

### 9.2.1 GTJA191 因子库

国泰君安 Alpha191 与现有 WorldQuant Alpha101 保持独立命名：新因子使用
`gtja_001` 到 `gtja_191`，不会覆盖 `alpha_001` 到 `alpha_101`。

测试单个因子：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_factor.py \
  --factor gtja_001 \
  --windows 1 5 10 20 \
  --groups 10
```

一条命令统一评测全部 191 个因子：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_gtja191_batch.py \
  --data data/raw \
  --factors all \
  --ignore-factor-config \
  --windows 1 5 10 20 \
  --top-counts 1 5 10 20 50 100 \
  --output factor_report/gtja191_batch
```

如果前面一部分因子已经跑过，可以从指定编号继续请求后续因子：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_gtja191_batch.py \
  --data data/raw \
  --factors all \
  --ignore-factor-config \
  --start-factor 037 \
  --windows 1 5 10 20 \
  --top-counts 1 5 10 20 50 100 \
  --output factor_report/gtja191_batch
```

`--start-factor` 接受 `37`、`037` 或 `gtja_037`，并在当前 `--factors` /
`--exclude` / 生命周期过滤结果中只保留该因子及之后的因子。数据、参数和实现
指纹一致时，已有单因子报告仍会自动跳过；最终 `leaderboard.csv` 会重新扫描
同一输出目录下所有已完成且指纹匹配的 GTJA191 报告，把之前跑过的因子也收进总表。
`--top-counts` 是 A 股纯多头筛选口径，默认输出 Top1、Top5、Top10、Top20、
Top50 和 Top100，不需要再依赖 10 组分组或 Top-Bottom 多空收益判断。
GTJA191 批处理默认显示因子级进度条和当前因子；如果要把输出重定向到日志文件，
可加 `--no-progress` 关闭进度条。

`gtja_030` 需要 `date,mkt,smb,hml` 文件；`gtja_075`、`gtja_149`、
`gtja_181`、`gtja_182` 需要基准指数 `date,open,close` 文件，分别通过
`--style-factor-file` 和 `--benchmark-file` 提供。缺失时批处理将对应因子
记为 `missing_input`，不会用股票池均值伪造市场数据，也不会中断其他因子。

输出位于 `factor_report/gtja191_batch/`：

- `batch_status.csv`：每个请求因子的最终状态，完整运行应有 191 行。
- `leaderboard.csv`：所有已完成且本次指纹匹配的 GTJA191 因子表现总表，包含 Rank IC、ICIR、Top1/5/10/20/50/100 纯多头收益、Sharpe、覆盖率和换手率等字段。
- `top_n_summary.csv` / `top_n_return.csv`：单因子的 TopN 纯多头统计汇总和逐日明细。
- `gtja_XXX/`：单因子的完整 FactorTester 报告。
- `logs/gtja_XXX.log`：失败堆栈；修复后可直接断点续跑。

所有 GTJA 因子仍由 FactorTester 统一滞后一个交易日。统计净值与可交易
净值保持分离，结果仅用于研究，不构成确定收益或自动交易指令。

### 9.3 因子相关性矩阵

用于识别排序信号高度重叠的 Alpha101 因子。默认读取
`config/factors.yaml`，只计算 `active` 和 `watch`（包括映射为 `watch` 的
`component_only`）因子：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/factor_correlation.py \
  --data data/raw \
  --metadata pipeline/stocklist.csv \
  --output factor_report/factor_correlation
```

也可显式限定因子和时间区间：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/factor_correlation.py \
  --factors alpha_040 alpha_013 alpha_016 alpha_044 alpha_069 \
  --start-date 2022-01-01 \
  --high-correlation-threshold 0.8
```

计算口径与因子评价一致：每个因子先按股票滞后 1 个交易日，然后在每个
交易日内对股票横截面计算相关系数，最后对有效交易日取平均。默认主矩阵是
Spearman，因为它直接衡量横截面排名是否重叠；Pearson 作为原始值线性相关的
辅助诊断。默认每日每对因子至少要有 20 只共同有效股票，且至少要有 20 个
有效交易日才输出相关系数。

输出：

- `spearman_matrix.csv`：平均日横截面等级相关矩阵
- `pearson_matrix.csv`：平均日横截面线性相关矩阵
- `valid_date_count_matrix.csv`：每对因子实际参与平均的交易日数
- `correlation_pairs.csv`：按 `abs_spearman` 从高到低排列的因子对，并标记是否超过阈值
- `spearman_heatmap.html`：可交互热力图
- `factor_status.csv` 和 `manifest.json`：计算失败、参数和数据期间的审计信息

该工具会从原始行情重算因子值，不会用单因子 `summary.csv` 伪造相关性。
如需临时纳入已停用因子，可显式指定 `--factors` 并加
`--ignore-factor-config`。

### 9.4 Alpha077 过滤、Alpha040 排序

这个两阶段因子选股器先用 `alpha_077` 缩小股票池，再用
`alpha_040` 决定最终排名。默认参数为：

- 所有因子值先滞后 1 个交易日，T 日信号只使用 T-1 日信息
- `alpha_077` 从高到低保留前 50%
- 在通过的股票中按 `alpha_040` 从高到低取前 10 只
- 每日入选股票等权，上市样本不少于 60 个交易日
- 有历史 `is_st` 则排除 ST；字段缺失时显式报告，不伪造过滤

生成最新交易日的候选：

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m pipeline.cli factor-select
```

回放一段历史信号：

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m pipeline.cli factor-select \
  --start 2025-01-01 \
  --end 2026-06-23 \
  --filter-top-quantile 0.5 \
  --top-n 10 \
  --output data/factor_signals/alpha077_alpha040_history
```

如果不选排名最靠前的股票，而是选过滤后 `alpha_040` 排名第 200 至第 500 名
（包含 200 和 500）：

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m pipeline.cli factor-select \
  --rank-start 200 \
  --rank-end 500 \
  --output data/factor_signals/alpha077_alpha040_rank200_500
```

`--rank-start` / `--rank-end` 是从 1 开始的闭区间，只对通过 `alpha_077` 过滤后的
`alpha_040` 排名生效。显式指定 `--rank-end` 时，它会代替 `--top-n` 决定截止名次。

输出位于 `data/factor_signals/alpha077_alpha040/`：

- `signals.csv` / `signals.json`：符合统一信号 schema 的买入信号
- `selections.csv`：因子原始值、横截面分位、排名和权重
- `daily_summary.csv`：每日完整因子数、可用股票数、过滤数和入选数
- `filter_status.csv`：上市时间、流动性和 ST 字段是否真实执行
- `manifest.json`：参数和输出路径

对这组信号运行考虑现金、整手、仓位、费用、涨跌停、停牌和 T+1 的组合回测：

`factor-backtest` 默认用 `alpha_077` 保留前 80%，再按 `alpha_040`
选前 500 只，初始资金为 1000 万元。需要其他口径时显式传入
`--filter-top-quantile`、`--top-n` 和 `--initial-cash`。

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m pipeline.cli factor-backtest \
  --start 2025-01-01 \
  --end 2026-06-23 \
  --hold-days 20 \
  --initial-cash 10000000 \
  --commission-wan 0.8 \
  --output data/portfolio_backtest_alpha077_alpha040
```

回测排名第 200 至第 500 名的区间：

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m pipeline.cli factor-backtest \
  --start 2025-01-01 \
  --end 2026-06-23 \
  --rank-start 200 \
  --rank-end 500 \
  --output data/portfolio_backtest_alpha077_alpha040_rank200_500
```

回测使用滞后因子生成信号，再于信号日后一个交易日开盘尝试成交。
前置的行情读取、因子计算和信号生成期间会显示“正在准备数据”；
真正进入组合回测后，按交易日显示动态进度条。需要把输出重定向到日志时，
可加 `--no-progress` 关闭提示和进度条。
默认持有 20 个交易日，同时把初始资金严格分为 20 个独立槽位：每个交易日
只调度一个槽位，到期后先卖出该槽位内的持仓，再按当日 `alpha_040` 排名顺序
买入新候选。如果跌停或停牌导致某槽位未完全卖出，该槽位不得新建持仓，
也不得额外创建第 21 个槽位。

每个槽位只能使用自己的现金，并在可买的当日候选中尽量做等权整手买入。
默认初始资金为 1000 万元，每个槽位初始分得 50 万元。每个槽位尽量分散到
10 只股票；若单股买不起 100 股，则记为
`insufficient_sleeve_cash`。
输出位于 `data/portfolio_backtest_alpha077_alpha040/`：

- `equity_curve.html` / `equity_curve.csv`：每日权益曲线
- `portfolio_trades.csv`：已平仓交易和单笔收益
- `portfolio_summary.json`：总收益、回撤、波动率、Sharpe 和持仓数
- `daily_trade_plan.csv` / `daily_trade_plan.json`：成交、被阻止和跳过原因
- `open_positions.csv`：截止日未平仓持仓
- 交易、订单和未平仓文件都包含 `cohort_id`，可追踪每一份资金
- `factor_signals/`：本次回测实际使用的历史因子信号及筛选证据

详细设计和后续样本外参数比较见 `docs/factor_strategies.md`。

### 9.5 Alpha101 综合评分

批量报告完成后，可以从每个 `alpha_*/summary.csv` 生成综合评分：

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m pipeline.factor_scoring --update-config
```

评分结果默认直接同步到唯一配置文件：

```text
config/factors.yaml
```

每个因子条目保存 `final_score, decision, useful_horizons`；批处理直接从同一个条目的 decision 映射出 `active/watch/disabled`，不再维护重复的 `factor_scores.csv`。评分规则的关键边界：

- signal、tradability、分组收益和单调性只组合 20d 与 10d，权重分别为 70% 和 30%
- 1d、5d 只参与 5 分的 horizon consistency，不参与主评分、惩罚或硬降档
- 三个正向类别原始满分合计 90，先按 `subtotal / 90 * 100` 换算为百分制，再扣最多 20 分
- 单调性已经通过正向得分体现，不再重复扣 penalty
- collapse ratio 只在 20d 统计 Sharpe 为正且相关数据完整时计算
- 只有 10d 和 20d 的可交易 Sharpe 同时为负才直接判为 `disabled`；仅短周期 1d/5d 表现差不会触发该规则
- `useful_horizons` 只检查 10d/20d；对应窗口需同时满足 Rank IC 与 Rank ICIR 为正、可交易 Sharpe 不低于 0.3、可交易年化收益不低于 3%

如需临时导出完整评分明细，可以显式指定 CSV；该文件只是快照，不是配置数据源：

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m pipeline.factor_scoring \
  --output /tmp/factor_scores.csv
```

同步时，评分为 `active` 的因子按 active 运行，`watch/component_only/low_priority_watch` 映射为 watch 以保留后续复查，`disabled` 以及尚无完整报告的因子映射为 disabled。YAML 顶层的 `factor_scoring` 只保存评分规则和来源，不再重复保存逐因子结果。

该评分按正向 Rank IC 和正向多空收益计分。对 `leaderboard.csv` 中 `direction=negative` 的因子，应先使用独立的样本内规则确定是否反向，再重新生成反向因子报告；评分模块不会用全样本结果自动翻转方向，以免引入数据窥探。

实现入口：

- `factors/alpha101.py`：公共算子、101 个公式、宽面板构建及 long-format 转换
- `factors/brick.py`：把完整 Brick 策略门控或连续砖型强度转换为 long-format 因子
- `reports.factor_tester.build_long_factor_frame_from_raw(...)`：把 `brick`、`brick_growth`、`alpha_001` 等名称接入现有 FactorTester
- `factors.alpha101.Alpha101`：代码调用入口，可按编号计算单个或多个因子

数据约定与限制：

- 必需列：`date, open, close, high, low, volume`
- `vwap`：优先使用原始 `vwap` 或 `avg`；缺失时使用 `(high + low + close) / 3` 作为代理值
- `advN`：按输入 `volume` 的 N 日均值计算，保持与公式中的 `volume` 同单位
- 行业中性化：优先读取 metadata 中的 `sector, industry, subindustry`；当前 `stocklist.csv` 只有 `industry` 时，会同时作为三级分类的回退值
- `alpha_056`：必须额外提供逐日 `cap`、`market_cap` 或 `total_mv`，当前基础 K 线没有市值时会明确报错，不会使用价格或成交额伪造市值
- 公式中的小数回看窗口按最近整数交易日处理；比较型公式按页面说明输出 `1/-1`
- Alpha101 是横截面因子，必须同时加载一个股票池；只给单只股票时，`rank` 和行业中性化结果没有研究意义

运行已有 long format 因子文件：

```bash
python scripts/test_factor.py \
  --factor my_factor \
  --factor-file data/factors/my_factor.csv \
  --date-col date \
  --symbol-col symbol \
  --factor-col factor_value \
  --close-col close \
  --windows 1 5 10 20 \
  --groups 10 \
  --winsorize \
  --zscore
```

输出目录：

- factor_report/因子名/summary.csv
- factor_report/因子名/coverage.csv
- factor_report/因子名/distribution.csv
- factor_report/因子名/ic.csv
- factor_report/因子名/rank_ic.csv
- factor_report/因子名/group_return.csv
- factor_report/因子名/long_short.csv
- factor_report/因子名/stat_long_short.csv
- factor_report/因子名/tradable_long_short.csv
- factor_report/因子名/turnover.csv
- factor_report/因子名/exposure.csv
- factor_report/因子名/neutralized_ic.csv
- factor_report/因子名/neutralized_ic_summary.csv
- factor_report/因子名/annual_performance.csv
- factor_report/因子名/sample_performance.csv
- factor_report/因子名/universe_filter.csv
- factor_report/因子名/filter_status.csv

FactorTester 支持覆盖率、分布、去极值、标准化、IC/Rank IC、中性化 IC、分组收益、统计多空净值、可交易持仓净值、Top 组换手、排名自相关、年度绩效和样本外报告。默认按时间前 70%/后 30% 分成样本内/样本外，可用 `--oos-start-date` 指定分界日。

默认单边佣金率为 `0.0003`、滑点为 `0.0005`、卖出印花税为 `0.0005`，并排除上市不足 60 个有效交易日的新股。低流动性门槛默认为 0（不启用），可用 `--min-liquidity` 设置过去 20 日平均成交额门槛。当基础数据缺少历史 `is_st` 或逐日市值时，程序会在 `filter_status.csv`/`exposure.csv` 明确标记缺失，不会用当前名称回填历史 ST，也不会用价格或成交额伪造市值。

`pipeline/stocklist.csv` 目前只是静态行业快照；在没有逐日行业归属时，历史暴露和中性化 IC 会使用这份静态分类。如果要严格消除行业变更的前视偏差，需要在因子 long-format 数据中提供逐日 `industry`。

---

## 10. 常见问题

### Q1：fetch_kline 报 token 错误

- 检查 TUSHARE_TOKEN 是否已设置
- 确认 token 有效且账号权限正常

### Q2：导出图表时报 write_image 错误

- 确认已安装 kaleido
- 重新安装：pip install -U kaleido

### Q3：Gemini 运行失败

- 检查 GEMINI_API_KEY 是否设置
- 观察是否命中限流，可提高 request_delay

### Q4：没有候选股票

- 检查 data/raw 是否有最新数据
- 放宽初选阈值（如 B1 或 Brick 参数）
- 检查 pick_date 是否在有效交易日

---
