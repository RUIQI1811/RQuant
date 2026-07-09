# RQuant

RQuant 是一个面向 A 股的本地量化研究项目。它把行情数据、因子研究、机器学习标签/模型、自定义买点、组合回测和报告输出拆成清晰的顶层模块。

本项目用于研究和决策辅助，不是自动交易系统。任何回测、因子评分或 AI 复评结果都不代表确定收益，也不应被表述为投资建议。

## 核心能力

- 使用 Tushare 抓取和维护本地日线数据。
- 计算 Alpha101、GTJA191、BrickChart 派生因子和外部 long-format 因子。
- 检验 IC、Rank IC、分组收益、中性化 IC、换手率和可交易净值。
- 生成 forward-return 标签，提供 Ridge、ElasticNet、LightGBM 和 Torch MLP 模型接口。
- 运行 B1、brick、mBDSR、BDSR/MACD/OBV 等自定义买点策略。
- 在现金、整手、仓位、费用、涨跌停、停牌和 T+1 约束下做组合回测。
- 导出候选股图表，可选调用 Gemini 做图表复评。
- 输出 CSV、JSON 和 HTML 研究报告。

## 项目结构

```text
market/      行情抓取、清洗、股票池和可交易状态
factors/     Alpha101、GTJA191、BrickChart 因子、注册表、评分和相关性
labels/      forward return 与机器学习标签
models/      Ridge、ElasticNet、LightGBM、MLP 等模型封装
training/    walk-forward 切分、验证和预测分数生成
signals/     统一信号结构和因子/模型/策略信号适配
strategies/  B1、brick、mBDSR、BDSR/MACD/OBV 等自定义买点
backtest/    组合构建、交易成本、绩效和基准比较
reports/     IC、批处理、信号收益、组合回测和研究报告
scripts/     可重复执行的命令行入口
agent/       Gemini 复评逻辑
dashboard/   看盘界面与图表导出
config/      抓取、初选、因子生命周期和复评配置
data/        本地行情和研究输出，默认不纳入 Git
```

## 研究边界

RQuant 保留三条互不混淆的研究路径：

- 因子研究：`factors/` 负责计算，`reports/factor_tester.py` 负责检验，`signals/` 负责转统一信号。
- 自定义买点：`strategies/` 保留明确规则，例如 B1、brick、mBDSR、BDSR/MACD/OBV。
- 机器学习：`labels/` 生成训练目标，`models/` 封装模型，`training/` 负责 walk-forward 和分数输出。

三条路径只通过统一信号结构进入回测：

```text
date, symbol, signal_type, source, score, weight, metadata
```

`symbol` 始终按六位字符串处理。因子值、模型分数和自定义买点不能互相反向污染。

## 快速开始

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

推荐使用已有的 stocktrade 环境运行测试和 CLI：

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/quant_cli.py --help
```

### 2. 配置密钥

需要抓取 Tushare 数据时，在项目根目录的 `.env` 或系统环境变量中设置：

```bash
TUSHARE_TOKEN=你的Tushare Token
```

需要 Gemini 图表复评时再设置：

```bash
GEMINI_API_KEY=你的Gemini API Key
```

不要把真实密钥写入代码、测试或文档。

### 3. 抓取行情

```bash
python -m market.fetch_kline
```

配置文件：

```text
config/fetch_kline.yaml
```

输出目录：

```text
data/raw/
```

每只股票一个 CSV，至少需要：

```text
date, open, close, high, low, volume
```

## 常用工作流

### 自定义买点日常流程

一键运行抓取、初选、图表导出、Gemini 复评和结果打印：

```bash
python run_all.py
```

跳过行情抓取：

```bash
python run_all.py --skip-fetch
```

只从某一步开始：

```bash
python run_all.py --start-from 3
```

### 自定义策略信号收益

```bash
python scripts/quant_cli.py signal-returns \
  --strategies bdsr_macd_obv \
  --horizons 1,5,10,20 \
  --buy-mode next_open
```

输出：

```text
data/backtest/
```

### 自定义策略组合回测

```bash
python scripts/quant_cli.py portfolio-backtest \
  --strategy bdsr_macd_obv \
  --buy-mode next_open \
  --hold-days 5 \
  --initial-cash 100000
```

输出：

```text
data/portfolio_backtest/
```

组合回测会记录成交、未成交原因、持仓、权益曲线和摘要。

### 单因子检验

```bash
python scripts/test_factor.py \
  --factor alpha_040 \
  --data data/raw \
  --metadata config/stocklist.csv \
  --windows 10 20 \
  --groups 10
```

输出：

```text
factor_report/alpha_040/
```

重点文件：

```text
summary.csv
ic.csv
rank_ic.csv
group_return.csv
stat_long_short.csv
tradable_long_short.csv
turnover.csv
neutralized_ic.csv
annual_performance.csv
sample_performance.csv
filter_status.csv
```

因子检验默认使用 `shift(1)` 后的因子值。`stat_cum_nav` 是基于 forward return 的统计诊断，`tradable_cum_nav` 才包含交易约束和费用。

### Alpha101 批处理

查看因子生命周期状态：

```bash
python scripts/test_factor_batch.py --list-factor-status
```

运行指定因子：

```bash
python scripts/test_factor_batch.py \
  --factors alpha_040 alpha_013 alpha_016 \
  --windows 10 20 \
  --groups 10
```

输出：

```text
factor_report/alpha101_batch/
```

生命周期配置：

```text
config/factors.yaml
```

### GTJA191 批处理

```bash
python scripts/test_factor_batch.py \
  --family gtja191 \
  --factors all \
  --windows 10 20 \
  --groups 10
```

输出：

```text
factor_report/gtja191_batch/
```

生命周期配置：

```text
config/gtja191_factors.yaml
```

### 因子筛选和组合回测

生成 Alpha077 过滤、Alpha040 排序信号：

```bash
python scripts/quant_cli.py factor-select \
  --start 2025-01-01 \
  --end 2026-06-23 \
  --filter-top-quantile 0.5 \
  --top-n 10
```

直接生成信号并运行组合回测：

```bash
python scripts/quant_cli.py factor-backtest \
  --start 2025-01-01 \
  --end 2026-06-23 \
  --hold-days 20
```

默认因子组合回测参数：

```text
filter-top-quantile = 0.8
top-n = 500
initial-cash = 10000000
```

### 研究报告

```bash
python scripts/quant_cli.py research-report \
  --signal-dir data/backtest \
  --portfolio-dir data/portfolio_backtest \
  --candidates data/candidates/candidates_latest.json \
  --review data/review/2026-06-23/suggestion.json \
  --output data/reports
```

输出：

```text
data/reports/research_report.json
data/reports/research_report.html
```

### 顶层 CLI 注册器

新布局的命令注册入口：

```bash
python scripts/quant_cli.py --help
```

当前它负责展示 RQuant 的顶层命令分组；已运行成熟的工作流仍通过 `python scripts/quant_cli.py ...` 和 `scripts/test_*.py` 保持兼容。

## 输出目录

```text
data/raw/                  原始日线行情
data/candidates/           初选候选列表
data/kline/<date>/          候选股票图表
data/review/<date>/         Gemini 复评结果
data/backtest*/             信号收益明细和汇总
data/portfolio_backtest*/   组合回测交易、持仓、权益曲线和摘要
data/reports/               综合研究报告
factor_report/              因子检验和批处理结果
```

`data/` 和 `factor_report/` 默认是本地研究产物，不纳入 Git。

## 验证

快速检查 CLI：

```bash
python scripts/quant_cli.py --help
python scripts/quant_cli.py --help
```

运行核心测试：

```bash
python -m unittest tests.test_cli
python -m unittest tests.test_factor_tester
python -m unittest tests.test_portfolio_backtest
python -m unittest tests.test_signal_returns
```

运行全量测试：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

当前完整迁移后的验证基线：

```text
Ran 180 tests
OK
```

## 开发原则

- 新行情逻辑放在 `market/`。
- 新因子、因子注册和相关性逻辑放在 `factors/`。
- 新标签放在 `labels/`。
- 新模型放在 `models/`。
- 新训练和 walk-forward 逻辑放在 `training/`。
- 新信号适配放在 `signals/`。
- 新自定义买点放在 `strategies/`。
- 新回测能力放在 `backtest/`。
- 新报告能力放在 `reports/`。
- 旧兼容目录已移除；新增能力应放入对应顶层模块。

涉及收益、信号、因子或组合回测的改动必须检查时间对齐、费用、交易约束、空信号、缺失数据和输出路径，并添加针对性回归测试。

## 重要限制

- RQuant 不提供收益保证。
- AI 图表复评只作为辅助筛选，不是交易指令。
- 历史回测不代表未来收益。
- 缺少历史时点字段时，系统应明确报告不可用，不能用当前字段伪造历史。
- 真实外部 API 调用不应作为普通单元测试。
