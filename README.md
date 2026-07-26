# RQuant

RQuant 是一个面向 A 股的本地量化研究项目。它把行情数据、因子研究、机器学习标签/模型、自定义买点、组合回测和报告输出拆成清晰的顶层模块。

本项目用于研究和决策辅助，不是自动交易系统。任何回测、因子检验或 AI 复评结果都不代表确定收益，也不应被表述为投资建议。

## 核心能力

- 使用 Tushare 抓取和维护本地日线数据。
- 计算 Alpha101、GTJA191、BrickChart 派生因子和外部 long-format 因子。
- 检验 IC、Rank IC、分组收益、中性化 IC、换手率和可交易净值。
- 将多个 Alpha101 因子的日横截面百分位按显式权重组合为可审计信号。
- 生成 forward-return 标签，提供 Ridge、ElasticNet、Qlib LightGBM、Qlib DoubleEnsemble 和 Torch MLP 模型接口。
- 运行 B1、brick、mBDSR、BDSR/MACD/OBV 等自定义买点策略。
- 在现金、整手、仓位、费用、涨跌停、停牌和 T+1 约束下做组合回测。
- 导出候选股图表，可选调用 Gemini 做图表复评。
- 输出 CSV、JSON 和 HTML 研究报告。
- 在 long-format 数据管线使用 Polars 做列式计算和 CSV I/O。

## 项目结构

```text
market/      行情抓取、清洗、股票池和可交易状态
domain/      跨研究路径的值对象、信号、执行结果和工作流产物契约
factors/     Alpha101、GTJA191、BrickChart 因子、注册表、生命周期和相关性
labels/      forward return 与机器学习标签
models/      Ridge、ElasticNet、Qlib LightGBM/DoubleEnsemble、MLP 等模型封装
training/    walk-forward 切分、Qlib Dataset 适配、验证和预测分数生成
signals/     统一信号结构和因子/模型/策略信号适配
strategies/  B1、brick、mBDSR、BDSR/MACD/OBV 等自定义买点
backtest/    组合构建、交易成本、绩效和基准比较
reports/     IC、批处理、信号收益、组合回测和研究报告
scripts/     可重复执行的命令行入口
rquant/      CLI、项目路径、运行日志和审计清单等工程框架层
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
跨模块对象的完整类型、兼容策略和数据流见 `docs/domain_model.md`。统一信号进入组合层后
会保留 `source`、`score`、`weight` 和 `metadata`，不再降级成只有股票代码的匿名候选。

Long-format 表的新内部契约是 `polars.DataFrame`。统一信号、forward-return
标签、模型分数转信号、信号回测入口和 ML 数据集的长表拼接/变换已使用
Polars。Alpha101/GTJA191 现有的“日期索引 × 股票列”宽表公式仍是显式兼容
边界，以保留因子数学语义和时间对齐。迁移边界和后续阶段见
[`docs/polars_migration.md`](docs/polars_migration.md)。

## 快速开始

### Windows 与 macOS

项目默认使用 Conda 管理 Python 3.11 环境。请先在项目根目录创建并激活 `stocktrade`
环境，再执行本文其余的 `python ...` 命令。

Windows PowerShell：

```powershell
conda create -n stocktrade python=3.11 -y
conda activate stocktrade
python -m pip install --upgrade pip
```

macOS（zsh/bash）：

```bash
conda create -n stocktrade python=3.11 -y
conda activate stocktrade
python -m pip install --upgrade pip
```

若已创建过 `stocktrade` 环境，后续只需执行 `conda activate stocktrade`，不必重复创建。首次使用
Windows PowerShell 或 macOS 终端时若找不到 `conda`，请分别运行 `conda init powershell` 或
`conda init zsh`，重开终端后再执行上述命令。所有路径参数都使用正斜杠（例如 `data/raw`），
可同时用于 Windows 和 macOS。

后文所有命令均为单行写法，可直接粘贴到 Windows PowerShell、macOS 的 zsh 或 bash；无需
替换续行符。

### 0. 统一工程入口

仓库的正式入口是：

```bash
python -m rquant --help
```

以可编辑模式安装后，也可以直接使用 `rquant`：

```bash
python -m pip install -e .
rquant doctor
```

`python -m rquant ...` 是正式入口。`python scripts/quant_cli.py ...` 仅保留为迁移期
兼容入口，执行时会显示弃用提示。研究业务仍由原来的 `market/`、`factors/`、`training/`、`strategies/`
和 `backtest/` 模块承担；`rquant/` 只负责入口、路径、日志和运行治理。

所有正式命令支持以下全局参数，参数可放在子命令前后：

```text
--project-root PATH
--runs-dir PATH
--run-id TEXT
--log-level DEBUG|INFO|WARNING|ERROR
```

相对路径统一相对项目根目录解析，因此可以从仓库外通过 `--project-root` 启动研究任务。
未显式指定时，运行 ID 由 UTC 时间和随机短标识生成。

每次真实运行会创建：

```text
data/runs/<run-id>/run.json
data/runs/<run-id>/run.log
```

`run.json` 记录命令、脱敏参数、解释器、Git commit、工作区状态、输入指纹、输出路径、
下游 manifest、warning 和最终退出状态。它不记录 `.env` 内容、Token、API Key 或完整
环境变量。业务失败和 Ctrl-C 中断也会先原子完成清单，再保持非零退出码。

查看最近运行或某次完整清单：

```bash
rquant runs list --limit 20
rquant runs list --status failed
rquant runs show <run-id>
```

`--help` 和 `runs list/show` 是只读框架操作，不创建新的运行记录。运行索引只引用已有
研究产物，不会混合比较因子 IC、ML 预测指标和可交易组合收益。

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

核心依赖和可选 ML 后端均按已验证版本锁定；`doctor` 会同时检查包是否真实可导入以及
固定版本是否匹配，避免“已安装但原生库缺失”或依赖升级漂移。
Polars 锁定为 `1.43.0`；Pandas 在宽表因子公式完成迁移前仍是明确的过渡依赖，
不应在新的 long-format 业务模块中新增 Pandas API。

激活环境后，用当前环境的解释器运行测试和 CLI：

```bash
python -m rquant --help
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
python -m rquant fetch-data --config config/fetch_kline.yaml
```

配置文件：

```text
config/fetch_kline.yaml
```

输出目录：

```text
data/raw/
```

抓取过程会在输出目录持续原子更新检查点：

```text
data/raw/_fetch_manifest.json
```

manifest 保存区间、股票池签名、逐股 outcome、失败原因、重试次数以及待处理代码。
任一股票三次重试后仍失败时，已完成 CSV 会保留，但 manifest 状态为 `partial`，
`fetch-data` 返回非零退出码。修复外部服务后可仅重试失败/未完成代码：

```bash
python -m rquant fetch-data --config config/fetch_kline.yaml --resume
```

只有日期区间、输出目录和股票池签名完全一致时才允许恢复，防止把另一批抓取结果混入。

`config/fetch_kline.yaml` 默认保留 8 个工作线程，同时用
`max_requests_per_minute: 180` 对所有 `ts.pro_bar` 请求做全局均匀节流，
为 Tushare 常见的 200 次/分钟限额留出余量。工作线程仍可覆盖，节流也可临时调整：

```bash
python -m rquant fetch-data --workers 8 --max-requests-per-minute 180
```

`--max-requests-per-minute 0` 会显式关闭主动节流，只建议在已确认账号限额或外部统一限流时使用。
manifest 同时记录实际线程数和节流值，便于复现运行条件。

每只股票一个 CSV，至少需要：

```text
date, open, close, high, low, volume
```

历史市值分档使用独立的 Tushare `daily_basic` 时点上下文，不把今天的市值回填到历史：

```bash
python -m rquant fetch-context --start 20180101 --end 20260710 --out data/context/daily_basic
```

抓取器先读取交易日历，再按开市日期一次提取全市场，逐日原子写入
`data/context/daily_basic/YYYY/YYYYMMDD.csv`。Tushare 返回的 `total_mv` 和
`circ_mv` 单位为万元，落盘时分别转换为以元计的 `market_cap` 和
`circulating_market_cap`；同时保存当日 `pb` 和正 PB 对应的 `book_to_market=1/pb`，
为以后构造有明确定义的价值风格序列保留时点输入。检查点位于
`_context_manifest.json`；部分失败会返回非零，
且在使用 `--resume` 补齐前，因子研究会拒绝读取该目录。

```bash
python -m rquant fetch-context --start 20180101 --end 20260710 --out data/context/daily_basic --resume
```

该接口需要 Tushare 对 `daily_basic` 的访问权限；官方说明单次最多返回 6000 行，适合
按交易日循环全市场。权限和积分要求以
[Tushare 每日指标文档](https://tushare.pro/document/2?doc_id=32)为准。抓取完成后可直接：

```bash
python -m rquant factor-batch --family gtja191 --context-file data/context/daily_basic --data data/raw --factors all --windows 1 5 10 20 --output factor_report/gtja191_batch/latest
```

GTJA075、149、181、182 需要真实大盘指数开收盘序列，可单独抓取沪深300并直接作为
`--benchmark-file`：

```bash
python -m rquant fetch-benchmark --index-code 000300.SH --start 20180101 --end 20260710 --out data/context/benchmark_000300.csv
```

该入口调用 Tushare `index_daily`，完整区间原子落盘并维护同名 manifest；
`--resume` 只复用指数代码、日期范围和字段完全一致的完成文件。接口字段与权限要求见
[Tushare 指数日线文档](https://tushare.pro/document/1?doc_id=95)。

`gtja_030` 的 MKT/SMB/HML 使用独立离线构造步骤。它以复权收盘收益为当日股票收益，
市值和 `1/PB` 必须严格来自该日之前最新可得的 `daily_basic`，每天按市值中位数和
账面市值比 30%/70% 分位形成六个价值加权组合：

```bash
python -m rquant build-style-factors --data data/raw --context data/context/daily_basic --start 2018-01-01 --end 2026-07-10 --min-stocks-per-portfolio 5 --out data/context/style_factors.csv
```

输出 `date,mkt,smb,hml` 及同名 manifest；其中 MKT 是滞后市值加权的全市场原始收益，
SMB=`mean(SL,SM,SH)-mean(BL,BM,BH)`，HML=`mean(SH,BH)-mean(SL,BL)`。
构造口径、输入签名、样本数和因组合不完整而剔除的日期均写入 manifest，不会把指数
收益改名成 MKT/SMB/HML。

完整 GTJA191 批量输入因此为：

```bash
python -m rquant factor-batch --family gtja191 --data data/raw --context-file data/context/daily_basic --benchmark-file data/context/benchmark_000300.csv --style-factor-file data/context/style_factors.csv --factors all --ignore-factor-config --windows 1 5 10 20 --output factor_report/gtja191_batch/latest
```

### 4. 系统自检

在抓取、因子批处理或模型训练前，可先运行只读自检：

```bash
python -m rquant doctor --output data/reports/system_doctor.json
```

默认检查当前 Python 解释器、`requirements.txt` 必需依赖、ML 可选依赖、核心 YAML、
`stocklist.csv` 六位代码约束、密钥是否已设置，以及前 25 个行情 CSV 的字段和日期。
它只记录 `TUSHARE_TOKEN` / `GEMINI_API_KEY` 是否存在及来源，绝不输出密钥值。
本地最新行情距当前日期超过 7 个自然日时会告警，可用 `--max-data-age-days N` 调整。
检查全部行情文件时显式使用：

```bash
python -m rquant doctor --deep
```

必需依赖、核心配置或已存在行情文件的结构错误会返回非零退出码；未安装 Torch、
pyqlib、scikit-learn，尚未配置外部服务密钥或尚无本地行情只产生警告。

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

只执行到本地图表导出，不调用 Gemini：

```bash
python run_all.py --stop-after 3
```

显式跳过 Gemini 复评时，步骤 4 和步骤 5 都会跳过，避免展示旧的复评文件：

```bash
python run_all.py --skip-review
```

`--start-from` / `--stop-after` 只接受 1~5。任何子步骤非零退出，或最终复评 JSON
缺失、损坏、字段类型不正确，都会立即停止并返回非零退出码。终端展示的是达到 AI
复评阈值的研究候选，不是买入建议或自动交易指令。

Gemini 复评默认按签名恢复：只有模型、提示词、股票代码、复评日期和图表内容均匹配的
逐股 JSON 才会复用；损坏或过期结果自动重算。每只股票原子落盘，并持续更新：

```text
data/review/<date>/<symbol>.json
data/review/<date>/run_manifest.json
data/review/<date>/suggestion.json
```

部分股票失败时仍保留已完成结果和 `partial` manifest，但命令返回非零，避免下游把
部分结果当作完整结果。强制重算会先把旧结果原子归档到 `.stale/`；即使新调用失败，
后续恢复也不会误用旧结果。恢复或强制重算可显式执行：

```bash
python agent/gemini_review.py --resume
python agent/gemini_review.py --force
```

即使直接运行 Gemini 脚本，也必须先存在状态为 `complete` 的图表导出 manifest；复评前
会逐股校验 `chart_end_date == pick_date` 及图片 SHA-256，旧图或被修改的图不会进入模型。

候选图表本身也严格遵守点时边界：导出器先把每只股票截断到 `pick_date`，并要求该日
bar 确实存在，绝不把候选日之后的 K 线交给 Gemini。导出检查点位于：

```text
data/kline/<date>/export_manifest.json
```

图表使用临时文件完成后再原子替换；缺行情、缺候选日 bar 或渲染失败都会形成
`partial` manifest 并返回非零。可安全恢复：

```bash
python dashboard/export_kline_charts.py --resume
```

### 自定义策略信号收益

```bash
python -m rquant signal-returns --strategies bdsr_macd_obv --horizons 1,5,10,20 --buy-mode next_open
```

输出：

```text
data/backtest/
```

### 自定义策略组合回测

```bash
python -m rquant portfolio-backtest --strategy bdsr_macd_obv --buy-mode next_open --hold-days 5 --initial-cash 100000
```

输出：

```text
data/portfolio_backtest/
```

组合回测会记录成交、未成交原因、持仓、权益曲线和摘要。

### 因子研究一键 run-all

`factor-run-all` 把同一因子库的批量检验、分类汇总、相关性去重、ML
walk-forward 和模型组合回测串成一个可恢复工作流。默认配置：

```text
config/factor_research_run_all.yaml
```

直接运行配置中选择的因子库：

```bash
python -m rquant factor-run-all --config config/factor_research_run_all.yaml
```

不改 YAML 也可以在命令行临时选择内置库：

```bash
python -m rquant factor-run-all --config config/factor_research_run_all.yaml --family gtja191 --factor-config config/gtja191_factors.yaml --output factor_report/factor_run_all/gtja191
```

收到新的宽表或长表因子库后，使用外部入口：

```bash
python -m rquant factor-run-all --config config/factor_research_run_all.yaml --family external --factor-file data/factors/my_factors.csv --factor-config config/my_factors.yaml --output factor_report/factor_run_all/my_factors
```

外部库的 `factor_config` 必须给每个因子填写研究分类，例如
`price_behavior`、`price_volume`、`traditional_technical`、`market_related`、
`liquidity` 或其他有明确含义的类别。若存在未分类因子，入口会在长任务开始前停止，
并在输出目录生成 `factor_classification_template.yaml`；补全后再运行即可。

流水线固定执行：

1. 对配置中的全部周期生成 IC/Rank IC、方向、高低值两侧只做多毛收益与净收益、
   扣费前后夏普、逐年收益、市值分档 IC、行业/板块 IC 和牛熊震荡 IC。
2. 用批量排行榜中的同周期扣费后净夏普作为质量优先级，计算每日横截面
   Spearman/Pearson，两两 `|Spearman| >= 0.8` 聚类去重。
3. 只把去重后、在指定周期扣费后仍盈利的因子写入 `ml_candidate_factors.csv`。
4. 严格使用过去三个完整日历年训练、预测下一个完整日历年；每个模型分别运行
   零成本和实际成本的 A 股约束组合回测。

这里“高值侧”和“低值侧”都表示买入该侧股票；不会卖空任何股票。统计型多空文件
仍可作为诊断，但不会进入 ML 或组合执行。若没有任何去重因子在指定周期扣费后盈利，
工作流会明确记录 `skipped_no_profitable_candidates` 并跳过 ML，而不是把无合格证据的
因子静默送入模型。

默认输出结构：

```text
factor_report/factor_run_all/alpha101/manifest.json
factor_report/factor_run_all/alpha101/summary.json
factor_report/factor_run_all/alpha101/batch/
factor_report/factor_run_all/alpha101/correlation/
factor_report/factor_run_all/alpha101/ml/
```

各阶段沿用原有断点和签名；只有确认需要重算时追加 `--force`。只检查因子研究和去重、
暂不启动 ML 时使用 `--skip-ml`。该命令不自动抓取行情或历史市值；应先完成
`fetch-data` / `fetch-context`，并保证上下文 manifest 为 `complete`。

### 单因子检验

```bash
python -m rquant factor-test --factor alpha_040 --data data/raw --metadata config/stocklist.csv --windows 10 20 --groups 10
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
tradable_top_n.csv
tradable_top_quantile.csv
turnover.csv
neutralized_ic.csv
annual_performance.csv
annual_ic.csv
annual_long_only.csv
market_cap_ic.csv
market_cap_ic_summary.csv
industry_ic.csv
industry_ic_summary.csv
market_regime_ic.csv
market_regime_ic_summary.csv
horizon_effectiveness.csv
tradable_bottom_n.csv
tradable_bottom_quantile.csv
sample_performance.csv
filter_status.csv
```

因子检验默认使用 `shift(1)` 后的因子值。A 股只能做多时，主评价应优先看
`tradable_top_n.csv` 和 `tradable_top_quantile.csv` 的 long-only 结果；
`stat_cum_nav` 和 `tradable_cum_nav` 保留为多空诊断与兼容输出，不作为主交易口径。

新增的分层报告不会替代原始 IC：`market_cap_ic*` 使用每日、滞后一日的市值横截面
分成小/中/大三档后分别计算 IC；`industry_ic*` 在每个行业内部计算 IC；
`market_regime_ic*` 按牛市、熊市和震荡市汇总每日 IC。若输入已有
`market_regime` 列则直接使用；否则用全市场等权日收益构造 60 日累计收益代理，严格
滞后一日后按 `>=10%`、`<=-10%`、其余分别定义牛、熊、震荡。窗口、阈值和市值档数
都可通过 `--market-regime-lookback-days`、`--bull-return-threshold`、
`--bear-return-threshold` 和 `--market-cap-groups` 调整。

`tradable_top_*` 表示买入因子值最高的一侧，`tradable_bottom_*` 表示买入因子值最低的
一侧；两者都只做多。各文件同时保留 `gross_*`（不扣成本）和 `net_*`（扣成本）指标。
`horizon_effectiveness.csv` 按 1/5/10/20 日等周期列出 Rank IC 方向、高低两侧毛/净
年化与夏普，以及扣费前后是否为正；`annual_ic.csv` 和 `annual_long_only.csv` 用于
检查单年稳定性。原多空文件只作为统计诊断，不会送入实际组合回测。

### 自定义因子检验

`custom_001` 定义为日收益率横截面排名与成交额横截面排名的
5 日滚动协方差，再取负的横截面排名：

```text
-rank(covariance(rank(close / delay(close, 1) - 1), rank(turnover_value), 5))
```

这里的 `turnover_value` 是成交额：优先使用同名原始字段，其次使用
`amount * 1000`，最后回退为 `close * volume`。运行：

```bash
python -m rquant factor-test --factor custom_001 --data data/raw --metadata config/stocklist.csv --windows 1 5 10 20 --groups 10 --output factor_report
```

输出目录：

```text
factor_report/custom_001/
```

旧名称 `custom_return_turnover_cov_5d` 仍可作为兼容别名使用，但 CLI 会将其
规范化为 `custom_001`。

`custom_002` 定义为收盘价相对 VWAP 折价的横截面排名：

```text
rank((vwap - close) / vwap)
```

可以使用同一个入口运行：

```bash
python -m rquant factor-test --factor custom_002 --data data/raw --metadata config/stocklist.csv --windows 1 5 10 20 --groups 10 --output factor_report
```

若原始数据没有 `vwap` 或 `avg`，面板构建器使用
`(high + low + close) / 3` 作为 VWAP 回退值。

### Alpha101 批处理

查看因子生命周期状态：

```bash
python -m rquant factor-batch --list-factor-status
```

按 `config/factors.yaml` 运行当前 `active` / `watch` 因子：

```bash
python -m rquant factor-batch --family alpha101 --data data/raw --metadata config/stocklist.csv --output factor_report/alpha101_batch/latest --factor-config config/factors.yaml --factors all --windows 1 5 10 20 --groups 10 --top-counts 1 5 10 20 50 100 --start-date 2020-01-01 --end-date 2026-06-30 --min-listing-days 60 --liquidity-lookback-days 20 --commission-rate 0.0003 --slippage-rate 0.0005 --stamp-tax-rate 0.0005 --oos-start-date 2025-01-01
```

默认会复用实现签名和数据签名一致的已落盘结果；只有确认需要全部重算时才追加
`--force`。

输出：

```text
factor_report/alpha101_batch/
```

生命周期配置：

```text
config/factors.yaml
```

Alpha101 的默认库是“经济解释优先”的研究池：只把具备明确价格位置、趋势/反转、
波动或价量/流动性假设的公式列为 `watch`；其余不透明的遗传规划表达式（任意小数
窗口/权重、原始价格阈值或高次幂）默认为 `disabled`。`watch` 仍须通过样本外的
长多 TopN / Top Quantile 检验后才可升级为 `active`。如需复现被禁用的原始 WorldQuant
公式，可在一次性实验中追加 `--ignore-factor-config`，不要将其作为默认特征池。

### GTJA191 批处理

```bash
python -m rquant factor-batch --family gtja191 --data data/raw --metadata config/stocklist.csv --output factor_report/gtja191_batch/latest --factor-config config/gtja191_factors.yaml --factors all --windows 1 5 10 20 --groups 10 --top-counts 1 5 10 20 50 100 --min-listing-days 60
```

同样只有确认需要全部重算时才追加 `--force`。GTJA191 当前批处理没有日期区间、
流动性门槛或交易费率参数；不要在该命令中复制 Alpha101 专用参数。

输出：

```text
factor_report/gtja191_batch/
```

生命周期配置：

```text
config/gtja191_factors.yaml
```

GTJA191 与 Alpha101 一样按经济解释管理：默认只运行 `watch` 的简单价格位置、
趋势/反转、波动与价量/流动性假设；任意拟合权重、原始价格阈值/高次幂和不透明嵌套
表达式默认 `disabled`。当前不设置 `active`，历史表现不能替代样本外长多验证。

为保持现有生命周期读取兼容，`factors` 继续保存纯状态字符串，研究分类放在独立的
`categories` 映射：

```yaml
factors:
  alpha_040: watch
categories:
  alpha_040: price_volume
```

当前分类名包括 `price_behavior`、`price_volume`、`traditional_technical`、
`market_related`、`liquidity` 等；新因子库应先分类再进入批处理。批量
`leaderboard.csv` 会保留 `factor_category`，不会因分类而改变公式或生命周期状态。
GTJA191 对全部 191 个公式提供确定性分类：需要指数/风格序列的公式归入
`market_related`，纯成交量/成交额公式归入 `liquidity`，经典 RSI/KDJ/ADX/ATR
等变体归入 `traditional_technical`，价格和成交量联合公式归入 `price_volume`，
其余归入 `price_behavior`。YAML 可显式覆盖类别；类别不会把 `disabled` 因子自动启用。

### 外部因子库统一入口

新的因子库无需先改写为 Alpha101/GTJA191。可以提供宽表：

```text
date,symbol,factor_a,factor_b,...
```

也可以提供长表：

```text
date,symbol,factor,factor_value
```

文件中的值统一表示当日尚未滞后的原始因子值；批量检验、相关分析和 ML 都会在各自
边界严格应用一次 `shift(1)`。主键重复、日期无效、股票代码无效或缺少指定因子时会
直接失败，不会静默聚合。批量完整检验命令：

```bash
python -m rquant factor-batch --family external --factor-file data/factors/my_factors.csv --factor-layout auto --data data/raw --metadata config/stocklist.csv --factors all --factor-config config/my_factors.yaml --windows 1 5 10 20 --groups 10 --output factor_report/external_batch/latest
```

当前 `data/raw/*.csv` 主要是 OHLCV/amount 日线，并不天然包含历史市值。市值分档、
动态行业和自定义牛熊状态应通过单独的时点上下文文件提供，不能拿当前静态市值回填：

```text
date,symbol,market_cap,industry,sector,market_regime
2020-01-02,000001,1234567890,银行,金融,sideways
```

```bash
python -m rquant factor-batch --family external --factor-file data/factors/my_factors.csv --context-file data/context/daily_context.csv --data data/raw --factors all --windows 1 5 10 20 --output factor_report/external_batch/latest
```

`--context-date-col`、`--context-symbol-col` 可映射自定义主键列。上下文必须按
`date + symbol` 唯一；动态市值、行业和市场状态只合并到对应日期，不向过去或未来
填充。内置 Alpha101/GTJA191 的 `factor-batch`、单因子 `factor-test` 和 ML 入口也支持
同一个 `--context-file`。

外部生命周期和分类文件仍使用同一结构：

```yaml
default_status: active
factors:
  factor_a: active
  factor_b: watch
categories:
  factor_a: price_behavior
  factor_b: price_volume
```

输出与内置批处理一致，包括每个因子的全套 IC/分层/市场状态/逐年/高低两侧长多报告、
`batch_status.csv` 和带 `factor_category` 的 `leaderboard.csv`。此外，
所有 Alpha101、GTJA191 和外部因子批处理都会生成 `long_only_profitability.csv`，
将每个因子、周期的高值/低值两侧拆成独立行，
`profitable_long_only.csv` 只保留扣费前或扣费后收益为正的行；两者都不会生成空头仓位。
行情、每日市值和交易状态
从 `data/raw` 按 `date + symbol` 一对一合并；外部文件已提供的时点字段优先。静态股票
表只补行业分类，不会用当前市值替代历史市值。外部因子文件内容、行情目录签名和检验
参数均未变化时会逐因子断点复用；`--force` 才会重算。

### 因子相关性与去重

先计算每日横截面的 Spearman（主口径）和 Pearson（辅助口径），再跨日期平均：

```bash
python -m rquant factor-correlation --data data/raw --metadata config/stocklist.csv --factors all --high-correlation-threshold 0.8 --output factor_report/factor_correlation
```

GTJA191 使用同一个相关性和去重入口；默认读取其独立生命周期配置：

```bash
python -m rquant factor-correlation --family gtja191 --data data/raw --metadata config/stocklist.csv --factors all --high-correlation-threshold 0.8 --priority-file factor_report/gtja191_batch/leaderboard.csv --priority-score-col preferred_net_sharpe --priority-window 20 --eligibility-col preferred_profitable_after_cost --output factor_report/gtja191_correlation
```

只有需要大盘或风格序列的 GTJA 因子才需额外传入 `--benchmark-file` 或
`--style-factor-file`；缺失输入会逐因子记为失败，不会伪造代理数据。

如需按样本外净夏普选择高相关簇中的代表因子，可提供批量结果表和质量字段：

```bash
python -m rquant factor-correlation --data data/raw --metadata config/stocklist.csv --factors all --high-correlation-threshold 0.8 --priority-file factor_report/alpha101_batch/latest/leaderboard.csv --priority-factor-col factor --priority-score-col preferred_net_sharpe --output factor_report/factor_correlation
```

输出包括完整矩阵、两两相关表、`deduplication.csv` 和
`deduplicated_factors.csv`。相关性绝对值达到阈值的因子构成连通簇；有质量分数时保留
簇内最高分，否则只用稳定名称顺序生成暂定名单，并在 `selection_reason` 中明确标注。

同一份外部因子文件可直接计算相关性并使用批量净夏普去重：

```bash
python -m rquant factor-correlation --factor-file data/factors/my_factors.csv --factor-layout auto --factors all --high-correlation-threshold 0.8 --priority-file factor_report/external_batch/latest/leaderboard.csv --priority-score-col preferred_net_sharpe --priority-window 20 --eligibility-col preferred_profitable_after_cost --output factor_report/external_correlation
```

上述命令先用 20 日周期、与 IC 方向一致的净夏普选择高相关簇代表，再把扣费后确实盈利的
代表写入 `ml_candidate_factors.csv`。完整的 `deduplicated_factors.csv` 仍保留作审计，
不会把不盈利代表静默送入 ML。

### 因子筛选和组合回测

生成 Alpha077 过滤、Alpha040 排序信号：

```bash
python -m rquant factor-select --start 2025-01-01 --end 2026-06-23 --filter-top-quantile 0.5 --top-n 10
```

直接生成信号并运行组合回测：

```bash
python -m rquant factor-backtest --start 2025-01-01 --end 2026-06-23 --hold-days 20
```

默认因子组合回测参数：

```text
filter-top-quantile = 0.8
top-n = 500
initial-cash = 10000000
```

### 多因子排名组合

`factor-ensemble-select` 先把每个因子在当日股票横截面转换为百分位，再按显式权重
计算综合分。它不会直接相加不同量纲的原始因子值。因子值和股票池字段默认统一
滞后一个交易日；默认要求每只股票的全部组合因子都有值。

生成 `alpha_040 + alpha_069 + alpha_077` 的研究信号：

```bash
python -m rquant factor-ensemble-select --factors alpha_040 alpha_069 alpha_077 --weights 0.6 0.2 0.2 --min-factor-coverage 1.0 --top-n 10 --start 2025-01-01 --end 2026-06-23 --output data/factor_signals/ensemble_040_069_077
```

如果某个因子是数值越低越好，必须通过 `--ascending-factors` 显式列出。权重只决定
各因子百分位的相对贡献，程序会将其归一化。输出包含每只股票的原始因子值、因子
百分位、有效权重覆盖率、综合分和最终名次：

```text
data/factor_signals/ensemble_040_069_077/signals.csv
data/factor_signals/ensemble_040_069_077/selections.csv
data/factor_signals/ensemble_040_069_077/daily_summary.csv
data/factor_signals/ensemble_040_069_077/filter_status.csv
data/factor_signals/ensemble_040_069_077/manifest.json
```

使用同一组合进入考虑次日开盘、费用、涨跌停、停牌、整手和固定资金槽位的组合回测：

```bash
python -m rquant factor-ensemble-backtest --factors alpha_040 alpha_069 alpha_077 --weights 0.6 0.2 0.2 --min-factor-coverage 1.0 --top-n 10 --start 2025-01-01 --end 2026-06-23 --hold-days 20 --initial-cash 10000000 --output data/portfolio_backtest_factor_ensemble_040_069_077
```

组合权重是需要样本外验证的研究假设，不代表已证明最优。该入口目前计算 Alpha101
组件；通用长表组合器位于 `factors/ensemble.py`，后续其他因子族应通过适配器接入，
不得复制组合公式或绕过统一信号层。

### ML 多因子拟合器

`fit-multifactor` 是从因子到样本外综合分数的一体化入口。它会先计算并滞后所有
显式指定的因子，再在完全相同的 walk-forward 窗口中运行 Ridge、ElasticNet、
Qlib LightGBM、Qlib DoubleEnsemble 和 Torch MLP。默认将特征和标签都转换为每日横截面百分位，因此学习
目标是做多排名，而不是精确拟合极端的原始收益率。

```bash
python -m rquant fit-multifactor --data data/raw --metadata config/stocklist.csv --factors alpha_040 alpha_069 alpha_077 custom_001 custom_002 --models ridge elasticnet lightgbm doubleensemble mlp --target-window 20 --feature-transform rank --target-transform rank --train-size 504 --test-size 21 --signal-top-n 10 --start 2018-01-01 --end 2026-06-30 --output data/ml/multifactor_20d
```

严格执行“过去三个完整日历年训练、预测下一年”时使用：

```bash
python -m rquant fit-multifactor --data data/raw --metadata config/stocklist.csv --factors alpha_040 alpha_069 alpha_077 custom_001 custom_002 --models ridge elasticnet lightgbm doubleensemble mlp --target-window 20 --feature-transform rank --target-transform rank --window-mode calendar-years --train-years 3 --test-years 1 --signal-top-n 10 --start 2018-01-01 --end 2026-06-30 --output data/ml/multifactor_3y_1y
```

每个测试年只使用之前三个日历年的数据；标签对应的 purge 交易日从训练期末端剔除，
预测仍只写入样本外年份。默认 `trading-days` 模式继续保留，用于原有 504/21 日滚动
实验，两种窗口不会静默混用，具体边界写入 `windows.csv` 和 manifest。

运行时不显示进度条，而是持续输出并记录当前因子、模型、训练/测试窗口、断点复用状态、耗时和产物路径。终端日志同步写入
`data/runs/<run-id>/run.log`。
MLP 每个 walk-forward 窗口默认训练 10 个 epoch；如需做对照实验，可显式传入
`--mlp-epochs N`。
ElasticNet 默认使用 `alpha=0.001` 和 `l1_ratio=0.5`，以避免在每日横截面
rank 特征上因惩罚过强而将所有系数压为零。两者可分别用
`--elasticnet-alpha` 和 `--elasticnet-l1-ratio` 覆盖；默认值只是研究起点，
仍需用样本外 Rank IC 和后续 long-only 回测验证。

GTJA191 可直接导入其生命周期配置中的可解释研究因子；当前
`config/gtja191_factors.yaml` 的 `watch` 项是待样本外验证的 ML 特征池。也可以将它与
显式指定的 Alpha101/custom 特征组合：

```bash
python -m rquant fit-multifactor --data data/raw --metadata config/stocklist.csv --factor-config config/gtja191_factors.yaml --lifecycle-statuses watch --models ridge elasticnet lightgbm doubleensemble mlp --target-window 20 --feature-transform rank --target-transform rank --train-size 504 --test-size 21 --signal-top-n 10 --start 2018-01-01 --end 2026-06-30 --output data/ml/gtja191_watch_20d
```

默认只导入 `active`；本配置当前没有 `active`，因此应显式使用
`--lifecycle-statuses watch`。导入仍会统一执行一日因子滞后和 walk-forward，
不会把 ML 分数反写到因子生命周期配置。

`next_open_return_20d` 会自动使用至少 21 个交易日的 purge gap。主要产物：

```text
data/ml/multifactor_20d/dataset/features.csv
data/ml/multifactor_20d/dataset/labels.csv
data/ml/multifactor_20d/models/<model>/predictions.csv
data/ml/multifactor_20d/models/<model>/signals.csv
data/ml/multifactor_20d/models/<model>/summary.json
data/ml/multifactor_20d/leaderboard.csv
data/ml/multifactor_20d/manifest.json
```

`leaderboard.csv` 只用样本外 Rank IC 等预测诊断排序，不将其表述为可交易利润。
每个模型的 `signals.csv` 还需通过 `signal-backtest` 进入次日开盘、费用、涨跌停、
停牌、整手和 T+1 约束下的 long-only 组合回测。

### 机器学习分步 walk-forward

先从本地日线生成统一滞后因子特征和 forward-return 标签：

```bash
python -m rquant make-ml-dataset --data data/raw --metadata config/stocklist.csv --factors alpha_040 alpha_077 custom_002 --target-windows 20 --factor-lag-days 1 --label-mode next_open --start 2018-01-01 --end 2026-06-30 --output data/ml/dataset_20d
```

`make-ml-dataset` 固定保留 `shift(1)`。默认标签与真实组合回测一致：信号日后的
下一个开盘价买入，持有 N 个交易日后按开盘价退出。程序先用完整行情计算标签，再
截取研究日期，因此不会因 `--end` 提前截断标签所需的未来价格。若只做统计诊断，
可显式传 `--label-mode close_to_close`。输出：

```text
data/ml/dataset_20d/features.csv
data/ml/dataset_20d/labels.csv
data/ml/dataset_20d/manifest.json
```

特征支持 Alpha101、GTJA191 和已注册 custom 因子。需要指数或 MKT/SMB/HML 的
GTJA 因子必须通过 `--benchmark-file` / `--style-factor-file` 提供真实时点序列，
缺失时会明确失败。

外部因子也可直接进入 ML，并可与内置因子混用：

```bash
python -m rquant fit-multifactor --data data/raw --factor-file data/factors/my_factors.csv --factor-layout auto --context-file data/context/daily_context.csv --factor-selection-file factor_report/external_correlation/ml_candidate_factors.csv --factor-selection-col factor --models ridge elasticnet lightgbm doubleensemble mlp --target-window 20 --feature-transform rank --target-transform rank --window-mode calendar-years --train-years 3 --test-years 1 --signal-top-n 10 --start 2018-01-01 --end 2026-06-30 --output data/ml/external_3y_1y
```

GTJA191 去重后的长多盈利因子可直接使用同一衔接文件；`--run-backtests` 会对每个模型的
样本外买入信号分别运行零成本和实际成本组合回测，并把 `gross_*`、`net_*`、是否盈利、
交易数和最大回撤写入模型排行榜；至少一个口径盈利的模型另写入
`profitable_models.csv`：

```bash
python -m rquant fit-multifactor --data data/raw --context-file data/context/daily_basic --factor-selection-file factor_report/gtja191_correlation/ml_candidate_factors.csv --models ridge elasticnet lightgbm doubleensemble mlp --target-window 20 --window-mode calendar-years --train-years 3 --test-years 1 --signal-top-n 10 --run-backtests --backtest-commission-wan 0.8 --start 2018-01-01 --end 2026-07-10 --output data/ml/gtja191_3y_1y
```

这里的两套回测都只读取 `signal_type=buy`；零成本口径把佣金、印花税和过户费同时设为
零，实际成本口径默认使用万分之 0.8 佣金、万分之 5 印花税和十万分之 1 过户费。
`--run-backtests` 默认关闭，避免普通模型拟合无意触发多次全市场组合回测。

`dataset/manifest.json` 会逐列记录 `external`、`alpha101`、`gtja191` 或 `custom`
来源。模型仍只产生样本外多头排名信号；未开启集成回测时，可把对应的 `signals.csv`
手工交给 `signal-backtest`，不会生成或执行空头仓位。

```bash
python -m rquant signal-backtest --signals data/ml/external_3y_1y/models/ridge/signals.csv --source model_ridge --data data/raw --hold-days 20 --max-positions 10 --commission-wan 0.8 --output data/portfolio_backtest_external_ridge
```

`train-model` 接受生成的两个 long-format CSV，并按 `date + symbol` 一对一对齐：

```text
# features.csv
date,symbol,alpha_040,alpha_077,custom_002

# labels.csv
date,symbol,next_open_return_20d
```

特征列必须在命令中显式列出。`next_open_return_Nd` 需要未来第 `N+1` 个开盘价，
程序会自动使用至少 `N+1` 个交易日的 purge gap；诊断型 `forward_return_Nd`
使用至少 N 日。其他目标必须显式传入 `--purge-days`。

当前环境无需额外 ML 依赖即可运行 Ridge：

```bash
python -m rquant train-model --features data/ml/dataset_20d/features.csv --labels data/ml/dataset_20d/labels.csv --feature-cols alpha_040 alpha_077 custom_002 --target-col next_open_return_20d --model ridge --train-size 504 --test-size 21 --signal-top-n 10 --output data/ml/ridge_20d
```

每个窗口都会重新训练，只将测试窗口预测写入总结果。逐窗结果带输入、配置和实现签名，
相同签名默认断点复用；确认需要全部重算时追加 `--force`。主要输出：

```text
data/ml/ridge_20d/predictions.csv       全部样本外分数
data/ml/ridge_20d/signals.csv           每日 Top-N 统一信号
data/ml/ridge_20d/windows.csv           训练、purge、测试边界
data/ml/ridge_20d/metrics.csv           逐窗 MSE、MAE、Pearson、Rank IC
data/ml/ridge_20d/summary.json          汇总和输入对齐审计
data/ml/ridge_20d/manifest.json         配置、签名和输出清单
data/ml/ridge_20d/windows/window_*/      逐窗预测、指标和模型文件
```

Ridge 和 ElasticNet 在没有 sklearn 时也有经过测试的 NumPy 实现；sklearn
后端和 Torch MLP 使用现有 ML 可选依赖：

```bash
python -m pip install -r requirements-ml.txt
```

Qlib LightGBM 和 DoubleEnsemble 单独安装，不进入核心 `requirements.txt`：

```bash
python -m pip install -r requirements-qlib.txt
```

`--model lightgbm` 直接绑定 Qlib `LGBModel`，`--model doubleensemble` 使用
Qlib `DEnsembleModel`；二者都没有旧的原生 LightGBM 回退或一致性对照。每个
walk-forward 训练窗口的末尾默认按时间切出 20% 作为 Qlib validation，
可用 `--qlib-valid-ratio` 调整；DoubleEnsemble 子模型数可用
`--doubleensemble-num-models` 调整。validation 和样本外 test 严格分离，原有
purge gap 仍由 RQuant walk-forward 层执行并写入 `windows.csv`。

Qlib LightGBM 示例：

```bash
python -m rquant train-model --features data/ml/dataset_20d/features.csv --labels data/ml/dataset_20d/labels.csv --feature-cols alpha_040 alpha_077 custom_002 --target-col next_open_return_20d --model lightgbm --lightgbm-estimators 200 --qlib-valid-ratio 0.2 --output data/ml/qlib_lightgbm_20d
```

DoubleEnsemble 示例：

```bash
python -m rquant train-model --features data/ml/dataset_20d/features.csv --labels data/ml/dataset_20d/labels.csv --feature-cols alpha_040 alpha_077 custom_002 --target-col next_open_return_20d --model doubleensemble --lightgbm-estimators 200 --doubleensemble-num-models 6 --qlib-valid-ratio 0.2 --output data/ml/qlib_doubleensemble_20d
```

macOS 上若 `doctor` 报告 Qlib 内部 LightGBM 缺少 `libomp.dylib`，在对应 Conda 环境补装：

```bash
conda install -n stocktrade -c conda-forge llvm-openmp
```

Windows 上若 `doctor` 报告 Qlib 内部 LightGBM 无法加载 DLL，请先确认使用的是已激活环境中的
`python`，然后重新安装 `requirements-qlib.txt`；仍失败时请安装 Microsoft Visual C++ 2015–2022
Redistributable 后重新打开终端。不要在项目目录中复制 DLL 文件。

Qlib LightGBM/DoubleEnsemble 默认使用单工作线程以保证本地研究可复现；可按机器资源显式传
`--lightgbm-n-jobs N`。Torch MLP 的 `--device auto` 会按 MPS、CUDA、CPU 的顺序
选择实际可用设备。

MLP 示例：

```bash
python -m rquant train-model --features data/ml/dataset_20d/features.csv --labels data/ml/dataset_20d/labels.csv --feature-cols alpha_040 alpha_077 custom_002 --target-col next_open_return_20d --model mlp --mlp-hidden-sizes 64 32 --device auto --output data/ml/mlp_20d
```

MLP 默认训练 10 个 epoch，可用 `--mlp-epochs N` 覆盖。

`device=auto` 在可用时依次选择 Apple MPS、CUDA、CPU。Qlib 预测在
RQuant 边界只保留 `score`，再转成 `signals.csv` 进入统一信号层；训练代码
不会改写因子值或自定义买点。

将模型输出送入与因子相同的次日开盘和固定资金槽位回测：

```bash
python -m rquant signal-backtest --signals data/ml/ridge_20d/signals.csv --source model_ridge --data data/raw --hold-days 20 --max-positions 10 --initial-cash 10000000 --output data/portfolio_backtest_model_ridge_20d
```

Qlib 模型使用同一入口，仅替换信号文件和 source，例如：

```bash
python -m rquant signal-backtest --signals data/ml/qlib_doubleensemble_20d/signals.csv --source model_doubleensemble --data data/raw --hold-days 20 --max-positions 10 --initial-cash 10000000 --output data/portfolio_backtest_qlib_doubleensemble_20d
```

`signal-backtest` 可读取任何符合统一字段的信号文件，并按 `score` 从高到低保留每日
候选。当前严格 cohort 引擎只支持等权候选；同一天存在不等权 `weight` 时会明确
拒绝，而不会静默忽略。信号文件包含多个 `source` 时必须显式传 `--source`。

### 研究报告

```bash
python -m rquant research-report --signal-dir data/backtest --portfolio-dir data/portfolio_backtest --candidates data/candidates/candidates_latest.json --review data/review/2026-06-23/suggestion.json --output data/reports
```

报告默认严格校验必需 JSON、候选/复评日期、复评完整状态、信号与组合的 `buy_mode`
以及可比较的研究日期范围；不一致时不会写出报告。仅在诊断旧产物时可显式追加：

```bash
--allow-inconsistent
```

JSON 报告包含 `validation` 和每个输入文件的 SHA-256 `source_fingerprints`，HTML
首页同步展示一致性错误与警告。旧版复评文件没有 `status` 时会明确标记“完整性未验证”。

输出：

```text
data/reports/research_report.json
data/reports/research_report.html
```

### 顶层 CLI 注册器

新布局的正式命令注册入口：

```bash
python -m rquant --help
```

它现在直接调度系统自检、行情抓取、单因子检验、生命周期批处理、自定义策略、
因子组合、机器学习和统一信号回测。旧的 `python -m market.fetch_kline`、
`scripts/quant_cli.py`、`scripts/test_factor.py` 与 `scripts/test_factor_batch.py` 仍保留为
兼容入口，内部与正式入口复用同一实现。

## 输出目录

```text
data/raw/                  原始日线行情和抓取检查点
data/candidates/           初选候选列表
data/kline/<date>/          点时候选图表和导出检查点
data/review/<date>/         Gemini 逐股结果、恢复清单和研究候选汇总
data/backtest*/             信号收益明细和汇总
data/portfolio_backtest*/   组合回测交易、持仓、权益曲线和摘要
data/reports/               综合研究报告
data/runs/<run-id>/          统一运行清单和日志
factor_report/              因子检验和批处理结果
```

`data/` 和 `factor_report/` 默认是本地研究产物，不纳入 Git。

## 验证

快速检查 CLI：

```bash
python -m rquant --help
python -m rquant --help
```

运行核心测试：

```bash
python -m unittest tests.test_cli
python -m unittest tests.test_factor_research_pipeline
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
Ran 325 tests
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
