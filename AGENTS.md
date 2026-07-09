# StockTradebyZ 代理工作指南

## 适用范围

本文件适用于整个仓库。开发代理在修改代码、配置、测试或文档前，应先阅读本文件、`README.md` 和与任务相关的源码。

## 项目定位

StockTradebyZ 是一个面向 A 股的半自动量化研究工具，主要能力包括：

- 通过 Tushare 获取日线数据。
- 用 B1、砖型图、mBDSR、BDSR/MACD/OBV 共振等自定义规则初选股票。
- 检验因子的 IC、分组收益、多空收益和换手率。
- 运行信号收益统计和考虑交易约束的组合回测。
- 导出候选股 K 线图，可选调用 Gemini 进行图表复评。
- 将候选、回测和复评结果汇总为 JSON、CSV 和 HTML 报告。

这是研究与决策辅助项目，不得把回测或 AI 评分表述为确定性收益、保证获利或自动交易指令。

## 架构与边界

项目必须保留相互独立的研究路径：

1. **因子研究**
   - 因子计算、检验与排名信号放在 `factors/`、`reports/factor_tester.py` 和 `signals/`。
   - 因子 IC、Rank IC、分组收益等统计不得与自定义买点的评价逻辑混合。
2. **自定义买入策略**
   - B1、brick、mBDSR、BDSR/MACD/OBV 共振及新的明确买点规则放在 `strategies/`。
   - 旧 `pipeline/` 路径仅作为兼容包装，不能成为新业务逻辑的主要位置。
3. **机器学习研究**
   - forward return 与训练标签放在 `labels/`。
   - Ridge、ElasticNet、LightGBM、MLP 等模型放在 `models/`。
   - walk-forward、验证和预测分数放在 `training/`。
   - ML 分数只能通过 `signals/` 进入组合回测，不得反向污染因子计算或自定义买点。

各研究路径只通过统一信号层衔接，不通过重写原有策略来强行合并。统一信号定义在 `signals/schema.py`，稳定字段为：

```text
date, symbol, signal_type, source, score, weight, metadata
```

- 自定义策略使用 `signals/strategy_adapters.py` 转换 `Candidate` / `CandidateRun`。
- 因子使用 `signals/factor_adapters.py` 转换为排名信号。
- `symbol` 始终按六位字符串处理，不得转换为会丢失前导零的整数。
- 详细映射以 `docs/architecture.md` 为准。

## 主要入口与数据流

- `run_all.py`：日常全流程编排，按顺序运行抓取、初选、图表导出、Gemini 复评和结果打印；子步骤失败时应立即终止。
- `pipeline/cli.py`：研究工作流的统一 CLI，对外命令包括 `preselect`、`signal-returns`、`portfolio-backtest` 和 `research-report`。
- `config/*.yaml`：运行参数的主要入口。参数不应无理由硬编码在业务逻辑中。
- `data/raw/`：原始日线数据。
- `data/candidates/`：初选结果。
- `data/backtest*/` 和 `data/portfolio_backtest*/`：信号及组合回测输出。
- `data/kline/` 和 `data/review/`：图表与 AI 复评输出。
- `data/reports/`：研究报告。
- `factor_report/`：因子检验结果。

## 工作原则

1. **先复现，再修改**
   - 调试时先记录实际解释器：`python -c "import sys; print(sys.executable)"`。
   - 从真实堆栈的第一个有效异常入手，区分环境缺失、外部服务失败和业务逻辑回归。
   - 优先用最小 smoke test 保留原始证据，不得一遇到错误就大范围重构。
2. **小步修改，保留兼容**
   - 优先在现有边界内修复问题，避免与任务无关的格式化、重命名和全局重构。
   - 新功能尽量通过注册、适配器或新模块接入，不直接破坏旧脚本、CLI 参数或文件格式。
   - 修改前检查 `git status --short`；现有未提交变更默认属于用户，不覆盖、不回滚、不顺手清理。
3. **默认交付 CLI 和可审计输出**
   - 研究能力应先提供可重复的命令行入口、结构化输出和明确的输出路径。
   - 除非任务明确要求，不将 Streamlit 或其他 UI 作为新能力的唯一入口。
   - 长时间任务应支持断点续跑或逐步落盘。Gemini 复评需保留 `skip_existing` 能力。
4. **文档与行为同步**
   - 如果改变了 CLI、配置键、输入格式、输出路径或研究流程，同步更新 `README.md`。
   - 如果改变了模块责任或信号路由，同步更新 `docs/architecture.md`。
   - 文档应包含能直接复制执行的命令、前置条件和实际输出路径。

## 量化与回测正确性

与信号、因子、收益或组合回测有关的修改，必须明确检查：

- 不使用信号时点之后才可得的数据，防止未来函数和幸存者偏差。
- `signal_close` 与 `next_open` 的价格时点语义不得混淆。
- 组合回测保留现金、持仓、整手买入、最大持仓数、单票仓位和交易费用约束。
- 保留涨停不买、跌停不卖、停牌不交易和 A 股 T+1 约束。
- 手续费、印花税和过户费的买卖方向及计算基数正确。
- 股票池过滤只使用当日或当日之前可得的成交额、价格和可交易状态。
- 空信号、数据不足、缺失值、复权/不复权价格及窗口边界有可预期行为。
- 回测摘要不得隐藏交易数、未成交原因或缺失数据。

修改上述逻辑时，必须添加针对时间对齐、费用或交易约束的回归测试，不能只比较最终收益数字。

## 数据、密钥与外部服务

- 使用项目根目录的 `.env` 或环境变量保存 `TUSHARE_TOKEN` 和 `GEMINI_API_KEY`。
- 不读取、打印、记录、提交或在文档中复制真实密钥。
- 测试中使用临时目录、小型构造数据和 mock；不覆盖 `data/raw/` 或用户已有研究结果。
- 不将外部 API 调用当作普通单元测试。Tushare 和 Gemini 网络测试应明确标注为 smoke/integration test。
- Gemini 连通性优先用 `python test_gemini_smoke.py`验证，不先运行整批复评。
- 重跑长任务前检查已落盘输出，避免破坏断点续跑能力。

## 开发与验证命令

安装依赖：

```bash
python -m pip install -r requirements.txt
```

快速检查 CLI 是否可加载：

```bash
python -m pipeline.cli --help
```

运行针对性测试：

```bash
python -m unittest tests.test_cli
python -m unittest tests.test_portfolio_backtest
python -m unittest tests.test_factor_tester
```

运行全量单元测试：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

验证原则：

- 先运行与改动最相关的小型测试，再运行全量测试。
- 依赖缺失导致测试无法收集时，如实报告解释器路径、缺少的包和未运行的测试；不得将“未运行”表述为“已通过”。
- 除非任务要求真实全流程，验证不应抓取全量行情、批量调用 Gemini 或改写生产型输出目录。

## 完成标准

一项代码变更只有在以下条件满足时才算完成：

- 改动位于正确的架构边界，没有混合因子线和自定义策略线。
- 新增或修改的行为有针对性测试，并已运行与风险相匹配的验证。
- CLI、配置、输入输出或架构变更已同步到文档。
- 没有泄露密钥，没有覆盖用户数据，没有夹带与任务无关的修改。
- 交付说明列出了改动文件、验证命令、验证结果和仍存限制。
