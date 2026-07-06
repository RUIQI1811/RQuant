# GTJA191 Factor Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `gtja_001` through `gtja_191`, expose them through FactorTester, and provide a resumable one-command evaluation of all 191 factors.

**Architecture:** `pipeline/factors/gtja191.py` owns the independent GTJA registry, report-defined operators, formulas, panels, and long adapter. A separate batch runner evaluates one factor at a time and writes auditable checkpoints under `factor_report/gtja191_batch`, without changing Alpha101 formulas or custom buy strategies.

**Tech Stack:** Python 3.11, pandas, NumPy, PyYAML, `unittest`, existing FactorTester reports.

---

## File structure

- Create `pipeline/factors/gtja191.py`: data model, operators, registry, formulas, adapters.
- Create `tests/test_gtja191.py`: operator, formula, registry, adapter, and missing-input tests.
- Modify `pipeline/factors/__init__.py`: public exports.
- Modify `pipeline/factor_tester.py`: strict `gtja` namespace routing.
- Modify `scripts/test_factor.py`: list and run GTJA factors with optional external inputs.
- Create `pipeline/gtja191_batch.py`: selection, resume fingerprints, checkpoints, reports.
- Create `scripts/test_gtja191_batch.py`: user-facing single/all-factor CLI.
- Create `tests/test_gtja191_batch.py`: parsing, resume, isolation, and full-status tests.
- Create `config/gtja191_factors.yaml`: independent lifecycle catalog.
- Modify `README.md` and `docs/architecture.md`: commands, outputs, and boundaries.

Existing unrelated working-tree changes belong to the user. Each commit stages only files listed by its task.

### Task 1: Data model and report-defined operators

**Files:**
- Create: `pipeline/factors/gtja191.py`
- Create: `tests/test_gtja191.py`

- [ ] **Step 1: Write failing operator tests**

```python
class GTJA191OperatorsTest(unittest.TestCase):
    def test_normalize_gtja_name(self):
        self.assertEqual(normalize_gtja_name(1), "gtja_001")
        self.assertEqual(normalize_gtja_name("gtja191"), "gtja_191")
        with self.assertRaises(KeyError):
            normalize_gtja_name("alpha_001")

    def test_sma_cn_uses_report_recursion(self):
        frame = pd.DataFrame({"a": [1.0, 4.0, 7.0]})
        expected = pd.DataFrame({"a": [1.0, 2.0, 11.0 / 3.0]})
        pd.testing.assert_frame_equal(sma_cn(frame, 3, 1), expected)

    def test_wma_uses_point_nine_distance_weights(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 4.0]})
        expected = (4.0 + .9 * 2.0 + .9**2) / (1.0 + .9 + .9**2)
        self.assertAlmostEqual(wma(frame, 3).iloc[-1, 0], expected)

    def test_rolling_regression_uses_intercept(self):
        x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        y = 2.0 * x + 3.0
        self.assertAlmostEqual(regbeta(y, x, 4).iloc[-1, 0], 2.0)
        self.assertAlmostEqual(regresi(y, x, 4).iloc[-1, 0], 0.0)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191.GTJA191OperatorsTest -v
```

Expected: import failure because `pipeline.factors.gtja191` does not exist.

- [ ] **Step 3: Implement the operator foundation**

Define `GTJA191Error`, `GTJA191DataError`, `GTJA191FormulaError`,
`GTJA191ExternalData`, and `GTJA191Panels` with aligned open/high/low/close,
volume, amount, VWAP, returns, optional market cap/ST/industry, benchmark open
and close, and MKT/SMB/HML fields. Implement `normalize_gtja_name`, `sma_cn`,
`wma`, `regbeta`, `regresi`, `highday`, `lowday`, `count`, `sumif`, and
`sumac`. Use full rolling windows, preserve missing values, and replace
infinite output with `NaN`.

```python
GTJA191_NAMES = tuple(f"gtja_{number:03d}" for number in range(1, 192))

def normalize_gtja_name(name: str | int) -> str:
    raw = str(name).strip().lower().replace("-", "_")
    if isinstance(name, int):
        number = name
    elif raw.startswith("gtja_"):
        number = int(raw.removeprefix("gtja_"))
    elif raw.startswith("gtja"):
        number = int(raw.removeprefix("gtja"))
    else:
        raise KeyError(f"invalid GTJA191 factor name: {name}")
    if not 1 <= number <= 191:
        raise KeyError(f"GTJA191 factor number must be in [1, 191], got {number}")
    return f"gtja_{number:03d}"
```

- [ ] **Step 4: Verify GREEN and commit**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191.GTJA191OperatorsTest -v
git add pipeline/factors/gtja191.py tests/test_gtja191.py
git commit -m "feat: add GTJA191 panel operators"
```

Expected: operator tests pass.

### Task 2: Formula blocks 001 through 080

**Files:**
- Modify: `pipeline/factors/gtja191.py`
- Modify: `tests/test_gtja191.py`

- [ ] **Step 1: Write failing registry and representative formula tests**

```python
class GTJA191FirstHalfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panels = _complete_panels(days=320, symbols=6, include_external=True)
        cls.calculator = GTJA191(cls.panels)

    def test_registry_has_all_names(self):
        self.assertEqual(len(GTJA191_NAMES), 191)
        self.assertEqual(
            (GTJA191_NAMES[0], GTJA191_NAMES[-1]),
            ("gtja_001", "gtja_191"),
        )

    def test_gtja_001_matches_formula(self):
        expected = -correlation(
            rank(delta(np.log(self.panels.volume), 1)),
            rank((self.panels.close - self.panels.open) / self.panels.open), 6,
        )
        pd.testing.assert_frame_equal(self.calculator.calculate(1), expected)

    def test_gtja_015_matches_open_gap(self):
        expected = self.panels.open / delay(self.panels.close, 1) - 1.0
        pd.testing.assert_frame_equal(self.calculator.calculate(15), expected)

    def test_gtja_070_is_amount_volatility(self):
        pd.testing.assert_frame_equal(
            self.calculator.calculate(70), stddev(self.panels.amount, 6)
        )

    def test_first_eighty_are_aligned(self):
        for number in range(1, 81):
            self.assertEqual(
                self.calculator.calculate(number).shape, self.panels.close.shape
            )
```

- [ ] **Step 2: Verify RED**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191.GTJA191FirstHalfTest -v
```

Expected: missing `GTJA191` calculator or missing factor methods.

- [ ] **Step 3: Implement exact formulas 001-040**

Add `GTJA191.calculate()` and `calculate_many()`, derived DTM/DBM/TR/HD/LD
panels, and methods `gtja_001` through `gtja_040`. Use the canonical formula
table and original appendix semantics. `gtja_030` requires MKT/SMB/HML and
raises `GTJA191DataError` naming absent fields. Record each web-source spelling
or parenthesis correction in `GTJA191_FORMULA_NOTES`.

```python
def gtja_001(self) -> Panel:
    return -correlation(
        rank(delta(np.log(self.d.volume.mask(self.d.volume <= 0)), 1)),
        rank(_safe_div(self.d.close - self.d.open, self.d.open)),
        6,
    )

def gtja_015(self) -> Panel:
    return _safe_div(self.d.open, delay(self.d.close, 1)) - 1.0
```

- [ ] **Step 4: Implement exact formulas 041-080**

Add methods `gtja_041` through `gtja_080`. Preserve recursive SMA semantics,
conditional missing values, and explicit benchmark open/close requirements for
`gtja_075`. Do not approximate external data with stock-pool averages.

- [ ] **Step 5: Verify GREEN and commit**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191 -v
git add pipeline/factors/gtja191.py tests/test_gtja191.py
git commit -m "feat: implement GTJA191 factors 001 to 080"
```

Expected: operator and first-80 formula tests pass.

### Task 3: Formula blocks 081 through 160

**Files:**
- Modify: `pipeline/factors/gtja191.py`
- Modify: `tests/test_gtja191.py`

- [ ] **Step 1: Write failing representative, recursion, and block tests**

```python
def test_gtja_100_is_twenty_day_volume_std(self):
    pd.testing.assert_frame_equal(
        self.calculator.calculate(100), stddev(self.panels.volume, 20)
    )

def test_gtja_126_is_typical_price(self):
    expected = (self.panels.close + self.panels.high + self.panels.low) / 3.0
    pd.testing.assert_frame_equal(self.calculator.calculate(126), expected)

def test_gtja_143_carries_prior_value_on_non_positive_day(self):
    actual = self.calculator.calculate(143)
    mask = self.panels.close <= delay(self.panels.close, 1)
    pd.testing.assert_frame_equal(actual.where(mask), delay(actual, 1).where(mask))

def test_middle_eighty_are_aligned(self):
    for number in range(81, 161):
        self.assertEqual(self.calculator.calculate(number).shape, self.panels.close.shape)
```

- [ ] **Step 2: Verify RED**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191 -v
```

Expected: missing `gtja_081` and later methods.

- [ ] **Step 3: Implement exact formulas 081-120**

Add `gtja_081` through `gtja_120`, including covariance corrections and rolling
regression in `gtja_116`. Add correction notes for every source typo.

- [ ] **Step 4: Implement exact formulas 121-160**

Add `gtja_121` through `gtja_160`, including `SELF` recursion in `gtja_143`,
benchmark-conditioned `gtja_149`, `DELAT` to `DELTA` in 131, and `HGIH` to
`HIGH` in 159. Remaining genuine ambiguity raises `GTJA191FormulaError`.

- [ ] **Step 5: Verify GREEN and commit**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191 -v
git add pipeline/factors/gtja191.py tests/test_gtja191.py
git commit -m "feat: implement GTJA191 factors 081 to 160"
```

Expected: all tests through factor 160 pass.

### Task 4: Formula block 161 through 191 and complete execution

**Files:**
- Modify: `pipeline/factors/gtja191.py`
- Modify: `tests/test_gtja191.py`

- [ ] **Step 1: Write failing final and complete-library tests**

```python
def test_gtja_168_is_negative_relative_volume(self):
    expected = -self.panels.volume / mean(self.panels.volume, 20)
    pd.testing.assert_frame_equal(self.calculator.calculate(168), expected)

def test_gtja_191_matches_formula(self):
    expected = correlation(mean(self.panels.volume, 20), self.panels.low, 5)
    expected += (self.panels.high + self.panels.low) / 2.0
    expected -= self.panels.close
    pd.testing.assert_frame_equal(self.calculator.calculate(191), expected)

def test_all_191_are_callable_and_aligned(self):
    for name in GTJA191_NAMES:
        result = self.calculator.calculate(name)
        self.assertEqual(result.index.tolist(), self.panels.close.index.tolist())
        self.assertEqual(result.columns.tolist(), self.panels.close.columns.tolist())
```

- [ ] **Step 2: Verify RED**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191 -v
```

Expected: missing `gtja_161` and later methods.

- [ ] **Step 3: Implement exact formulas 161-191**

Add directional-movement, cumulative, benchmark-dependent 181/182, conditional
variance-ratio 190, and all other methods through `gtja_191`. Resolve malformed
165/166/183 parentheses from the original report and record corrections.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191 -v
git add pipeline/factors/gtja191.py tests/test_gtja191.py
git commit -m "feat: complete GTJA191 factor formulas"
```

Expected: all 191 execute on complete generated data.

### Task 5: FactorTester and single-factor CLI integration

**Files:**
- Modify: `pipeline/factors/gtja191.py`
- Modify: `pipeline/factors/__init__.py`
- Modify: `pipeline/factor_tester.py`
- Modify: `scripts/test_factor.py`
- Modify: `tests/test_gtja191.py`
- Modify: `tests/test_factor_tester.py`

- [ ] **Step 1: Write failing routing tests**

```python
def test_gtja_adapter_uses_long_schema_and_six_digit_symbols(self):
    raw = _raw_symbol_frames(days=40, symbols=2)
    direct = gtja191_to_long(raw, "gtja_015")
    routed = build_long_factor_frame_from_raw(raw, factor_name="gtja_015")
    self.assertEqual(direct["symbol"].str.len().unique().tolist(), [6])
    self.assertIn("turnover_value", direct.columns)
    pd.testing.assert_frame_equal(direct, routed)

def test_alpha_and_gtja_names_do_not_collide(self):
    self.assertNotIn("alpha_001", GTJA191_NAMES)
    self.assertNotIn("gtja_001", ALPHA101_NAMES)
```

- [ ] **Step 2: Verify RED**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191 tests.test_factor_tester -v
```

Expected: GTJA raw routing is unsupported.

- [ ] **Step 3: Implement adapters, exports, and CLI flags**

Build aligned panels from raw OHLCV; prefer explicit VWAP, otherwise typical
price; convert Tushare `amount` consistently; preserve metadata. Add optional
benchmark/style data to the adapter. Route only `gtja` names in FactorTester.
Export the public API and add `--benchmark-file`/`--style-factor-file` plus all
GTJA names to `scripts/test_factor.py --list-factors`.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191 tests.test_factor_tester tests.test_alpha101 -v
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_factor.py --list-factors
git add pipeline/factors/gtja191.py pipeline/factors/__init__.py pipeline/factor_tester.py scripts/test_factor.py tests/test_gtja191.py tests/test_factor_tester.py
git commit -m "feat: expose GTJA191 through FactorTester"
```

Expected: tests pass and list output includes both factor families.

### Task 6: Unified resumable batch evaluation

**Files:**
- Create: `pipeline/gtja191_batch.py`
- Create: `scripts/test_gtja191_batch.py`
- Create: `tests/test_gtja191_batch.py`
- Create: `config/gtja191_factors.yaml`

- [ ] **Step 1: Write failing selection and terminal-status tests**

```python
def test_parse_all_returns_ordered_191_names(self):
    self.assertEqual(parse_gtja_selection(("all",), ()), GTJA191_NAMES)

def test_every_requested_factor_gets_one_terminal_status(self):
    result = _run_small_batch(("gtja_001", "gtja_030", "gtja_191"))
    self.assertEqual(result.status["factor"].tolist(), [
        "gtja_001", "gtja_030", "gtja_191"
    ])
    status = result.status.set_index("factor")["status"]
    self.assertEqual(status.loc["gtja_030"], "missing_input")
```

- [ ] **Step 2: Verify RED**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191_batch -v
```

Expected: batch module import failure.

- [ ] **Step 3: Implement the runner and CLI**

Follow the existing Alpha101 sequential checkpoint design with the GTJA registry.
Support include/exclude ranges, `all`, lifecycle filtering, `--ignore-factor-config`,
force, fail-fast, external files, windows, groups, data, and output. Status values
are `success`, `missing_input`, `formula_error`, `failed`, and `skipped`.
Checkpoint every requested factor and write atomically. Default config is:

```yaml
default_status: active
factors: {}
```

- [ ] **Step 4: Verify GREEN and commit**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191_batch tests.test_alpha101_batch -v
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_gtja191_batch.py --help
git add pipeline/gtja191_batch.py scripts/test_gtja191_batch.py tests/test_gtja191_batch.py config/gtja191_factors.yaml
git commit -m "feat: add resumable GTJA191 batch evaluation"
```

Expected: resume/failure tests pass and CLI exposes the full-run flags.

### Task 7: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify only GTJA files if verification reveals a GTJA regression.

- [ ] **Step 1: Document exact commands and boundaries**

Add the single-factor command, external-data flags, outputs, missing-input
behavior, and the unified command:

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_gtja191_batch.py \
  --data data/raw --factors all --ignore-factor-config \
  --windows 1 5 10 20 --groups 10 \
  --output factor_report/gtja191_batch
```

Document `batch_status.csv`, `leaderboard.csv`, logs, per-factor reports, and
the factor/custom-strategy boundary.

- [ ] **Step 2: Run syntax and targeted verification**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m py_compile pipeline/factors/gtja191.py pipeline/gtja191_batch.py scripts/test_gtja191_batch.py
/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_gtja191 tests.test_gtja191_batch tests.test_factor_tester tests.test_alpha101 tests.test_alpha101_batch tests.test_cli -v
```

Expected: exit code 0 and all targeted tests pass.

- [ ] **Step 3: Run the full unit suite**

```bash
/opt/miniconda3/envs/stocktrade/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expected: all collected tests pass; any unrelated pre-existing failure is
reported separately with its exact traceback.

- [ ] **Step 4: Verify scope and commit docs**

```bash
git diff --check
git status --short
git add README.md docs/architecture.md
git commit -m "docs: document GTJA191 research workflow"
```

Expected: no generated data, secrets, or unrelated files are staged.
