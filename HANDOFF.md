# HANDOFF: Fix time_delta Reference & Retrain All Models

## Table of Contents
1. [Background & Context](#1-background--context)
2. [The Bug: time_delta Reference Mismatch](#2-the-bug-time_delta-reference-mismatch)
3. [The Fix: Use In-Play Transition as Reference](#3-the-fix-use-in-play-transition-as-reference)
4. [Step-by-Step Implementation Plan](#4-step-by-step-implementation-plan)
5. [Key Files Reference](#5-key-files-reference)
6. [SLURM Configuration Reference](#6-slurm-configuration-reference)
7. [Safety Rules & Gotchas](#7-safety-rules--gotchas)

---

## 1. Background & Context

### What is this project?
A quantitative trading system that predicts greyhound racing outcomes on the Betfair exchange. ML models (XGBoost + LightGBM) predict win probabilities, identify underpriced runners, and place value bets with limit orders.

### What happened?
We completed the full R&D pipeline (data processing -> feature engineering -> model training -> backtesting -> fill simulation). The backtest showed strong results: **+19% ROI, Sharpe 3.6, z=3.90** on 2,010 out-of-sample bets at edge>3%.

We then built a paper trading system (`_05_paper_trading/`) and ran it live for 1 day (2026-02-27). The results were wildly different from the backtest:

| Metric | Live (Day 1) | Backtest |
|--------|-------------|----------|
| model_prob_sum per race | **0.574** | **0.987** |
| Avg odds of bets | 18.5 | 3.9 |
| % bets on favorites | 0% | 73% |
| Avg runner rank | 4.2 | 1.5 |

The live system was systematically under-predicting probabilities, causing every runner to look underpriced and generating false edges on longshots.

### Root cause investigation
After ruling out normalization bugs, missing features, and model loading issues, we found the **root cause**: a mismatch in how `time_delta` is computed between the historical feature engineering pipeline and the live trading system.

---

## 2. The Bug: time_delta Reference Mismatch

### How time_delta works
Features are computed at specific time points relative to a reference. For example, `best_back_m0` is the best back price at time `t0` before the reference, `best_lay_m3` is at `tm3`, etc. The reference point anchors the entire feature extraction.

### Historical pipeline (training data)
File: `_03_feature_engeneering/_01_feature_engineering_para.py`, line 36:
```python
df['time_delta'] = df.groupby('file_name')['time'].transform('max') - df['time']
```
This uses `max(time)` — the **last tick recorded for that race** — as the reference. We measured this on Spartan and found:

```
offset = max(time) - scheduledStart:
  median = +42 seconds (AFTER scheduled start)
  P25 = +26 seconds
  P75 = +58 seconds
```

So the reference point is typically **42 seconds after the race starts** — deep into in-running trading.

### Live pipeline (paper trading)
File: `_05_paper_trading/feature_engine.py`, lines 337-341:
```python
scheduled_start = pd.Timestamp(market_cache.market_start_time, tz="UTC")
tick_df["time_delta"] = scheduled_start - tick_df["time"]
tick_df["time_delta"] = tick_df["time_delta"].clip(lower=pd.Timedelta(0))
```
This uses `scheduled_start` as the reference — **before the race starts**.

### Impact on each time definition

| t_def | t0 (entry) | Historical features from (with +42s offset) | Live features from | Pre-race or in-running? |
|-------|-----------|----------------------------------------------|-------------------|------------------------|
| 0 | 60s | scheduled_start **- 18s** | scheduled_start **- 60s** | Historical: pre-race (barely). Live: pre-race |
| 1 | 120s | scheduled_start **- 78s** | scheduled_start **- 120s** | Both pre-race |
| 2 | 30s | scheduled_start **+ 12s** | scheduled_start **- 30s** | **Historical: IN-RUNNING.** Live: pre-race |
| 3 | 20s | scheduled_start **+ 22s** | scheduled_start **- 20s** | **Historical: IN-RUNNING.** Live: pre-race |

**Key insight**: t_def 2 and 3 models were trained on data from AFTER the race started (in-running odds), but the live system fed them data from BEFORE the race started. This is lookahead bias for a pre-race strategy.

### Why this matters for the fix
Using `scheduled_start` as the fix would make all features purely pre-race. But **races don't all start at the same time relative to scheduled start** — some are delayed by 30s, some by 2 minutes. The liquidity dynamics that the features capture are driven by **proximity to the actual race start** (the "off"), not the clock. A snapshot 30 seconds before the off looks completely different from 3 minutes before, regardless of what the scheduled time says.

The only reference point that captures the same market regime consistently is the **actual in-play transition** — the moment the race actually starts and the market goes in-play.

---

## 3. The Fix: Use In-Play Transition as Reference

### Design
Replace `max(time)` (historical) and `scheduled_start` (live) with **`in_play_time`** — the timestamp when the market's `inPlay` flag flips to `true` in Betfair streaming data.

This is available in both contexts:
- **Historical**: The raw BZ2 files contain `marketDefinition` changes including `inPlay: true`. Currently the processing pipeline (`process_races.py:176-178`) only extracts the **first** market definition and discards all subsequent updates. We need to scan ALL market definitions to find the in-play transition.
- **Live**: Betfair streaming sends `marketDefinition` updates. The `market_cache.py` code already processes these but only checks for `CLOSED`/`COMPLETE` status (line 192). We need to add detection of `inPlay: true`.

### What changes

**Feature engineering (historical)**:
```python
# OLD (line 36 of _01_feature_engineering_para.py):
df['time_delta'] = df.groupby('file_name')['time'].transform('max') - df['time']

# NEW:
df['time_delta'] = df['in_play_time'] - df['time']
# Then clamp: negative time_deltas mean ticks happened after the race started
df['time_delta'] = df['time_delta'].clip(lower=pd.Timedelta(0))
```

**Feature engine (live)**:
```python
# OLD (line 338 of feature_engine.py):
tick_df["time_delta"] = scheduled_start - tick_df["time"]

# NEW:
tick_df["time_delta"] = in_play_time - tick_df["time"]
tick_df["time_delta"] = tick_df["time_delta"].clip(lower=pd.Timedelta(0))
```

With this change, **all features become genuinely pre-race** for all t_defs. A `time_delta` of 30 seconds means "30 seconds before the race actually started" in both training and live.

### Important consequence for t_def design
With in-play as the reference, the current time definitions still make sense:
- t0=60s, t0=120s, t0=30s, t0=20s all refer to seconds before the actual start
- All features are from pre-race market microstructure
- No more lookahead bias from in-running data

The tp1 (target) time point also shifts. With the old reference:
- t_def 2: tp1 = 10s before max(time) = ~32s after start (in-running)
- t_def 3: tp1 = 5s before max(time) = ~37s after start (in-running)

With in-play as reference:
- t_def 2: tp1 = 10s before in-play = 10s before race starts (pre-race)
- t_def 3: tp1 = 5s before in-play = 5s before race starts (pre-race)

This is actually better — tp1 captures the price very close to the off, which is the most informationally rich pre-race moment.

---

## 4. Step-by-Step Implementation Plan

### Overview of pipeline stages to re-run:
```
Step 0: Extract in-play timestamps from raw BZ2 files
Step 1: Modify feature engineering to use in-play reference
Step 2: Re-run feature engineering (4 SLURM arrays × 10 jobs each = 40 jobs)
Step 3: Merge & validate features
Step 4: Retrain V1 models (reuse existing SLURM configs, existing model output dirs WILL be overwritten)
Step 5: Retrain V2 models (same)
Step 6: Build cross-t super-ensemble
Step 7: (Optional) Re-run fill simulation
Step 8: Save model artifacts for live trading
Step 9: Update live trading code
Step 10: Download artifacts & redeploy
```

**Estimated total time**: ~2-3 days of SLURM queue time. Steps 0-3 take ~6 hours. Steps 4-5 are the bottleneck (~1-2 days for the V1 grid search with 2376 combos).

---

### Step 0: Extract In-Play Timestamps from Raw BZ2 Files

**Goal**: For each market (identified by `file_name`), find the exact timestamp when `inPlay` becomes `true` in the raw streaming data. Store as a lookup table.

**Create a new script**: `_01_process_files/_04_extract_in_play_times.py`

This script needs to:
1. Iterate through all raw BZ2 files in `/data/projects/punim2039/alpha_odds/untar/greyhound_au/PRO/`
2. For each file, parse the JSON lines
3. Find `marketDefinition` entries where `inPlay` changes to `true`
4. Record `(file_name, in_play_time)` — the `pt` (publish time) of that entry
5. Save as a single parquet: `/data/projects/punim2039/alpha_odds/data/p/greyhound_au/in_play_times.parquet`

**Key details about the raw data format**:
The BZ2 files contain JSON lines. Each line has:
- `pt`: publish time in milliseconds since epoch
- `mc`: list of market changes, each containing either:
  - `marketDefinition`: dict with keys including `inPlay` (boolean), `status`, `runners`, etc.
  - `rc`: list of runner changes (price/volume updates)

From `process_races.py`, line 176:
```python
ind_market_def = df_full['mc'].apply(lambda x: 'marketDefinition' in x[0].keys())
m_def = df_full.loc[ind_market_def, 'mc'].apply(lambda x: x[0]['marketDefinition'])
```

So `marketDefinition` appears as a key inside `mc[0]`. To find the in-play transition:
```python
for idx, row in df_full.iterrows():
    mc = row['mc'][0]
    if 'marketDefinition' in mc:
        mdef = mc['marketDefinition']
        if mdef.get('inPlay', False) == True:
            in_play_time = row['pt']  # This is already a datetime after pd.to_datetime(df_full["pt"], unit="ms")
            break
```

**Important**: Some markets may never go in-play (cancelled, void, etc.). For these, `in_play_time` should be `NaT`. Markets without in-play times will be excluded from feature engineering.

**SLURM**: This can reuse the same parallel structure as `_02_process_all_files_para.py` (array 0-323, same grid of year/month/day). Or it can be a simpler script since it only reads the BZ2 files and extracts one timestamp per file (much lighter than full processing).

**Output schema**:
```
file_name (str): e.g., "1.248460630.bz2"
in_play_time (datetime64[ns, UTC]): when inPlay became true
```

One parquet file per (year, month, day_para_number), named `inplay_{year}_{month}_{day}.parquet`, following the existing naming convention. Then merge into a single `in_play_times_all.parquet`.

---

### Step 1: Modify Feature Engineering

**File to modify**: `_03_feature_engeneering/_01_feature_engineering_para.py`

**Changes**:

1. **Load in-play lookup** at the top of `compute_features()`:
```python
def compute_features(paths, t_definition=0):
    df = pd.read_parquet(paths)
    mdef = pd.read_parquet(paths.replace('win_', 'mdef_'))

    # Load in-play times for this day partition
    inplay_path = paths.replace('win_', 'inplay_')
    if os.path.exists(inplay_path):
        inplay_df = pd.read_parquet(inplay_path)
    else:
        # Fallback: try the merged file
        inplay_path_all = os.path.join(os.path.dirname(paths), 'in_play_times_all.parquet')
        inplay_df = pd.read_parquet(inplay_path_all)

    # ... existing code ...
```

2. **Replace time_delta computation** (currently line 36):
```python
# OLD:
# df['time_delta'] = df.groupby('file_name')['time'].transform('max') - df['time']

# NEW:
df = df.merge(inplay_df[['file_name', 'in_play_time']], on='file_name', how='left')
# Drop races with no in-play time (cancelled/void markets)
df = df.dropna(subset=['in_play_time'])
df['time_delta'] = df['in_play_time'] - df['time']
# Clamp: only use pre-race data (time_delta >= 0 means before in-play)
df['time_delta'] = df['time_delta'].clip(lower=pd.Timedelta(0))
df = df.drop(columns=['in_play_time'])
```

3. **No other changes needed** in the feature engineering code. The `add_time_snapshot()` function, momentum calculations, std windows, etc., all work on `time_delta` and don't need modification.

**Important**: Save the modified script, deploy to Spartan with `bash scripts/sh/code_to_spartan.sh`, then run.

---

### Step 2: Re-Run Feature Engineering (SLURM)

Use the existing SLURM scripts. The feature engineering scripts take `(chunk_id, t_definition)` as arguments.

```bash
# Deploy code first
bash scripts/sh/code_to_spartan.sh

# Submit for all 4 time definitions
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _01_feature_engineering_para_t0.slurm"
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _01_feature_engineering_para_t1.slurm"
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _01_feature_engineering_para_v1.slurm"  # This is t2
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _01_feature_engineering_para_v2.slurm"  # This is t3
```

Each SLURM script runs array 0-9, 64G RAM, 8 CPUs, ~2 day time limit.

**Output**: `res/features_t{0,1,2,3}/greyhound_au_features_part_{0-9}.parquet`

**WARNING**: This will **overwrite** the existing feature parquets. The old features (with the `max(time)` reference) will be gone. This is intentional — we want to replace them.

---

### Step 3: Merge & Validate Features

After ALL 40 jobs complete (verify with `squeue -u adidishe`):

```bash
# Run merge for each t_def (can use srun since they're quick)
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && srun --partition=interactive --time=00:30:00 --mem=32G bash -c 'source load_module.sh && python3 _00_merge_features.py 0 0'"
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && srun --partition=interactive --time=00:30:00 --mem=32G bash -c 'source load_module.sh && python3 _00_merge_features.py 1 0'"
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && srun --partition=interactive --time=00:30:00 --mem=32G bash -c 'source load_module.sh && python3 _00_merge_features.py 2 0'"
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && srun --partition=interactive --time=00:30:00 --mem=32G bash -c 'source load_module.sh && python3 _00_merge_features.py 3 0'"
```

**Validation checks** (the merge script does these automatically):
- All 10 parts exist
- Column counts consistent
- No `_x`/`_y` duplicate columns
- Win rate ~12.5% (for 8-runner races)
- Row count should be similar to (but may be slightly different from) the old features, because some markets without in-play data will be dropped

**CRITICAL**: Do NOT proceed to model training until merge/validation passes for ALL 4 t_defs.

---

### Step 4: Retrain V1 Models

V1 models are trained by `_04_first_run_and_analysis/_01_run.py` with a massive grid search.

The relevant SLURM scripts:
- `_01_run_0.slurm`: array 0-11, model_type=0
- `_01_run_1.slurm`: array 0-17, model_type=1
- `_01_run_2.slurm`: array 0-2375, model_type=2 (the big one)

```bash
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _01_run_0.slurm"
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _01_run_1.slurm"
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _01_run_2.slurm"
```

**Output**: `res/win_model/t{0,1,2,3}/{config_hash}/save_df.parquet` per grid combination.

**Note**: `_01_run_2.slurm` has 2376 array jobs. This is the longest step. Each job takes ~15-60 minutes, 16G RAM, 4 CPUs.

---

### Step 5: Retrain V2 Models

V2 models use `_04_first_run_and_analysis/_03_win_probability_model_v2.py`. The V2 SLURM script encodes both grid_id and t_def in the array index:

```bash
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _03_win_probability_model_v2.slurm"
```

This runs array 0-107 (27 grid combos x 4 t_defs), 64G RAM, 8 CPUs, 6 hour time limit.

Also retrain V1 with the V2-style normalizer (used for the ensemble):
```bash
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _02_win_probability_model.slurm"
```

Array 0-107 (27 grid combos x 4 t_defs), same resources.

**Output**: `res/win_model_v2/t{0,1,2,3}/{config_hash}/save_df.parquet` and `res/win_model/t{0,1,2,3}/{config_hash}/save_df.parquet`.

---

### Step 6: Build Cross-T Super-Ensemble

After ALL model training jobs complete:

```bash
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && srun --partition=interactive --time=01:00:00 --mem=64G bash -c 'source load_module.sh && python3 _10_ultimate_ensemble.py'"
```

This script:
1. Loads all V1 and V2 models for all t_defs
2. Selects top-7 (V1) and top-15 (V2) by log-loss
3. Builds cross-t averages
4. Optimizes V1/V2 blend weights
5. Runs backtest at various edge thresholds
6. Saves `res/analysis/ultimate_cross_t_ensemble_predictions.parquet`

**Key metrics to compare against the old results**:
- Old: LL=0.316321, 2010 bets at edge>3%, +19% ROI, Sharpe 3.6
- New: should be different (features computed differently), hopefully still strong

**Note**: With the new in-play reference, t2/t3 will have genuinely pre-race features. The ensemble may be better or worse — the in-running information that t2/t3 previously exploited is gone, but the features are now actually replicable in live trading.

---

### Step 7: (Optional) Re-Run Fill Simulation

If the backtest results look good, re-run the fill simulation:

```bash
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _13_fill_simulator.slurm"
# After all 30 tasks complete:
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && srun --partition=interactive --time=00:30:00 --mem=32G bash -c 'source load_module.sh && python3 _14_fill_merge.py'"
# Generate report:
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && srun --partition=interactive --time=00:30:00 --mem=32G bash -c 'source load_module.sh && module load matplotlib/3.5.2 && python3 _15_fill_analysis.py'"
```

---

### Step 8: Save Model Artifacts for Live Trading

The live system needs pickled models, normalization params, and isotonic calibrators. The existing script handles this:

```bash
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch _save_model_artifacts.slurm"
```

This runs `save_artifacts_resume.py` which:
1. Ranks all configs by log-loss
2. Saves top-N model references to `manifest.json`
3. Fits and saves isotonic calibrators for V2 models
4. Extracts and saves normalization parameters

**Output**: `res/paper_trading_artifacts/` containing:
- `manifest.json` (which configs to use per t_def)
- `normalization/feature_normalization_params_{v1,v2}_t{0,1,2,3}.parquet`
- Isotonic calibrators saved in each V2 model directory

---

### Step 9: Update Live Trading Code

#### 9a. Add in-play detection to `market_cache.py`

In `MarketCache.process_market_change()`, add detection for in-play transition:

```python
class MarketCache:
    def __init__(self, ...):
        # ... existing init ...
        self.in_play_time = None  # NEW: when market went in-play

    def process_market_change(self, market_book):
        timestamp = datetime.now(timezone.utc)

        if hasattr(market_book, "market_definition") and market_book.market_definition:
            md = market_book.market_definition
            # NEW: detect in-play transition
            if hasattr(md, "in_play") and md.in_play and self.in_play_time is None:
                self.in_play_time = timestamp
                logger.info(f"Market {self.market_id}: went IN-PLAY at {timestamp}")

            if hasattr(md, "status") and md.status in ("CLOSED", "COMPLETE"):
                self.is_settled = True
        # ... rest unchanged ...
```

**Note**: The exact attribute name may be `in_play` or `inPlay` depending on how betfairlightweight exposes it. Check the betfairlightweight MarketDefinition class. In betfairlightweight, it's typically `market_definition.in_play` (snake_case).

#### 9b. Update `feature_engine.py` to use in-play time

In `FeatureComputer.compute()`, replace the time_delta calculation:

```python
def compute(self, market_cache):
    tick_df = market_cache.to_dataframe()
    if tick_df.empty:
        return {}

    # NEW: use in-play time as reference (matches historical pipeline)
    if market_cache.in_play_time is not None:
        reference_time = pd.Timestamp(market_cache.in_play_time, tz="UTC")
    else:
        # Fallback to scheduled start if in-play not detected yet
        logger.warning(f"No in-play time for {market_cache.market_id}, using scheduled start")
        reference_time = pd.Timestamp(market_cache.market_start_time, tz="UTC")

    tick_df["time_delta"] = reference_time - tick_df["time"]
    tick_df["time_delta"] = tick_df["time_delta"].clip(lower=pd.Timedelta(0))

    # ... rest unchanged ...
```

#### 9c. Update decision timing in `main.py`

Currently the decision is made at `scheduled_start - DECISION_SECONDS_BEFORE_START` (20s before scheduled start). With the new reference, we should make the decision **after the race goes in-play**, because we now need the in-play time as our reference.

**Option A (recommended)**: Keep the pre-race decision timing but use scheduled_start as a fallback for the reference. Then add a **second decision point** right after in-play is detected — recompute features with the correct reference and update/place bets. This is more complex but more accurate.

**Option B (simpler)**: Make the decision shortly after in-play (e.g., in_play_time + 2 seconds). This gives us the correct reference point. The bet would be placed in-running, but with only pre-race features (since we clamp time_delta >= 0). Betfair allows in-play betting on greyhound markets.

**Option C (simplest for now)**: Keep the decision at scheduled_start - 20s, use scheduled_start as the reference in live (NOT in-play time). Accept that there's some noise from races not starting on schedule. This avoids the complexity of in-play detection but is less accurate than the retrained models expect.

**Recommendation**: Start with Option C for initial testing (it's a huge improvement over the current broken state), then implement Option B for production accuracy. The median offset between scheduled_start and in_play is ~42s, but the median offset between scheduled_start and the ACTUAL start is probably much smaller (most of that 42s is post-race ticks, not late starts). Need to verify.

Actually, the key question is: **how close is `in_play_time` to `scheduled_start`?** If they're typically within 5-10 seconds, then using scheduled_start is fine. The Step 0 in-play extraction will give us data to answer this.

---

### Step 10: Download Artifacts & Redeploy

After Step 8 completes, download the new artifacts to the local machine:

```bash
# Download new model artifacts
scp -r adidishe@spartan.hpc.unimelb.edu.au:/data/projects/punim2039/alpha_odds/res/paper_trading_artifacts/ ./models_new/

# Download new ensemble predictions (for updated baselines)
scp adidishe@spartan.hpc.unimelb.edu.au:/data/projects/punim2039/alpha_odds/res/analysis/ultimate_cross_t_ensemble_predictions.parquet ./res/analysis/

# Replace old models
mv models models_old
mv models_new models
```

Then update `config.py` baselines with new backtest metrics, and redeploy the paper trading system.

---

## 5. Key Files Reference

### Historical Pipeline (Spartan: `/home/adidishe/alpha_odds/`, local: project root)
| File | Purpose | Changes Needed |
|------|---------|---------------|
| `_01_process_files/_02_process_all_files_para.py` | Raw BZ2 -> processed parquets | None (reuse as-is) |
| **`_01_process_files/_04_extract_in_play_times.py`** | **NEW: Extract in-play timestamps** | **CREATE THIS** |
| `utils_locals/process_races.py` | Core data processing functions | None |
| `_03_feature_engeneering/_01_feature_engineering_para.py` | Feature engineering (THE CRITICAL FILE) | **Change line 36: use in_play_time instead of max(time)** |
| `_04_first_run_and_analysis/_00_merge_features.py` | Merge feature parts | None |
| `_04_first_run_and_analysis/_01_run.py` | V1 XGBoost grid search | None |
| `_04_first_run_and_analysis/_02_win_probability_model.py` | V1 with V2-style normalizer | None |
| `_04_first_run_and_analysis/_03_win_probability_model_v2.py` | V2 XGB+LightGBM+isotonic | None |
| `_04_first_run_and_analysis/_10_ultimate_ensemble.py` | Cross-t super-ensemble | None |
| `_05_paper_trading/scripts/save_artifacts_resume.py` | Save artifacts for live | None |

### Live Trading (local: `_05_paper_trading/`)
| File | Purpose | Changes Needed |
|------|---------|---------------|
| `_05_paper_trading/market_cache.py` | Order book state from streaming | **Add `in_play_time` detection** |
| `_05_paper_trading/feature_engine.py` | Live feature computation | **Change time_delta to use in_play_time** |
| `_05_paper_trading/main.py` | Main orchestrator | **Update decision timing logic** |
| `_05_paper_trading/config.py` | Configuration & baselines | **Update BACKTEST_BASELINES** after retraining |

### Data on Spartan
| Path | Content |
|------|---------|
| `/data/projects/punim2039/alpha_odds/untar/greyhound_au/PRO/` | Raw BZ2 files (source for Step 0) |
| `/data/projects/punim2039/alpha_odds/data/p/greyhound_au/` | Processed parquets (win_, mdef_, inplay_) |
| `/data/projects/punim2039/alpha_odds/res/features_t{0,1,2,3}/` | Feature parquets (will be overwritten) |
| `/data/projects/punim2039/alpha_odds/res/win_model/` | V1 models (will be overwritten) |
| `/data/projects/punim2039/alpha_odds/res/win_model_v2/` | V2 models (will be overwritten) |
| `/data/projects/punim2039/alpha_odds/res/analysis/` | Ensemble predictions |
| `/data/projects/punim2039/alpha_odds/res/paper_trading_artifacts/` | Artifacts for live trading |

---

## 6. SLURM Configuration Reference

### Module stack (all scripts use the same):
```bash
module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source ~/venvs/alpha_odds_venv/bin/activate
```

Or simply: `source load_module.sh`

### Job configurations:
| Step | Script | Array | RAM | CPUs | Time |
|------|--------|-------|-----|------|------|
| 0 | New: `_04_extract_in_play_times.slurm` | 0-323 | 16G | 2 | 4h |
| 2 | `_01_feature_engineering_para_t{0,1}.slurm`, `_v{1,2}.slurm` | 0-9 each | 64G | 8 | 2d |
| 4 | `_01_run_0.slurm` | 0-11 | 16G | 4 | 2d |
| 4 | `_01_run_1.slurm` | 0-17 | 16G | 4 | 2d |
| 4 | `_01_run_2.slurm` | 0-2375 | 16G | 4 | 2d |
| 5 | `_02_win_probability_model.slurm` | 0-107 | 64G | 8 | 6h |
| 5 | `_03_win_probability_model_v2.slurm` | 0-107 | 64G | 8 | 6h |
| 8 | `_save_model_artifacts.slurm` | single | 64G | 4 | 2h |

### Code deployment:
```bash
# From local machine (project root):
bash scripts/sh/code_to_spartan.sh
# This copies all .py files flat to /home/adidishe/alpha_odds/ on Spartan
```

### Monitoring:
```bash
ssh adidishe@spartan.hpc.unimelb.edu.au "squeue -u adidishe"
ssh adidishe@spartan.hpc.unimelb.edu.au "tail -50 /home/adidishe/alpha_odds/out/<jobname>_<arrayid>.out"
```

---

## 7. Safety Rules & Gotchas

### NEVER delete anything under `/data/projects/punim2039/`
This includes the raw data, processed data, features, models, and results. The pipeline will **overwrite** files in place (e.g., new features replace old features), but NEVER delete entire directories.

### Always validate before proceeding
After each SLURM batch completes:
1. Check `squeue -u adidishe` shows no remaining tasks
2. Run the merge/validation step
3. Check output for errors, correct row counts, sensible win rates
4. Only then submit the next stage

### Only one `srun` at a time
The interactive partition allows only one `srun` session per user. If one is running, use `sbatch` instead. Check first with `squeue -u adidishe`.

### The feature engineering overwrites existing features
The re-run will replace `res/features_t{0,1,2,3}/greyhound_au_features_part_*.parquet`. The old features (with `max(time)` reference) will be gone. This is expected and intended.

### Model retraining also overwrites
Similarly, model training will overwrite `res/win_model/` and `res/win_model_v2/` directories if the same config hashes are used. Since we're using the same grid, the hashes will be the same.

### Local Python environment has issues
Don't try to run ML code locally (numpy/pandas/scipy compatibility problems). All pipeline code should run on Spartan.

### Code deployed flat
When deploying with `code_to_spartan.sh`, all Python files from `_XX_` subdirectories are copied flat to `/home/adidishe/alpha_odds/`. The subdirectory organization is local-only.

### No scipy on Spartan venv
Use `from math import erf, sqrt` and `0.5*(1+erf(x/sqrt(2)))` for norm_cdf instead of `scipy.stats.norm.cdf`.

### betfairlightweight in-play attribute
In the betfairlightweight library, the market definition's in-play status is accessed as `market_definition.in_play` (Python snake_case, not camelCase). Double-check the exact attribute name in the library source if needed.

### Step 0 may reveal that some races never went in-play
Historical Betfair data may include markets that were cancelled, voided, or never turned in-play. These should be excluded from feature engineering (they'd have `in_play_time = NaT`). Check how many are affected — should be a small fraction.

### The `in_play_time` reference will change the row count of features
With `max(time)`, every race had a reference. With `in_play_time`, races without in-play data get dropped. This may reduce the training set slightly. Monitor this in Step 3 (merge validation).

### After retraining, update `config.py` baselines
The BACKTEST_BASELINES in `_05_paper_trading/config.py` were derived from the old backtest. After Step 6, extract new metrics from the ensemble output and update:
- `bets_per_day`, `avg_edge`, `avg_odds`, `avg_model_prob`, `avg_market_prob`, `roi_pct`, `win_rate`

---

## Appendix: Quick Sanity Check After Step 0

After extracting in-play times, run a quick sanity check on Spartan:

```python
import pandas as pd
import os

base = '/data/projects/punim2039/alpha_odds/data/p/greyhound_au/'
inplay_files = sorted([f for f in os.listdir(base) if f.startswith('inplay_')])
print(f"Found {len(inplay_files)} in-play files")

# Load a sample
df = pd.read_parquet(base + inplay_files[0])
print(f"Sample: {df.shape}")
print(f"NaT count: {df['in_play_time'].isna().sum()} / {len(df)}")

# Compare to mdef scheduled start
mdef_file = inplay_files[0].replace('inplay_', 'mdef_')
mdef = pd.read_parquet(base + mdef_file)
mdef['marketTime'] = pd.to_datetime(mdef['marketTime'], utc=True)

mg = df.merge(mdef[['file_name', 'marketTime']].drop_duplicates(), on='file_name')
mg['offset_s'] = (mg['in_play_time'] - mg['marketTime']).dt.total_seconds()
print(f"\nin_play_time - scheduledStart (seconds):")
print(mg['offset_s'].describe())
```

This tells us how close `in_play_time` is to `scheduled_start`. If the median offset is small (< 10s), then `scheduled_start` is a decent fallback for live trading (Step 9, Option C).
