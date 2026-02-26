# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL SAFETY RULES

**NEVER delete anything under `/data/projects/punim2039/` on Spartan. Not alpha_odds, not anything else. This is irreplaceable research data. No `rm`, no overwrite, no truncation. This rule has no exceptions.**

It is OK to delete things in `/home/adidishe/` (scratch/home) on Spartan.

## Project Overview

Quantitative trading research project using Betfair exchange data. ML-based strategies predict win probabilities in greyhound racing markets, identifying underpriced runners for value betting with limit orders.

**Two phases:**
1. **R&D** — Historical data analysis on Spartan HPC: process raw Betfair streaming data, engineer features, train models, backtest strategies
2. **Live trading** — Deploy profitable strategies on AWS with Betfair API (future work)

The pipeline processes raw BZ2-compressed JSON streaming data through feature engineering to model training (XGBoost, LightGBM with isotonic calibration).

## Project Status (Feb 2026)

**R&D phase is COMPLETE.** All pipeline stages (data processing → feature engineering → model training → ensembling → backtesting → fill simulation) have been run and validated.

**Completed:**
1. ~~Data pipeline~~ — Raw BZ2 → processed parquets → features across 4 time definitions
2. ~~Bug fixes~~ — ffill cross-contamination, duplicate merges, breakpoints. No lookahead bias.
3. ~~Strategy selection~~ — Value betting with limit orders
4. ~~V1 models~~ — XGBoost grid search, top-7 ensemble per t_def
5. ~~V2 models~~ — XGBoost + LightGBM + isotonic calibration, top-15 ensemble per t_def
6. ~~Cross-t Super-Ensemble~~ — V1(20%) + V2(80%) averaged across t0-t3 (best model)
7. ~~Fill rate simulation~~ — 97% conservative fill rate confirmed via tick replay
8. ~~Reports~~ — Model report (`res/alpha_odds_report.pdf`) + fill report (`res/fill_simulation/fill_analysis_report.pdf`)

**Next:** Paper trading → live testing with small stakes → scale up (see `NEXT_STEPS.md`)

### Best Model: Cross-t Super-Ensemble
- **Architecture**: V1_cross(20%) + V2_cross(80%), 88 model components
- **OOS (edge>3%)**: 2,010 bets, +19% ROI, Sharpe 3.6, z=3.90, p=0.000048, $9,440 at $25/bet
- **Fill-adjusted**: 97% conservative fill, +17% ROI, $8,451 at $25/bet (retains ~90% of profit)
- **Log-loss**: 0.316321 vs Market 0.319722

## Spartan HPC Operations

### Connection
```bash
ssh adidishe@spartan.hpc.unimelb.edu.au    # Passwordless SSH works from this machine
```

### Three Ways to Run Code on Spartan
| Method | Command | Use for | ML libs available? |
|--------|---------|---------|-------------------|
| **Login node** | `ssh adidishe@spartan... "<cmd>"` | `ls`, `du`, `cat`, file ops | NO (bare Python 3.9) |
| **`srun` (interactive)** | `ssh ... "cd /home/adidishe/alpha_odds && srun --partition=interactive --time=00:10:00 --mem=16G bash -c 'source load_module.sh && python3 ...'"` | Quick debugging, data inspection, short analyses | YES |
| **`sbatch` (batch)** | `ssh ... "cd /home/adidishe/alpha_odds && sbatch script.slurm"` | Long-running jobs, parallelized processing | YES |

`srun` typically queues for 5-15 seconds then runs. For heavier interactive work, increase `--mem` and `--time`.
`load_module.sh` loads the full module stack + activates the venv.

**Important:** Only one `srun` session can run at a time. Before using `srun`, check if one is already active (`squeue -u adidishe`). If an `srun` job is already running, write a small `.py` script and submit it via `sbatch` instead.

### Directory Layout on Spartan
| Path | Purpose | Can delete? |
|------|---------|-------------|
| `/home/adidishe/alpha_odds/` | Code + SLURM scripts (flat structure) | YES |
| `/home/adidishe/alpha_odds/out/` | SLURM stdout/stderr logs | YES |
| `/data/projects/punim2039/alpha_odds/data/` | Raw & processed data (117GB) | **NEVER** |
| `/data/projects/punim2039/alpha_odds/res/` | Results, models, features, reports | **NEVER** |

### Deploying Code
```bash
bash scripts/sh/code_to_spartan.sh   # scp all .py files + slurm scripts to Spartan
```
**Important:** On the server, all Python files from `_XX_` subdirectories are copied flat to `/home/adidishe/alpha_odds/`. The subdirectory organization is local-only for development.

### Submitting SLURM Jobs
```bash
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && sbatch <script>.slurm"
```

### Running Interactive Python on Spartan
The login node has bare Python 3.9 without pyarrow/pandas — **only use it for bash commands** (ls, du, file ops). For anything needing the ML stack, use `srun`:
```bash
# Quick interactive Python on a compute node (queues briefly, then runs)
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && srun --partition=interactive --time=00:10:00 --cpus-per-task=2 --mem=16G bash -c 'source load_module.sh && python3 -c \"<code>\"'"

# For longer scripts, write to a .py file first, then:
ssh adidishe@spartan.hpc.unimelb.edu.au "cd /home/adidishe/alpha_odds && srun --partition=interactive --time=00:30:00 --cpus-per-task=4 --mem=32G bash -c 'source load_module.sh && python3 my_script.py'"
```

### Monitoring Jobs
```bash
ssh adidishe@spartan.hpc.unimelb.edu.au "squeue -u adidishe"              # List running jobs
ssh adidishe@spartan.hpc.unimelb.edu.au "tail -50 /home/adidishe/alpha_odds/out/<job>.out"  # Check output
ssh adidishe@spartan.hpc.unimelb.edu.au "scancel <job_id>"                # Cancel a job
```

### Downloading Results
```bash
scp adidishe@spartan.hpc.unimelb.edu.au:/data/projects/punim2039/alpha_odds/res/<file> ./res/
scp -r adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/alpha_odds/out/ ./spartan_logs/
```

### SLURM Module Stack
All SLURM scripts load: `foss/2022a`, `GCCcore/11.3.0`, `Python/3.10.4`, `cuDNN/8.4.1.50-CUDA-11.7.0`, `TensorFlow/2.11.0`, `PyTorch/1.12.1`. Venv: `~/venvs/alpha_odds_venv/bin/activate`.

For matplotlib (needed by report scripts): also `module load matplotlib/3.5.2`.

### SLURM Configurations
| Script | Array | RAM | CPUs | What it runs |
|--------|-------|-----|------|-------------|
| `_02_process_all_files_para.slurm` | 0-323 | 16G | 2 | Raw data processing |
| `_01_feature_engineering_para_v1.slurm` | 0-9 | 64G | 8 | Feature engineering, t_definition=2 |
| `_01_feature_engineering_para_v2.slurm` | 0-9 | 64G | 8 | Feature engineering, t_definition=3 |
| `_01_feature_engineering_para_t0.slurm` | 0-9 | 64G | 8 | Feature engineering, t_definition=0 |
| `_01_feature_engineering_para_t1.slurm` | 0-9 | 64G | 8 | Feature engineering, t_definition=1 |
| `_01_run_0.slurm` | 0-11 | 16G | 4 | V1 model training, model_type=0 |
| `_01_run_1.slurm` | 0-17 | 16G | 4 | V1 model training, model_type=1 |
| `_01_run_2.slurm` | 0-2375 | 16G | 4 | V1 model training, model_type=2 (full grid) |
| `_02_win_probability_model.slurm` | varies | 16G | 4 | V2 win probability model |
| `_03_win_probability_model_v2.slurm` | varies | 16G | 4 | V2 with LightGBM + isotonic |
| `_13_fill_simulator.slurm` | 0-29 | 16G | 2 | Fill rate simulation (30 tasks) |

### Data on Spartan
- **Raw processed:** `data/p/greyhound_au/` — `win_`, `place_`, `mdef_` parquets per (year, month, day_partition)
  - 305 day-files for 2025 OOS period
  - Tick data with `time` index (datetime), `best_back`, `best_lay`, `prc`, `qty`, `order_type`, `id`, `file_name`
- **Features:** `res/features_t{0,1,2,3}/` — 10 parquet parts each + merged
- **V1 models:** `res/win_model/t{0,1,2,3}/` — grid search configs with `save_df.parquet`
- **V2 models:** `res/win_model_v2/t{0,1,2,3}/` — XGBoost+LightGBM+isotonic configs
- **Ensemble predictions:** `res/analysis/ultimate_cross_t_ensemble_predictions.parquet` (285K rows)
- **Fill simulation:** `res/fill_simulation/fill_results_merged.parquet` (665K rows, 110K runners × 6 scenarios)
- **Reports:** `res/alpha_odds_report.pdf`, `res/fill_simulation/fill_analysis_report.pdf`

## CRITICAL WORKFLOW RULE: Always Validate Before Proceeding

**Before running ANY downstream step (model training, analysis, backtesting), ALWAYS:**
1. Verify ALL upstream SLURM jobs completed successfully (`squeue -u adidishe` shows no remaining tasks)
2. Run the merge/validation script for that stage (e.g., `_00_merge_features.py` for features, `_14_fill_merge.py` for fill simulation)
3. Check the validation output: correct row/column counts, no `_x`/`_y` duplicates, sane win rates
4. Only then submit the next stage's jobs

**Never assume jobs finished — always check. Never skip the merge step.**

## Running the Pipeline

Activate local environment: `source venv_comp/bin/activate`

Stages run sequentially (all scripts within a `_XX_` folder in order, then the next folder):

### Stage 1: `_01_process_files/` — Raw Data Processing
```bash
python _01_untar_files.py                          # Untar raw Betfair files
python _02_process_all_files_para.py <job_id> 1    # Process files (job_id: 0-323)
python _03_merge_files.py                          # Merge outputs after ALL _02 jobs complete
```

### Stage 2: `_02_summary_stats/` — Summary Statistics
```bash
python _01_summary_stats_quick.py
```

### Stage 3: `_03_feature_engeneering/` — Feature Engineering
```bash
python _01_feature_engineering_para.py <chunk_id> <t_definition>  # chunk_id: 0-9, t_definition: 0-3
```

### Stage 3.5: Merge & Validate Features (MANDATORY before Stage 4)
```bash
python _00_merge_features.py <t_definition> 0    # t_definition: 0, 1, 2, or 3
```

### Stage 4: `_04_first_run_and_analysis/` — Model Training & Analysis
```bash
python _01_run.py <grid_comb_id> <model_type>                     # V1 XGBoost grid search
python _02_win_probability_model.py <grid_comb_id> <t_definition> # V2 win probability
python _03_win_probability_model_v2.py <grid_comb_id> <t_def>     # V2 with LightGBM + isotonic
python _10_ultimate_ensemble.py                                    # Cross-t super-ensemble (no args)
```

### Stage 5: Fill Rate Simulation
```bash
python _13_fill_simulator.py <task_id>    # task_id: 0-29 (SLURM array)
python _14_fill_merge.py                  # Merge all 30 parts (no args)
python _15_fill_analysis.py               # Generate fill analysis PDF (no args, needs matplotlib)
```

### Reports (run on Spartan with matplotlib loaded)
```bash
python _05_generate_report.py    # Main model report
python _15_fill_analysis.py      # Fill simulation report
```

## Architecture

### Argument Parsing
Custom parser (`utils_locals/parser.py`), not argparse. Positional args map to `args.a`, `args.b`, `args.c`. Named args use `--key=value`. First positional arg (`args.a`) is typically the SLURM array task ID.

### Parameter System (`parameters.py`)
- `Params` object holds nested sub-objects: `data`, `grid`, `model`
- `update_param_grid(grid_list, id_comb)` iterates over all hyperparameter combinations by index
- `dict_to_string_for_dir()` generates directory names via SHA256 hash (or string concatenation in old style)
- `Constant` class switches paths between local (`./res/`, `./data/`) and HPC based on hostname (`UML-FNQ2JDW1GV` = local Mac)

### Data Conventions
- Odds are converted to implied probabilities (1/odds) during feature engineering
- Raw tick data (`win_*.parquet`) stores odds in raw form (e.g., 5.0 means 5-to-1)
- Missing order book values: lays filled with 1001, backs filled with 1.0
- Quantity levels: q_100, q_200, q_1000 (execution price for that order size)
- Time: `time` column (datetime index in tick data), `time_delta = max(time) - time` per file_name (time before close)
- Winner convention: `id = -1` in processed parquets
- Market key: `file_name + "_" + id` uniquely identifies a runner across DataFrames
- In-sample: years 2017–2024; Out-of-sample: 2025

### Time Definitions
| t_def | tp1 (post) | t0 (entry) | tm1 | tm2 | tm3 (earliest) |
|-------|-----------|------------|-----|-----|---------|
| 0 | 10s | 60s | 1m15s | 3m | 10m |
| 1 | 10s | 2m | 2m15s | 4m | 10m |
| 2 | 10s | 30s | 35s | 2m | 5m |
| 3 | 5s | 20s | 25s | 2m | 5m |

Features use suffix `_m0` (at t0), `_m1` (at tm1), `_m2`, `_m3`, `_p1` (at tp1 for targets).

### Key Utility Modules (`utils_locals/`)
- **`process_races.py`**: Core data processing — loads BZ2 JSON, extracts order books per runner, computes best bid/ask at quantity levels, infers trade direction (back/lay/cross_matching)
- **`feature_tools.py`**: `add_time_snapshot()` creates features at time deltas with forward-fill
- **`loader.py`**: Merges summary statistics across years
- **`parser.py`**: Custom CLI argument parser (positional → `args.a`, `args.b`, `args.c`)

### Model Architecture
- **V1**: XGBoost classifiers trained on `win` target. Grid search over topk, spread, t_definition, y_var. Top-7 by log-loss averaged per t_def.
- **V2**: XGBoost + LightGBM with isotonic calibration. Top-15 by log-loss averaged per t_def.
- **Cross-t Super-Ensemble**: Average V1 across t0-t3, average V2 across t0-t3, then blend V1(20%) + V2(80%).
- Cross-t alignment: inner join by `key = file_name + "_" + id` (different t_defs may have different row counts).

### Fill Simulation
- Replays tick-by-tick order book data after model's decision point
- Simulates 2 windows (60s, 20s before close) × 3 price variants (best_back, ±1 tick) = 6 scenarios
- **Conservative fill**: trade at `prc >= limit_price` after order placement
- **Moderate fill**: conservative OR `best_lay <= limit_price` (spread crosses our price)
- Betfair tick table used for ±1 tick price adjustments

### Output Structure
- Processed data: `data/p/greyhound_au/` — parquet files prefixed `win_`, `place_`, `mdef_`
- Features: `res/features_t{0,1,2,3}/greyhound_au_features_part_0-9.parquet`
- V1 models: `res/win_model/t{0,1,2,3}/{config}/save_df.parquet`
- V2 models: `res/win_model_v2/t{0,1,2,3}/{config}/save_df.parquet`
- Ensemble predictions: `res/analysis/ultimate_cross_t_ensemble_predictions.parquet`
- Fill simulation: `res/fill_simulation/fill_part_{0-29}.parquet` → `fill_results_merged.parquet`
- Reports: `res/alpha_odds_report.pdf`, `res/fill_simulation/fill_analysis_report.pdf`

## Grid Search Parameters
Shared grid dimensions (defined in `parameters.py` as `SHARED_GRID`):
- `topk_restriction`: [1, 2, 3] — number of top runners per market
- `t_definition`: [1, 2, 3] — time snapshot configuration
- `spread_restriction`: [-1, 0.1, 0.05] — bid-ask width filter (-1 = no filter)
- `spread_top_k_criterion`: VANILLA or Q100
- `y_var`: target variable selection

## Local Data Available
- `data/raw/PRO/` — sample raw BZ2 Betfair files (2025 Oct, Nov)
- `data/raw/PRO_horse/` — horse racing sample data
- `res/` — mirrors key Spartan results including reports, fill simulation results
- `res/fill_simulation/fill_analysis_report.pdf` — fill simulation report (downloaded)
- `res/alpha_odds_report.pdf` — main model report (downloaded)

## Key Technical Notes
- **Local Python environment has issues** — numpy/pandas/scipy compatibility problems across conda envs. Run analysis on Spartan.
- **No scipy on Spartan venv** — use `from math import erf, sqrt` and `0.5*(1+erf(x/sqrt(2)))` for norm_cdf
- **V2 marketBaseRate is z-scored** — use default 0.08 commission when loading V2 save_df
- **matplotlib on Spartan** — must `module load matplotlib/3.5.2` in addition to base stack for report scripts
- **Code deployed flat** — all Python files go to `/home/adidishe/alpha_odds/` (no subdirectories on server)

## Key Python Dependencies (venv_comp)
pandas, numpy, pyarrow, scikit-learn, xgboost, lightgbm, tqdm, betfair_data (0.3.4)
