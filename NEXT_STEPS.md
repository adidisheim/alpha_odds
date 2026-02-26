# Next Steps — Value Betting Strategy

## Current Status (Feb 2026)

### Completed

1. **Data pipeline** — Raw BZ2 → processed parquets → features across 4 time definitions (t0-t3)
2. **Bug fixes** — ffill cross-contamination, duplicate merges, breakpoints. Verified no lookahead bias.
3. **Strategy chosen** — Value betting with limit orders (predict true win prob, back underpriced runners)
4. **V1 models trained** — XGBoost grid search across topk, spread, t_definition. Top-7 ensemble per t_def.
5. **V2 models trained** — XGBoost + LightGBM + isotonic calibration. Top-15 ensemble per t_def.
6. **Cross-t Super-Ensemble built** — V1_cross(20%) + V2_cross(80%), best model overall
7. **Fill rate simulation COMPLETE** — Replayed tick-by-tick order book data for all 2,010 qualifying bets

### Best Model: Cross-t Super-Ensemble

| Metric | Value |
|--------|-------|
| Architecture | V1(20%) + V2(80%), averaged across t0-t3 |
| Components | 28 V1 models + 60 V2 models = 88 total |
| Log-loss | 0.316321 (vs Market: 0.319722) |
| OOS period | Jan–Nov 2025 (286K predictions) |
| Bets (edge>3%) | 2,010 |
| Win rate | 47.3% |
| ROI | +19% |
| Sharpe (annual) | 3.6 |
| z-stat / p-value | 3.90 / 0.000048 |
| Profit ($25/bet) | $9,440 |
| Profitable months | 10/11 |

### Fill Rate Simulation Results

Replayed tick-by-tick order book data for all 110,865 runners with positive edge across 305 day-files.

**Primary scenario (60s window, best_back price, edge>3%):**

| Fill Model | Fill Rate | ROI | P&L ($25/bet) | z-stat |
|------------|-----------|-----|---------------|--------|
| 100% (baseline) | 100% | +19% | $9,440 | 3.89 |
| Conservative | 97% | +17% | $8,451 | 3.52 |
| Moderate | 99% | +18% | $9,212 | 3.77 |

**Key findings:**
- **Strategy retains ~90% of profit under conservative fills** — no material fill risk
- Fill rates high across all odds buckets: 84-98% conservative, 95-100% moderate
- Median time-to-fill: 0.5 seconds (markets very liquid in last 60s)
- Median volume at limit price: $1,362 (ample for $25 bets)
- **Longshot profit (15-30 odds) survives**: $3,725 conservative vs $4,493 baseline (83% retained)
- Even +1 tick (asking for better odds) still achieves 86% fill rate
- 20s window performs similarly to 60s window (both >96% fill)

Reports:
- Main model report: `res/alpha_odds_report.pdf`
- Fill simulation report: `res/fill_simulation/fill_analysis_report.pdf`

---

## Remaining Steps (in order)

### Phase 3: Live Validation

1. **Paper trading via Betfair API** — **NEXT UP (resume here)**
   - **Code is built** — `_05_paper_trading/` has the full system (8 modules, streaming, features, 88-model ensemble, P&L tracking). Streaming integration was fixed on 2026-02-26.
   - **Still needed before running:**
     1. Betfair API credentials (app key, SSL certs, username/password) → fill in `.env`
     2. Model artifacts from Spartan → run `sbatch scripts/slurm/_save_model_artifacts.slurm` then `bash _05_paper_trading/scripts/download_models.sh`
     3. Decide where to run (local / Spartan / AWS)
   - **To test:** `cd _05_paper_trading && pip install -r requirements.txt && python main.py --dry-run`
   - Run for 2-4 weeks to verify OOS performance holds in real-time
   - **Key risk**: latency between model prediction and order placement

2. **Live testing with small stakes (10 AUD/bet)**
   - Same system, but placing real limit orders
   - Start with 10 AUD/bet (conservative)
   - Monitor daily P&L, fill rates, and slippage
   - Kill switch if cumulative loss exceeds $500 or 3 consecutive losing weeks
   - Target: 2-4 weeks of live data

3. **Scale to 25-50 AUD/bet**
   - Once 10 AUD/bet confirms the edge, increase stake
   - Monitor market impact (are we moving prices?)
   - Expected: ~$10K/year at $25/bet based on OOS results

### Phase 4: Production

4. **AWS deployment**
   - EC2 instance running 24/7 (greyhound racing runs throughout the day)
   - Betfair API connection with automated order management
   - Real-time feature computation from streaming data
   - Monitoring dashboard (P&L, fill rates, model confidence)
   - Alert system for anomalies (unusual losses, API disconnections)

5. **Strategy extensions**
   - Explore lay strategy (early analysis shows strong potential)
   - Horse racing markets (already have sample data in `data/raw/PRO_horse/`)
   - Dynamic position sizing (Kelly criterion or fractional Kelly)
   - Cross-runner features (relative pricing within a race)

---

## Technical Decisions for Live Trading

| Decision | Status |
|----------|--------|
| Which t_definition? | All 4 (cross-t ensemble averages them) |
| Edge threshold | 3% (best risk-adjusted returns) |
| Bet size | Start 10 AUD, scale to 25-50 AUD |
| Commission assumption | 7.5% (conservative, actual ~8% from marketBaseRate) |
| Order type | Back limit at best_back price, 60s before close |
| Fill assumption | Conservative: 97% fill rate confirmed by simulation |
| AWS vs local | AWS (needs 24/7 uptime for AU racing schedule) |
| Betfair API library | `betfairlightweight` (implemented in `_05_paper_trading/`) |
| Model retraining | TBD — monthly? quarterly? on expanding window? |

---

## Files Reference

### Pipeline Scripts
| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `_01_process_files/_01_untar_files.py` | Untar raw Betfair BZ2 files |
| 1 | `_01_process_files/_02_process_all_files_para.py` | Process raw → parquet (parallelized) |
| 1 | `_01_process_files/_03_merge_files.py` | Merge processed outputs |
| 2 | `_02_summary_stats/_01_summary_stats_quick.py` | Compute summary statistics |
| 3 | `_03_feature_engeneering/_01_feature_engineering_para.py` | Feature engineering (parallelized) |
| 3.5 | `_04_first_run_and_analysis/_00_merge_features.py` | Merge & validate feature parts |
| 4 | `_04_first_run_and_analysis/_01_run.py` | V1 model training (grid search) |
| 4 | `_04_first_run_and_analysis/_02_win_probability_model.py` | V2 win probability model |
| 4 | `_04_first_run_and_analysis/_03_win_probability_model_v2.py` | V2 with LightGBM + isotonic |

### Analysis Scripts (run on Spartan)
| Script | Purpose |
|--------|---------|
| `_05_generate_report.py` | Full PDF report (cross-t super-ensemble) |
| `_06_remote_analysis.py` | V1 comprehensive backtest |
| `_07_v2_analysis.py` | V2 multi-model ensemble analysis |
| `_08_super_ensemble.py` | V1+V2 t0-only super-ensemble |
| `_09_multi_t_analysis.py` | Cross-t ensemble comparison |
| `_10_ultimate_ensemble.py` | Ultimate cross-t super-ensemble + backtest |
| `_11_stacking_experiment.py` | Stacking experiment (meta-learner) |
| `_12_dense_ensemble_experiment.py` | Dense ensemble experiments |
| `_13_fill_simulator.py` | Fill rate simulation (parallelized, 30 tasks) |
| `_14_fill_merge.py` | Merge fill simulation results |
| `_15_fill_analysis.py` | Fill-adjusted backtest + PDF report |

### Key Utility Modules
- `utils_locals/process_races.py` — Raw BZ2 JSON → order book extraction
- `utils_locals/feature_tools.py` — Time snapshot creation with forward-fill
- `utils_locals/parser.py` — Custom CLI argument parser
- `utils_locals/loader.py` — Summary statistics loader
- `parameters.py` — Params/Constant classes, grid search, path management
