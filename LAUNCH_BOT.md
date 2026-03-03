# Launch Paper Trading Bot

Instructions for running the alpha_odds paper trading bot with real (but symbolic) $1 stakes on Betfair Australia greyhound racing markets.

## What This Does

Runs the cross-t super-ensemble (88 models) live against Betfair markets. When the model finds an edge > 3% on a runner, it places a **real $1 back limit order** on Betfair. This is real money — but at $1/bet the risk is minimal.

### Risk Profile ($1/bet)

| Metric | Value |
|--------|-------|
| Avg bets/day | ~7 |
| Avg daily exposure | ~$7 |
| Max daily exposure | ~$15 |
| Expected daily P&L | +$1.63 |
| Worst observed day (backtest) | -$8.86 |
| Worst observed week (backtest) | -$22 |
| Kill switch triggers at | -$20 daily loss |
| **Theoretical max daily loss** | **$15** |

## Prerequisites (Already Verified)

- [x] Betfair cert-based login works from this machine
- [x] `.env` file configured with credentials
- [x] SSL certs in `certs/` directory
- [x] Model artifacts in `models/` (88 components)
- [x] `venv_comp` has all required packages
- [ ] Email notifications configured (optional but recommended)

## Step-by-Step Instructions

### 1. Update config for $1 stakes

Edit `_05_paper_trading/config.py` and change these values:

```python
STAKE_SIZE = 1.0        # was 25.0 — symbolic $1 bets
MAX_DAILY_LOSS = 20.0   # was 500.0 — kill switch at $20 loss
```

Do NOT change any other config values (edge threshold, model weights, etc.).

### 1b. Email notifications (auto-configured)

The bot sends email alerts via the `claude-email-mcp` Gmail API client at `~/claude-email-mcp/`. This uses OAuth2 with existing credentials — no passwords needed. Emails go to `antoinedidisheim@gmail.com`.

If `~/claude-email-mcp/token.json` exists (it should — Claude Code already uses it), emails work out of the box.

**What gets emailed:**
| Event | Subject line example |
|-------|---------------------|
| Bot starts | `Bot Started - LIVE ($1 bets)` |
| Bet wins | `WIN +$3.70 \| Day +$5.20 \| Total +$12.40` |
| Bet loses | `LOSE -$1.00 \| Day +$4.20 \| Total +$11.40` |
| Anomaly detected | `ANOMALY [CRITICAL] - Paper Trading` |
| Kill switch | `KILL SWITCH - Daily loss $20.00` |
| Bot stops | `Bot Stopped - Day +$5.20 \| Total +$12.40` |

Each email includes a mobile-friendly HTML table with edge, odds, model probability, and running P&L.

### 2. Verify Betfair connection still works

```bash
cd /home/unimelb.edu.au/adidisheim/Dropbox/side_project/alpha_odds
curl -s -X POST https://identitysso-cert.betfair.com.au/api/certlogin \
  --data-urlencode "username=antoinedidisheim@gmail.com" \
  --data-urlencode "password=VViseron6757+" \
  -H "X-Application: Uebo4sXlb5USEutl" \
  --cert certs/client-2048.crt \
  --key certs/client-2048.key
```

Expected: `{"sessionToken":"...","loginStatus":"SUCCESS"}`

If you get `CERT_AUTH_REQUIRED`, the cert may need re-uploading to Betfair — ask the user.

### 3. Launch the bot (LIVE mode, real $1 bets)

```bash
cd /home/unimelb.edu.au/adidisheim/Dropbox/side_project/alpha_odds/_05_paper_trading
source ../venv_comp/bin/activate
python main.py 2>&1 | tee ../paper_trading_logs/launch_$(date +%Y-%m-%d).log
```

**IMPORTANT**: No `--dry-run` flag — this places real orders. The user has explicitly approved $1 live bets.

### 4. Verify startup

You should see in the logs:
```
Paper Trading System Starting
Dry run: False
Loading model artifacts...
  t0: 7 V1 + 15 V2 models
  t1: 7 V1 + 15 V2 models
  t2: 7 V1 + 15 V2 models
  t3: 7 V1 + 15 V2 models
Loaded 88 total model components
Loaded V1 normalization params for t0: 90 features
...
Setup complete. Entering main loop.
```

If login fails or models don't load, stop and investigate — do not retry blindly.

### 5. Monitor

Check logs:
```bash
tail -f paper_trading_logs/launch_$(date +%Y-%m-%d).log
```

Check daily summary (created at end of day or on shutdown):
```bash
cat paper_trading_logs/daily/$(date +%Y-%m-%d).json
```

Check placed bets:
```bash
ls -la paper_trading_logs/bets/
```

### 6. Stop the bot

Press `Ctrl+C` — the bot handles SIGINT gracefully:
- Stops streaming
- Flushes all buffered data to disk
- Saves daily summary JSON
- Saves signal tracker state (for backtest comparison)
- Sends shutdown email with daily summary
- Logs shutdown

## Safety Rules

1. **NEVER increase STAKE_SIZE beyond $1** without explicit user approval
2. **NEVER disable the kill switch** (MAX_DAILY_LOSS)
3. **NEVER run multiple bot instances** simultaneously — this would double-bet
4. The bot only bets on **Australian greyhound WIN markets**
5. The bot only bets when **edge > 3%** and odds are between 1.01 and 1000
6. If anything looks wrong (errors, unexpected behavior, API issues), **stop the bot first, investigate second**

## What Gets Logged

| Location | Content |
|----------|---------|
| `paper_trading_logs/YYYY-MM-DD.log` | Full console output |
| `paper_trading_logs/ticks/YYYY-MM-DD/` | Raw tick data per market |
| `paper_trading_logs/features/YYYY-MM-DD.parquet` | Computed features |
| `paper_trading_logs/decisions/YYYY-MM-DD.parquet` | Model predictions |
| `paper_trading_logs/bets/YYYY-MM-DD.parquet` | Bet outcomes + P&L |
| `paper_trading_logs/daily/YYYY-MM-DD.json` | Daily summary |
| `paper_trading_logs/signal_tracker/YYYY-MM-DD.json` | Signal distributions vs backtest |

## Signal Tracking & Anomaly Detection

The bot continuously tracks live signal distributions and compares them to backtest baselines (OOS edge>3%: 2010 bets, +19% ROI, 97% fill rate).

### What's tracked

| Metric | Backtest baseline | What deviation means |
|--------|------------------|---------------------|
| Avg edge | ~4.5% | Too high = feature bug, too low = model degraded |
| Bet rate | ~6.6/day | Too high = edge filtering broken |
| Avg model prob | ~25% | Near 0 or 1 = model broken |
| Avg odds | ~5.0 | Shift = market composition changed |
| Win rate | ~23% | Significant shift = model stale |
| Fill rate | ~97% | Low = liquidity conditions changed |
| V1 vs V2 gap | Small | Large divergence = ensemble inconsistency |

### Auto-fixes

When anomalies are detected, the bot attempts to self-correct before pausing:

| Anomaly | Auto-fix |
|---------|----------|
| Edge too high (>2.5x backtest) | Reload normalization params from disk |
| Bet rate too high (>3x backtest) | Bump edge threshold by +1% |
| Model probs extreme (<5% or >60%) | Reload all 88 model artifacts |
| Win rate off (z>2.5) | Warning email (no auto-fix) |
| Fill rate too low (<70% of backtest) | Warning email (no auto-fix) |

If a CRITICAL anomaly persists after auto-fix, the bot **pauses trading** and sends an alert email. Restart the bot to resume.

### Signal tracker output

Saved to `paper_trading_logs/signal_tracker/YYYY-MM-DD.json` on shutdown. Contains:
- Live vs backtest comparison with % deviations
- Full anomaly history with timestamps and fixes applied
- All signal distribution statistics

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CERT_AUTH_REQUIRED` | Re-upload `certs/client-2048.crt` to Betfair account |
| `NO_SESSION` / token expired | Bot auto-sends keep-alive every 15min — if this fails, restart |
| No bets placed all day | Normal if no races have edge > 3% — check `decisions/` parquet |
| Kill switch activated | Daily loss hit $20 — bot stops betting until midnight UTC reset |
| Models fail to load | Check `models/manifest.json` exists and model files are present |
| `--data-urlencode` needed | Password contains `+` — always use `--data-urlencode` for curl |
| No email received | Check `~/claude-email-mcp/token.json` exists. If not, run `cd ~/claude-email-mcp && python server.py` once to authorize |
| Anomaly pause activated | Bot stopped placing bets due to signal deviation. Check `signal_tracker/` JSON, fix root cause, restart |
| Edge threshold bumped | Auto-fix increased threshold. Check logs for why. Restart bot to reset to default 3% |

## Architecture Reference

- **Entry point**: `_05_paper_trading/main.py`
- **Config**: `_05_paper_trading/config.py` (all thresholds and params)
- **Betfair client**: `_05_paper_trading/betfair_client.py` (cert login, streaming, API)
- **Paper trader**: `_05_paper_trading/paper_trader.py` (edge evaluation, bet placement, fill tracking)
- **Model engine**: `_05_paper_trading/model_engine.py` (88-model cross-t ensemble)
- **Feature engine**: `_05_paper_trading/feature_engine.py` (mirrors historical pipeline)
- **Market cache**: `_05_paper_trading/market_cache.py` (live tick accumulation)
- **Data logger**: `_05_paper_trading/data_logger.py` (all logging)
- **Notifier**: `_05_paper_trading/notifier.py` (email alerts on bets, anomalies, startup/shutdown)
- **Signal tracker**: `_05_paper_trading/signal_tracker.py` (live vs backtest comparison, anomaly detection, auto-fixes)

## Racing Schedule

Australian greyhound races typically run from ~10am to ~11pm AEST (UTC+10/11). The bot will be idle outside these hours. Most bets occur between 12pm–10pm AEST.
