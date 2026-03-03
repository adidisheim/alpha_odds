"""
Percentile-based validation checker for live model outputs.

Loads precomputed empirical baselines and tests whether live quantities
fall within historical percentile bounds at three significance levels:
  - 0% alpha: value inside [min, max]
  - 5% alpha: value inside [p2.5, p97.5]  (two-sided)
  - 1% alpha: value inside [p0.5, p99.5]  (two-sided)

Usage:
    checker = PercentileChecker()

    # Check individual metrics
    checker.check_bet(edge=0.05, back_odds=4.2)
    checker.check_race(n_bets=2, sum_model_prob=0.85)
    checker.check_daily(win_rate=0.45, n_bets=6)

    # Full session validation from a bets DataFrame
    report = checker.validate_session(bets_df)
    print(checker.format_session_report(report))

    # Feature-level checks (requires feature baselines extracted on Spartan)
    checker.check_features(feature_values_dict, t_def=0)
    checker.check_daily_feature_means(daily_means_dict, t_def=0)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


# ── Test definitions ──
ALPHA_TESTS = {
    "0%":  ("min", "max"),       # exact historical range
    "5%":  ("p2.5", "p97.5"),    # two-sided 5%
    "1%":  ("p0.5", "p99.5"),    # two-sided 1%
}


def _run_test(value, stats):
    """Run all three alpha tests on a single value against baseline stats."""
    if stats is None or stats.get("n", 0) == 0:
        return None

    result = {
        "value": value,
        "mean": stats["mean"],
        "std": stats["std"],
        "z": ((value - stats["mean"]) / stats["std"]) if stats["std"] > 0 else 0.0,
        "tests": {},
    }
    for alpha, (lo_key, hi_key) in ALPHA_TESTS.items():
        lo = stats[lo_key]
        hi = stats[hi_key]
        if value < lo:
            result["tests"][alpha] = "FAIL_LOW"
        elif value > hi:
            result["tests"][alpha] = "FAIL_HIGH"
        else:
            result["tests"][alpha] = "PASS"
    return result


class PercentileChecker:
    """Check live values against historical percentile baselines."""

    def __init__(self, baselines_dir: Optional[Union[str, Path]] = None):
        if baselines_dir is None:
            baselines_dir = Path(__file__).parent / "baselines"
        self.baselines_dir = Path(baselines_dir)

        # Load signal baselines
        sig_path = self.baselines_dir / "percentile_baselines.json"
        if sig_path.exists():
            with open(sig_path) as f:
                self._sig = json.load(f)
        else:
            self._sig = {}

        self.metadata = self._sig.get("metadata", {})

        # Load feature baselines (optional — only if extracted on Spartan)
        self._feat_bet = self._load_parquet("feature_bet_percentiles.parquet")
        self._feat_daily_mean = self._load_parquet("feature_daily_mean_percentiles.parquet")
        self._feat_daily_std = self._load_parquet("feature_daily_std_percentiles.parquet")

    def _load_parquet(self, filename):
        p = self.baselines_dir / filename
        if p.exists():
            return pd.read_parquet(p)
        return None

    # ────────────────────────────────────────────────────────
    #  Single-metric checks
    # ────────────────────────────────────────────────────────

    def check(self, level: str, metric: str, value: float) -> Optional[Dict]:
        """Check one value against baselines at a given level."""
        stats = self._sig.get(level, {}).get(metric)
        if stats is None:
            return None
        r = _run_test(value, stats)
        if r:
            r["metric"] = metric
            r["level"] = level
        return r

    def check_bet(self, **kw) -> List[Dict]:
        """Check bet-level metrics (edge, back_odds, model_prob, market_prob)."""
        return [r for k, v in kw.items()
                if v is not None and (r := self.check("bet_level", k, float(v))) is not None]

    def check_race(self, **kw) -> List[Dict]:
        """Check race-level metrics (n_bets, sum_model_prob, mean_edge, ...)."""
        return [r for k, v in kw.items()
                if v is not None and (r := self.check("race_level", k, float(v))) is not None]

    def check_daily(self, **kw) -> List[Dict]:
        """Check daily-level metrics (win_rate, n_bets, roi_pct, ...)."""
        return [r for k, v in kw.items()
                if v is not None and (r := self.check("daily_level", k, float(v))) is not None]

    # ────────────────────────────────────────────────────────
    #  Feature checks
    # ────────────────────────────────────────────────────────

    def check_features(self, feature_values: Dict[str, float], t_def: int = 0) -> List[Dict]:
        """Check individual feature values against bet-runner baselines.

        Args:
            feature_values: {feature_name: value} for one runner's features
            t_def: time definition (0-3)
        """
        if self._feat_bet is None:
            return []
        subset = self._feat_bet[self._feat_bet["t_def"] == t_def]
        if subset.empty:
            return []
        lookup = {row["feature"]: row.to_dict() for _, row in subset.iterrows()}
        results = []
        for feat, val in feature_values.items():
            if feat in lookup and val is not None and not np.isnan(val):
                r = _run_test(val, lookup[feat])
                if r:
                    r["metric"] = feat
                    r["level"] = f"feature_bet_t{t_def}"
                    results.append(r)
        return results

    def check_daily_feature_means(self, daily_means: Dict[str, float], t_def: int = 0) -> List[Dict]:
        """Check daily mean of each feature against historical daily means."""
        if self._feat_daily_mean is None:
            return []
        subset = self._feat_daily_mean[self._feat_daily_mean["t_def"] == t_def]
        if subset.empty:
            return []
        lookup = {row["feature"]: row.to_dict() for _, row in subset.iterrows()}
        results = []
        for feat, val in daily_means.items():
            if feat in lookup and val is not None and not np.isnan(val):
                r = _run_test(val, lookup[feat])
                if r:
                    r["metric"] = feat
                    r["level"] = f"feature_daily_mean_t{t_def}"
                    results.append(r)
        return results

    def check_daily_feature_stds(self, daily_stds: Dict[str, float], t_def: int = 0) -> List[Dict]:
        """Check daily std of each feature against historical daily stds."""
        if self._feat_daily_std is None:
            return []
        subset = self._feat_daily_std[self._feat_daily_std["t_def"] == t_def]
        if subset.empty:
            return []
        lookup = {row["feature"]: row.to_dict() for _, row in subset.iterrows()}
        results = []
        for feat, val in daily_stds.items():
            if feat in lookup and val is not None and not np.isnan(val):
                r = _run_test(val, lookup[feat])
                if r:
                    r["metric"] = feat
                    r["level"] = f"feature_daily_std_t{t_def}"
                    results.append(r)
        return results

    # ────────────────────────────────────────────────────────
    #  Session validation
    # ────────────────────────────────────────────────────────

    def validate_session(self, bets_df: pd.DataFrame,
                         commission: float = 0.075,
                         stake: float = 25.0) -> Dict[str, Any]:
        """
        Full validation from a DataFrame of qualifying bets.

        Expected columns: edge, back_odds, model_prob, market_prob, win,
                          file_name, date (or marketTime_local).
        """
        if len(bets_df) == 0:
            return {"error": "empty", "summary": {}}

        df = bets_df.copy()
        if "date" not in df.columns and "marketTime_local" in df.columns:
            df["date"] = pd.to_datetime(df["marketTime_local"]).dt.date

        report: Dict[str, Any] = {"bet": [], "race": [], "daily": [], "summary": {}}

        # ── Bet-level: aggregate distribution of all bets ──
        for col in ["edge", "back_odds", "model_prob", "market_prob"]:
            if col in df.columns:
                report["bet"].append(self.check("bet_level", col, float(df[col].mean())))

        # ── Race-level ──
        if "file_name" in df.columns:
            rg = df.groupby("file_name")
            race_n_bets = rg.size()
            race_sum_mp = rg["model_prob"].sum() if "model_prob" in df else None
            race_mean_edge = rg["edge"].mean() if "edge" in df else None
            race_mean_odds = rg["back_odds"].mean() if "back_odds" in df else None

            report["race"].extend(self.check_race(
                n_bets=float(race_n_bets.mean()),
                sum_model_prob=float(race_sum_mp.mean()) if race_sum_mp is not None else None,
                mean_edge=float(race_mean_edge.mean()) if race_mean_edge is not None else None,
                mean_back_odds=float(race_mean_odds.mean()) if race_mean_odds is not None else None,
            ))

        # ── Daily-level ──
        if "date" in df.columns and "win" in df.columns:
            df["pnl"] = np.where(
                df["win"] == 1,
                (df["back_odds"] - 1) * (1 - commission) * stake,
                -stake,
            )
            dg = df.groupby("date")
            for date_val, grp in dg:
                n = len(grp)
                checks = self.check_daily(
                    win_rate=float(grp["win"].mean()),
                    n_bets=float(n),
                    total_pnl=float(grp["pnl"].sum()),
                    roi_pct=float((grp["pnl"].sum() / (n * stake)) * 100),
                    pnl_per_bet=float(grp["pnl"].mean()),
                    mean_edge=float(grp["edge"].mean()) if "edge" in grp else None,
                    mean_back_odds=float(grp["back_odds"].mean()) if "back_odds" in grp else None,
                    mean_model_prob=float(grp["model_prob"].mean()) if "model_prob" in grp else None,
                    mean_market_prob=float(grp["market_prob"].mean()) if "market_prob" in grp else None,
                )
                report["daily"].append({"date": str(date_val), "checks": checks})

        # ── Summary ──
        report["summary"] = self._summarize(report)
        return report

    def _summarize(self, report: Dict) -> Dict:
        """Aggregate pass/fail counts across all checks."""
        summary = {}
        for level in ["bet", "race", "daily"]:
            items = report.get(level, [])
            flat = []
            for item in items:
                if isinstance(item, dict):
                    if "tests" in item:
                        flat.append(item)
                    elif "checks" in item:  # daily level wraps in {date, checks}
                        flat.extend(item["checks"])
                elif isinstance(item, list):
                    flat.extend(item)

            for alpha in ["0%", "5%", "1%"]:
                n_tot = n_pass = n_fail_lo = n_fail_hi = 0
                for r in flat:
                    if r is None or "tests" not in r:
                        continue
                    t = r["tests"].get(alpha)
                    if t is None:
                        continue
                    n_tot += 1
                    if t == "PASS":
                        n_pass += 1
                    elif t == "FAIL_LOW":
                        n_fail_lo += 1
                    else:
                        n_fail_hi += 1
                if n_tot > 0:
                    summary[f"{level}_{alpha}"] = {
                        "n": n_tot, "pass": n_pass,
                        "fail_low": n_fail_lo, "fail_high": n_fail_hi,
                        "pass_rate": round(n_pass / n_tot, 3),
                    }
        return summary

    # ────────────────────────────────────────────────────────
    #  Formatting
    # ────────────────────────────────────────────────────────

    def format_checks(self, results: List[Optional[Dict]], title: str = "") -> str:
        """Format a list of check results as a readable table."""
        results = [r for r in results if r is not None]
        if not results:
            return f"{title}: no checks\n"

        lines = []
        if title:
            lines.append(f"\n{'='*90}")
            lines.append(f"  {title}")
            lines.append(f"{'='*90}")

        hdr = f"  {'Metric':30s} {'Value':>10s} {'Mean':>10s} {'Z':>7s}  {'0%':^9s} {'5%':^9s} {'1%':^9s}"
        lines.append(hdr)
        lines.append("  " + "-" * 86)

        for r in results:
            v = f"{r['value']:.4f}" if isinstance(r["value"], (int, float)) else str(r["value"])
            m = f"{r['mean']:.4f}" if r["mean"] is not None else "N/A"
            z = f"{r['z']:.2f}" if abs(r["z"]) < 1e6 else "inf"
            t0 = r["tests"].get("0%", "?")
            t5 = r["tests"].get("5%", "?")
            t1 = r["tests"].get("1%", "?")
            lines.append(f"  {r['metric']:30s} {v:>10s} {m:>10s} {z:>7s}  {t0:^9s} {t5:^9s} {t1:^9s}")

        return "\n".join(lines)

    def format_session_report(self, report: Dict) -> str:
        """Format a full session validation report."""
        parts = []
        if report.get("error"):
            return f"Validation error: {report['error']}"

        parts.append(self.format_checks(
            [r for r in report.get("bet", []) if r is not None],
            "BET-LEVEL (session averages vs historical per-bet)"
        ))
        parts.append(self.format_checks(
            [r for r in report.get("race", []) if r is not None],
            "RACE-LEVEL (session averages vs historical per-race)"
        ))

        # Daily: show each day
        for day_item in report.get("daily", []):
            if isinstance(day_item, dict) and "checks" in day_item:
                parts.append(self.format_checks(
                    day_item["checks"],
                    f"DAILY: {day_item['date']}"
                ))

        # Summary
        summary = report.get("summary", {})
        if summary:
            parts.append(f"\n{'='*90}")
            parts.append("  SUMMARY")
            parts.append(f"{'='*90}")
            for k, v in summary.items():
                parts.append(f"  {k:25s}  n={v['n']}  pass={v['pass']}  "
                             f"fail_lo={v['fail_low']}  fail_hi={v['fail_high']}  "
                             f"pass_rate={v['pass_rate']:.1%}")

        return "\n".join(parts)

    # ────────────────────────────────────────────────────────
    #  Quick one-liner
    # ────────────────────────────────────────────────────────

    def quick_check(self, level: str, **kw) -> str:
        """Quick formatted check at a given level."""
        if level == "bet_level":
            return self.format_checks(self.check_bet(**kw), "Bet Check")
        elif level == "race_level":
            return self.format_checks(self.check_race(**kw), "Race Check")
        elif level == "daily_level":
            return self.format_checks(self.check_daily(**kw), "Daily Check")
        return f"Unknown level: {level}"

    # ────────────────────────────────────────────────────────
    #  Convenience: load paper trading logs and validate
    # ────────────────────────────────────────────────────────

    @staticmethod
    def load_paper_trading_bets(log_dir: Union[str, Path],
                                dates: Optional[List[str]] = None) -> pd.DataFrame:
        """Load bets from paper trading log parquets.

        Args:
            log_dir: path to paper_trading_logs/ directory
            dates: optional list of date strings (YYYY-MM-DD) to filter
        """
        log_dir = Path(log_dir)
        trades_dir = log_dir / "trades"
        if not trades_dir.exists():
            return pd.DataFrame()

        parts = []
        for f in sorted(trades_dir.glob("*.parquet")):
            if dates and f.stem not in dates:
                continue
            parts.append(pd.read_parquet(f))

        if not parts:
            return pd.DataFrame()

        df = pd.concat(parts, ignore_index=True)
        return df


# ── CLI ──
if __name__ == "__main__":
    import sys

    checker = PercentileChecker()
    print(f"Loaded baselines: {checker.metadata}")

    # Demo checks
    print(checker.quick_check("bet_level", edge=0.05, back_odds=4.0, model_prob=0.42, market_prob=0.37))
    print(checker.quick_check("race_level", n_bets=2, sum_model_prob=0.85, mean_edge=0.05))
    print(checker.quick_check("daily_level", win_rate=0.40, n_bets=6, roi_pct=25.0, mean_edge=0.06))

    # If a predictions parquet is passed as argument, run full validation
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"\nLoading {path} for full validation...")
        df = pd.read_parquet(path)
        if "file_name" not in df.columns and "key" in df.columns:
            df["file_name"] = df["key"].str.rsplit("_", n=1).str[0]
        if "date" not in df.columns and "marketTime_local" in df.columns:
            df["date"] = pd.to_datetime(df["marketTime_local"]).dt.date
        # Filter to qualifying bets
        bets = df[
            (df["edge"] > 0.03)
            & (df["back_odds"] >= 1.01)
            & (df["back_odds"] <= 50.0)
            & (df.get("market_prob", pd.Series(dtype=float)) >= 0.02)
        ] if "market_prob" in df.columns else df[df["edge"] > 0.03]

        report = checker.validate_session(bets)
        print(checker.format_session_report(report))
