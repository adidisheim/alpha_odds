"""
Email Notifier — sends email alerts for paper trading events via claude-email-mcp GmailClient.

Uses the claude-email-mcp project's venv (which has google-api-python-client installed)
by shelling out to its Python interpreter. This avoids polluting the bot's own venv.

Events:
  - Bot startup
  - Bet settled (win/lose) with running P&L
  - Kill switch activation
  - Anomaly detection
  - Bot shutdown with daily summary
"""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from config import EMAIL_TO, STAKE_SIZE

logger = logging.getLogger(__name__)

GMAIL_MCP_DIR = Path.home() / "claude-email-mcp"
GMAIL_MCP_PYTHON = GMAIL_MCP_DIR / "venv" / "bin" / "python"


class EmailNotifier:
    """Sends email notifications for paper trading events via Gmail API."""

    def __init__(self):
        self.enabled = False

        if not GMAIL_MCP_PYTHON.exists():
            logger.warning(
                f"Email disabled: {GMAIL_MCP_PYTHON} not found. "
                f"Set up ~/claude-email-mcp/ with its venv."
            )
            return

        token_file = GMAIL_MCP_DIR / "token.json"
        if not token_file.exists():
            logger.warning(
                f"Email disabled: {token_file} not found. "
                f"Run 'cd ~/claude-email-mcp && python server.py' to authorize."
            )
            return

        self.enabled = True
        logger.info("Email notifications enabled via claude-email-mcp Gmail API")

    def _send(self, subject, body):
        """Send a plain-text email by invoking the MCP venv's Python. Fails silently."""
        if not self.enabled:
            return

        # Inline script that imports GmailClient and sends
        script = f"""
import sys, json
sys.path.insert(0, {str(GMAIL_MCP_DIR)!r})
from gmail_client import GmailClient
gmail = GmailClient(
    credentials_file={str(GMAIL_MCP_DIR / 'credentials.json')!r},
    token_file={str(GMAIL_MCP_DIR / 'token.json')!r},
)
gmail.send_email(to={EMAIL_TO!r}, subject={('[AlphaOdds] ' + subject)!r}, body={body!r})
print("ok")
"""
        try:
            result = subprocess.run(
                [str(GMAIL_MCP_PYTHON), "-c", script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(f"Email sent: {subject}")
            else:
                logger.warning(f"Email send failed: {result.stderr.strip()}")
        except Exception as e:
            logger.warning(f"Email send failed: {e}")

    # ── Startup ──

    def notify_startup(self, model_count, dry_run):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        mode = "DRY RUN" if dry_run else f"LIVE (${STAKE_SIZE:.0f} bets)"
        self._send(
            f"Bot Started - {mode}",
            f"Paper Trading Bot Started\n"
            f"{'=' * 35}\n"
            f"Mode:    {mode}\n"
            f"Models:  {model_count} components loaded\n"
            f"Time:    {now}\n"
            f"\n"
            f"Monitoring AU greyhound WIN markets.\n"
            f"Bets placed when edge > 3%.\n",
        )

    # ── Bet settled ──

    def notify_bet_settled(self, bet, daily_pnl, total_pnl, daily_record):
        """
        Send notification when a bet is settled.

        Args:
            bet: settled PaperBet object
            daily_pnl: today's cumulative P&L
            total_pnl: all-time cumulative P&L
            daily_record: dict with keys bets_settled, wins, losses, fills
        """
        if not bet.conservative_fill:
            return  # Don't spam about unfilled bets

        result = "WIN" if bet.winner else "LOSE"

        self._send(
            f"{result} ${bet.pnl:+.2f} | Day ${daily_pnl:+.2f} | Total ${total_pnl:+.2f}",
            f"{result} - Bet Settled\n"
            f"{'=' * 35}\n"
            f"Runner:      {bet.runner_id}\n"
            f"Edge:        {bet.edge:.1%}\n"
            f"Odds:        {bet.back_odds:.2f}\n"
            f"Model Prob:  {bet.model_prob:.1%}\n"
            f"Market Prob: {bet.market_prob:.1%}\n"
            f"Fill Price:  {bet.fill_price:.2f}\n"
            f"Bet P&L:     ${bet.pnl:+.2f}\n"
            f"{'-' * 35}\n"
            f"Daily P&L:   ${daily_pnl:+.2f}\n"
            f"Total P&L:   ${total_pnl:+.2f}\n"
            f"Today:       {daily_record['bets_settled']} bets, "
            f"{daily_record['wins']}W / {daily_record['losses']}L\n",
        )

    # ── Kill switch ──

    def notify_kill_switch(self, daily_pnl, total_pnl):
        self._send(
            f"KILL SWITCH - Daily loss ${daily_pnl:.2f}",
            f"KILL SWITCH ACTIVATED\n"
            f"{'=' * 35}\n"
            f"Daily loss has exceeded the maximum.\n"
            f"\n"
            f"Daily P&L:  ${daily_pnl:+.2f}\n"
            f"Total P&L:  ${total_pnl:+.2f}\n"
            f"\n"
            f"Bot has stopped placing bets until\n"
            f"midnight UTC reset.\n",
        )

    # ── Anomaly ──

    def notify_anomaly(self, anomaly_report):
        """Send alert when signals deviate from backtest expectations."""
        findings = anomaly_report["findings"]
        severity = anomaly_report["severity"]
        action = anomaly_report.get("action_taken", "None")
        fixes = anomaly_report.get("fixes_applied", [])

        findings_str = "\n".join(f"  - {f}" for f in findings)
        fixes_str = ""
        if fixes:
            fixes_str = "\nAuto-fixes applied:\n" + "\n".join(f"  - {f}" for f in fixes)

        self._send(
            f"ANOMALY [{severity}] - Paper Trading",
            f"ANOMALY DETECTED - {severity}\n"
            f"{'=' * 35}\n"
            f"Findings:\n{findings_str}\n"
            f"{fixes_str}\n"
            f"\nAction: {action}\n",
        )

    # ── Shutdown ──

    def notify_shutdown(self, summary):
        pnl = summary.get("daily_pnl", 0)
        self._send(
            f"Bot Stopped - Day ${pnl:+.2f} | Total ${summary.get('total_pnl', 0):+.2f}",
            f"Bot Shutdown\n"
            f"{'=' * 35}\n"
            f"Bets placed:  {summary.get('bets_placed', 0)}\n"
            f"Bets filled:  {summary.get('bets_filled', 0)}\n"
            f"Wins:         {summary.get('bets_won', 0)}\n"
            f"Win rate:     {summary.get('win_rate', 0):.1%}\n"
            f"Fill rate:    {summary.get('fill_rate', 0):.1%}\n"
            f"{'-' * 35}\n"
            f"Daily P&L:    ${pnl:+.2f}\n"
            f"Total P&L:    ${summary.get('total_pnl', 0):+.2f}\n"
            f"ROI:          {summary.get('roi_pct', 0):+.1f}%\n",
        )
