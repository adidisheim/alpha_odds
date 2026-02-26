"""Send the Alpha Odds report PDF via Gmail."""
import os
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

MCP_DIR = "/Users/adidisheim/Dropbox/side_project/claude-email-mcp"
TOKEN_FILE = os.path.join(MCP_DIR, "token.json")
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
]

PDF_PATH = "/Users/adidisheim/Dropbox/side_project/alpha_odds/res/alpha_odds_report.pdf"
TO = "antoinedidisheim@gmail.com"
SUBJECT = "Alpha Odds — Updated Report (Odds & Liquidity Analysis)"

BODY = """\
Hi,

Attached is the updated Alpha Odds report with 3 new pages:

- Page 7: Odds & Liquidity Charts — profit by odds range, ROI by odds range, profit concentration, robustness under odds caps
- Page 8: Detailed Tables — odds bucket breakdown, robustness stats, profit concentration analysis, strategy recommendations
- Page 9: Spread & Liquidity Charts — ROI by spread quintile, profit share by spread, edge vs odds scatter, odds distribution

Key findings:
- Odds 15-30 (1.9% of bets) generate 37% of profit — high concentration
- Top 10 winning bets = 58% of total profit
- But strategy IS robust: capping at odds <= 10 still gives z=3.01, p=0.001
- Recommendations: cap odds at 20-30, consider min odds of 3, tiered edge thresholds

Best,
Claude Code
"""

creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
if creds.expired and creds.refresh_token:
    creds.refresh(Request())
service = build("gmail", "v1", credentials=creds)

msg = MIMEMultipart()
msg["to"] = TO
msg["subject"] = SUBJECT
msg.attach(MIMEText(BODY, "plain"))

with open(PDF_PATH, "rb") as f:
    attachment = MIMEApplication(f.read(), _subtype="pdf")
    attachment.add_header("Content-Disposition", "attachment", filename="alpha_odds_report.pdf")
    msg.attach(attachment)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
print(f"Email sent! Message ID: {sent['id']}")
