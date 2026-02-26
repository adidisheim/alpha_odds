# Connecting to the Betfair API

This guide covers setting up Betfair API access for the paper trading system. These steps have already been completed for the main development account — follow this if setting up on a new machine (e.g., AWS VM).

## Prerequisites

- A Betfair Australia account with funds deposited
- The `certs/` directory is already in this repo (SSL cert + key)
- The app key has already been created (Application Id: 138885, app name: `alpha_odds`)

## What's Already Done (don't repeat)

1. **Betfair account created** at betfair.com.au
2. **App key created** via the API-NG Demo Tool at `apps.betfair.com/visualisers/api-ng-account-operations/`
   - Delay key (for dev/testing): `Uebo4sXlb5USEutl` (active)
   - Live key (for production): `I2HBofEU1n1OJOv9` (inactive — apply when ready)
3. **SSL certificate generated** and stored in `certs/client-2048.crt` and `certs/client-2048.key`
4. **Certificate uploaded** to Betfair account under My Account > Account Details > My Security > Automated Betting Program Access (Status: On)

## Setup on a New Machine

### Step 1: Pull the repo

```bash
git pull
```

The `certs/` directory with `client-2048.crt` and `client-2048.key` is included in the repo.

### Step 2: Create the `.env` file

The `.env` file is git-ignored for security. Create it manually:

```bash
cp .env.example .env
```

Then edit `.env` and fill in:

```
BETFAIR_USERNAME=antoinedidisheim@gmail.com
BETFAIR_PASSWORD=<your_betfair_password>
BETFAIR_APP_KEY=Uebo4sXlb5USEutl
BETFAIR_CERT_PATH=./certs/client-2048.crt
BETFAIR_KEY_PATH=./certs/client-2048.key
```

### Step 3: Verify the connection

Test that cert-based login works:

```bash
curl -s -X POST https://identitysso-cert.betfair.com.au/api/certlogin \
  -d "username=antoinedidisheim@gmail.com&password=<your_password>" \
  -H "X-Application: Uebo4sXlb5USEutl" \
  --cert certs/client-2048.crt \
  --key certs/client-2048.key
```

Expected response:
```json
{"sessionToken":"<token>","loginStatus":"SUCCESS"}
```

### Step 4: Install dependencies and run

```bash
cd _05_paper_trading
pip install -r requirements.txt
python main.py --dry-run
```

## API Endpoints (Australia)

| Purpose | URL |
|---------|-----|
| Cert login | `https://identitysso-cert.betfair.com.au/api/certlogin` |
| Interactive login | `https://identitysso.betfair.com.au/api/login` |
| Keep-alive | `https://identitysso.betfair.com.au/api/keepAlive` |
| Betting API | `https://api.betfair.com/exchange/betting/rest/v1.0/` |
| Streaming | `stream-api.betfair.com:443` |

## Key Details

- **App key type**: Delay key (provides delayed odds data — fine for paper trading)
- **Session tokens expire** after 20 minutes — the code sends keep-alive every 15 minutes
- **Certificate expires** in 365 days from 2026-02-26 (regenerate before 2027-02-26)
- **To get a live key**: apply through Betfair developer program once paper trading is validated

## Troubleshooting

| Error | Fix |
|-------|-----|
| `CERT_AUTH_REQUIRED` | Certificate not uploaded to Betfair account, or wrong cert file |
| `NO_SESSION` | Session expired — re-login |
| `INVALID_APP_KEY` | Wrong app key in `.env` |
| `APP_KEY_CREATION_FAILED` | App key already exists — use `getDeveloperAppKeys` to retrieve it |
| `INVALID_SESSION_INFORMATION` | Login first, or session token expired |
