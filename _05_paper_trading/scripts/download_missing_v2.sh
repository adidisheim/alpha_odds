#!/usr/bin/env bash
# Download missing V2 model files from Spartan
set -e

SPARTAN="adidishe@spartan.hpc.unimelb.edu.au"
REMOTE_BASE="/data/projects/punim2039/alpha_odds/res/win_model_v2"
LOCAL_BASE="models/v2"

cd "$(dirname "$0")/../.."

TOTAL=0
DOWNLOADED=0
FAILED=0

for T in 0 1 2 3; do
    # Read configs from manifest.json for this t_def
    CONFIGS=$(python3 -c "import json; m=json.load(open('models/manifest.json')); print('\n'.join(m['v2']['t${T}']))")

    for CONFIG in $CONFIGS; do
        LOCAL_DIR="${LOCAL_BASE}/t${T}/${CONFIG}"
        mkdir -p "$LOCAL_DIR"

        for FILE in xgboost_model.json lightgbm_model.txt isotonic_calibrator.pkl; do
            TOTAL=$((TOTAL + 1))
            LOCAL_FILE="${LOCAL_DIR}/${FILE}"

            if [ -f "$LOCAL_FILE" ] && [ -s "$LOCAL_FILE" ]; then
                continue
            fi

            REMOTE_FILE="${REMOTE_BASE}/t${T}/${CONFIG}/${FILE}"
            echo "Downloading t${T}/${CONFIG}/${FILE}..."
            if scp "${SPARTAN}:${REMOTE_FILE}" "${LOCAL_FILE}" 2>&1; then
                DOWNLOADED=$((DOWNLOADED + 1))
            else
                echo "  FAILED: t${T}/${CONFIG}/${FILE}"
                FAILED=$((FAILED + 1))
                rm -f "$LOCAL_FILE"
            fi
        done
    done
done

echo ""
echo "=== Summary ==="
echo "Total files needed: $TOTAL"
echo "Downloaded: $DOWNLOADED"
echo "Failed: $FAILED"

# Verify counts
echo ""
echo "=== File counts per t_def ==="
for T in 0 1 2 3; do
    COUNT=$(find "${LOCAL_BASE}/t${T}" -type f | wc -l)
    echo "t${T}: ${COUNT} files"
done
