#!/usr/bin/env bash
#
# Download model artifacts from Spartan to local models/ directory.
#
# Prerequisites:
#   1. Run save_model_artifacts.py on Spartan first (generates manifest + isotonic calibrators)
#   2. Passwordless SSH to spartan.hpc.unimelb.edu.au
#
# Usage:
#   bash _05_paper_trading/scripts/download_models.sh

set -euo pipefail

SPARTAN="adidishe@spartan.hpc.unimelb.edu.au"
SPARTAN_RES="/data/projects/punim2039/alpha_odds/res"
SPARTAN_ARTIFACTS="${SPARTAN_RES}/paper_trading_artifacts"
LOCAL_MODELS="$(dirname "$0")/../../models"

echo "=== Downloading Paper Trading Model Artifacts ==="
echo "From: ${SPARTAN}:${SPARTAN_RES}"
echo "To:   ${LOCAL_MODELS}"

# Create local directories
mkdir -p "${LOCAL_MODELS}/normalization"

# 1. Download manifest
echo ""
echo "--- Downloading manifest ---"
scp "${SPARTAN}:${SPARTAN_ARTIFACTS}/manifest.json" "${LOCAL_MODELS}/manifest.json"

# 2. Download normalization params
echo ""
echo "--- Downloading normalization parameters ---"
scp -r "${SPARTAN}:${SPARTAN_ARTIFACTS}/normalization/" "${LOCAL_MODELS}/normalization/"

# 3. Read manifest and download model files
echo ""
echo "--- Downloading V1 models ---"
for T in 0 1 2 3; do
    # Parse config names from manifest using Python
    CONFIGS=$(python3 -c "
import json
with open('${LOCAL_MODELS}/manifest.json') as f:
    m = json.load(f)
for c in m['v1'].get('t${T}', []):
    print(c)
")
    for CONFIG in $CONFIGS; do
        echo "  V1 t${T}/${CONFIG}"
        mkdir -p "${LOCAL_MODELS}/v1/t${T}/${CONFIG}"
        scp "${SPARTAN}:${SPARTAN_RES}/win_model/t${T}/${CONFIG}/xgboost_model.json" \
            "${LOCAL_MODELS}/v1/t${T}/${CONFIG}/" 2>/dev/null || echo "    WARNING: xgboost_model.json not found"
    done
done

echo ""
echo "--- Downloading V2 models ---"
for T in 0 1 2 3; do
    CONFIGS=$(python3 -c "
import json
with open('${LOCAL_MODELS}/manifest.json') as f:
    m = json.load(f)
for c in m['v2'].get('t${T}', []):
    print(c)
")
    for CONFIG in $CONFIGS; do
        echo "  V2 t${T}/${CONFIG}"
        mkdir -p "${LOCAL_MODELS}/v2/t${T}/${CONFIG}"
        # XGBoost model
        scp "${SPARTAN}:${SPARTAN_RES}/win_model_v2/t${T}/${CONFIG}/xgboost_model.json" \
            "${LOCAL_MODELS}/v2/t${T}/${CONFIG}/" 2>/dev/null || echo "    WARNING: xgboost_model.json not found"
        # LightGBM model
        scp "${SPARTAN}:${SPARTAN_RES}/win_model_v2/t${T}/${CONFIG}/lightgbm_model.txt" \
            "${LOCAL_MODELS}/v2/t${T}/${CONFIG}/" 2>/dev/null || echo "    WARNING: lightgbm_model.txt not found"
        # Isotonic calibrator
        scp "${SPARTAN}:${SPARTAN_RES}/win_model_v2/t${T}/${CONFIG}/isotonic_calibrator.pkl" \
            "${LOCAL_MODELS}/v2/t${T}/${CONFIG}/" 2>/dev/null || echo "    WARNING: isotonic_calibrator.pkl not found"
    done
done

# 4. Summary
echo ""
echo "=== Download Complete ==="
echo "Model directory contents:"
find "${LOCAL_MODELS}" -type f | head -30
echo "..."
echo "Total files: $(find "${LOCAL_MODELS}" -type f | wc -l)"
echo ""
echo "Next steps:"
echo "  1. Verify manifest: cat ${LOCAL_MODELS}/manifest.json"
echo "  2. Check all models downloaded: find ${LOCAL_MODELS} -name '*.json' -o -name '*.txt' -o -name '*.pkl' | wc -l"
echo "  3. Run paper trader: cd _05_paper_trading && python main.py --dry-run"
