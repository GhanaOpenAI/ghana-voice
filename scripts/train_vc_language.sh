#!/bin/bash
# Fine-tune one language LoRA on africa-female-vc converted audio only,
# optionally topped up with VC-converted Ghanaian English as separate clips.
# Training flags match the run that produced the good 10h LoRA samples.
#
# Usage: scripts/train_vc_language.sh <lang> [english_ratio] [hours]
#   e.g. scripts/train_vc_language.sh asante_twi 0.3 10   (english_ratio default 0 = language only)
#
# Env overrides: PROJ, BASE_MODEL, VC_ROOT
set -e

LANG_NAME="${1:?usage: $0 <lang> [english_ratio] [hours]}"
ENGLISH_RATIO="${2:-0}"
HOURS="${3:-10}"

PROJ="${PROJ:-/mnt/volume_d2wey28/projects/vieneu-ghana}"
BASE_MODEL="${BASE_MODEL:-$PROJ/output/neu-tts-ghana-all-ipa-noref-converged/final}"
VC_ROOT="${VC_ROOT:-/mnt/volume_d2wey28/data/harvest_cache}"

EN_PCT=$(python3 -c "print(int(round($ENGLISH_RATIO * 100)))")
DATA_DIR="$PROJ/dataset_vc/${LANG_NAME}_en${EN_PCT}"
RUN_NAME="ghana-voice_${LANG_NAME}_vc_en${EN_PCT}"

if [ ! -f "$DATA_DIR/metadata_encoded.csv" ]; then
    ghana-voice build-dataset \
        --lang "$LANG_NAME" \
        --out-dir "$DATA_DIR" \
        --hours "$HOURS" \
        --english-ratio "$ENGLISH_RATIO" \
        --vc-root "$VC_ROOT"
fi

ghana-voice train \
    --data-dir "$DATA_DIR" \
    --metadata metadata_encoded.csv \
    --output-dir "$PROJ/output/lora_language_models" \
    --run-name "$RUN_NAME" \
    --base-model "$BASE_MODEL" \
    --no-ref \
    --eval-cer \
    --epochs 40 \
    --learning-rate 5e-5 \
    --batch-size 8 \
    --grad-accum 4 \
    --eval-fraction 0.03 \
    --eval-steps 80 \
    --save-steps 80 \
    --logging-steps 20 \
    --early-stopping-patience 10
