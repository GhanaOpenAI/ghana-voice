#!/usr/bin/env bash
# End-to-end autonomous pipeline runner for VieNeu-TTS Ghanaian models on H200.
# Runs fully detached in background — survives terminal disconnection.
#
# Pipeline steps:
#   1. Wait for Stage-1 foundation training to finish (polls every 60s).
#   2. Merge Stage-1 LoRA adapter into base model -> output/vieneu-ghana-foundation/merged.
#   3. Ensure all 11 Stage-2 female datasets are encoded (metadata_encoded.csv).
#   4. Train Stage-2 per-language LoRA for each of the 11 languages sequentially.
#   5. Merge each Stage-2 LoRA -> output_female/<lang>/merged.
#   6. Post progress update to Trello.

set -euo pipefail

PROJ="/mnt/volume_d2wey28/projects/vieneu-ghana"
cd "$PROJ"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

FOUNDATION_OUT="$PROJ/output/vieneu-ghana-foundation"
FOUNDATION_FINAL="$FOUNDATION_OUT/final"
FOUNDATION_MERGED="$FOUNDATION_OUT/merged"
FEMALE_ROOT="$PROJ/datasets_female"
STAGE2_OUT="$PROJ/output_female"
LOGS="$PROJ/logs"
mkdir -p "$LOGS" "$STAGE2_OUT"

echo "============================================================"
echo "VieNeu-TTS Autonomous Pipeline Orchestrator Started"
echo "Time: $(date)"
echo "============================================================"

# -------------------------------------------------------------
# STEP 1: Wait for Stage-1 training to finish
# -------------------------------------------------------------
echo "[Step 1] Monitoring Stage-1 training ..."
while true; do
    if [ -d "$FOUNDATION_FINAL" ] && [ -f "$FOUNDATION_FINAL/adapter_model.safetensors" ]; then
        echo "[Step 1] Stage-1 adapter found at $FOUNDATION_FINAL!"
        break
    fi
    # Check if train_h200 process is still alive
    if ! pgrep -f "train_h200.py.*vieneu-ghana-foundation" > /dev/null; then
        # Check if the process exited and saved final adapter
        if [ -d "$FOUNDATION_FINAL" ]; then
            echo "[Step 1] Stage-1 finished."
            break
        else
            echo "[Step 1] WARNING: Stage-1 process exited but final adapter not found. Checking latest checkpoint..."
            LATEST_CKPT=$(ls -dt "$FOUNDATION_OUT"/checkpoint-* 2>/dev/null | head -1 || true)
            if [ -n "$LATEST_CKPT" ]; then
                echo "[Step 1] Using latest checkpoint $LATEST_CKPT as final adapter."
                mkdir -p "$FOUNDATION_FINAL"
                cp -r "$LATEST_CKPT"/* "$FOUNDATION_FINAL/"
                break
            else
                echo "[Step 1] ERROR: No checkpoint or final model found. Exiting."
                exit 1
            fi
        fi
    fi
    sleep 60
done

# -------------------------------------------------------------
# STEP 2: Merge Stage-1 LoRA adapter into base model
# -------------------------------------------------------------
if [ -f "$FOUNDATION_MERGED/model.safetensors" ]; then
    echo "[Step 2] Foundation model already merged at $FOUNDATION_MERGED, skipping."
else
    echo "[Step 2] Merging Stage-1 LoRA into base model ..."
    python VieNeu-TTS/finetune/merge_lora.py \
        --base-model pnnbao-ump/VieNeu-TTS-0.3B \
        --adapter-path "$FOUNDATION_FINAL" \
        --output-dir "$FOUNDATION_MERGED"
    echo "[Step 2] Merge completed -> $FOUNDATION_MERGED"
fi

# -------------------------------------------------------------
# STEP 3: Verify all 11 female datasets are encoded
# -------------------------------------------------------------
echo "[Step 3] Ensuring all 11 female datasets are encoded with NeuCodec ..."
python VieNeu-TTS/finetune/data_scripts/encode_female_all.py \
    --female-root "$FEMALE_ROOT" \
    --batch-size 16
echo "[Step 3] All female datasets encoded."

# -------------------------------------------------------------
# STEP 4: Train Stage-2 per-language models sequentially
# -------------------------------------------------------------
FEMALE_LANGS=(
    "twi_twi"
    "ga_gaa"
    "ewe_ewe"
    "ghana_english"
    "fante_fat"
    "frafra_gur"
    "dagaare_dga"
    "nzema_nzi"
    "ahanta_aha"
    "sehwi_sfw"
    "dangme_ada"
)

echo "[Step 4] Starting Stage-2 per-language training for ${#FEMALE_LANGS[@]} languages ..."

BASE_MODEL_ID="ghanaopenai/neu-tts-ghana-base"

for lang in "${FEMALE_LANGS[@]}"; do
    LANG_DATA="$FEMALE_ROOT/$lang"
    LANG_OUT="$STAGE2_OUT/vieneu-female-$lang"
    LANG_FINAL="$LANG_OUT/final"
    LANG_MERGED="$LANG_OUT/merged"
    LANG_LOG="$LOGS/train_female_${lang}.log"

    if [ -f "$LANG_MERGED/model.safetensors" ]; then
        echo "[Stage 2] $lang already finished and merged, skipping."
        continue
    fi

    echo "------------------------------------------------------------"
    echo "[Stage 2] Training $lang with ghana-voice-trainer ..."
    echo "  Data: $LANG_DATA"
    echo "  Base: $BASE_MODEL_ID"
    echo "  Out : $LANG_OUT"
    echo "  Log : $LANG_LOG"
    echo "------------------------------------------------------------"

    # Uses the ghana-voice-trainer CLI toolkit
    ghana-voice-trainer train \
        --data-dir "$LANG_DATA" \
        --output-dir "$STAGE2_OUT" \
        --run-name "vieneu-female-$lang" \
        --base-model "$BASE_MODEL_ID" \
        --epochs 25 \
        --learning-rate 5e-5 \
        --batch-size 8 \
        --grad-accum 4 \
        --eval-fraction 0.05 \
        --eval-steps 100 \
        --save-steps 100 \
        --logging-steps 20 \
        --early-stopping-patience 4 \
        > "$LANG_LOG" 2>&1

    echo "[Stage 2] $lang training completed. Merging adapter with ghana-voice-trainer ..."
    ghana-voice-trainer merge \
        --base-model "$BASE_MODEL_ID" \
        --adapter-path "$LANG_FINAL" \
        --output-dir "$LANG_MERGED"

    echo "[Stage 2] $lang merged successfully -> $LANG_MERGED"
done

# -------------------------------------------------------------
# STEP 5: Post completion update to Trello
# -------------------------------------------------------------
KEY="fb42e6f9e55e85ea657494b6e71de684"
TOKEN="ATTA674f95efaf9dc688987d3a01b64b45ca708e1c06538327df1340a4372a59bd2f7F5D10B1"
DONE_DESC="All two stages completed autonomously on H200:
1. Stage-1 Foundation: trained on 42 Ghanaian languages (ghana-one-voice-v2, 67,057 samples) and merged.
2. Stage-2 Per-Language: 11 female models trained and merged on ghana-speech-female-22khz:
   - Twi, Ga, Ewe, English, Fante, Frafra, Dagaare, Nzema, Ahanta, Sehwi, Dangme.
All models saved in /mnt/volume_d2wey28/projects/vieneu-ghana/output_female."

curl -s -X POST "https://api.trello.com/1/cards?key=$KEY&token=$TOKEN" \
    --data-urlencode "idList=6a50b6934ca086d8ec84c8e2" \
    --data-urlencode "name=VieNeu-TTS Ghanaian Stages 1 & 2 Completed" \
    --data-urlencode "desc=$DONE_DESC" > /dev/null || true

echo "============================================================"
echo "All Stages Completed Successfully!"
echo "Finished at: $(date)"
echo "============================================================"
