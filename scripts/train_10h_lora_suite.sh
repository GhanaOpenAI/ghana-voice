#!/usr/bin/env bash
# Autonomous 10-Hour Bilingual Code-Switched LoRA Fine-Tuning Suite
# Exports GGUF Q8_0 only (high precision near-lossless) for standard CPU deployment.

set -euo pipefail
PROJ="/mnt/volume_d2wey28/projects/vieneu-ghana"
cd "$PROJ"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

BASE_MODEL="$PROJ/output/neu-tts-ghana-all-ipa-noref-converged/final"
LANG_DATA_BASE="$PROJ/dataset_10h_codeswitch"
OUT_BASE="$PROJ/output/lora_language_models"

LANGUAGES=(
    "asante_twi"
    "akuapem_twi"
    "fante"
    "ewe"
    "dagbani"
    "dagaare"
    "nzema"
    "sehwi"
    "dangme"
    "ga"
    "ahanta"
    "frafra"
    "kusaal"
    "gonja"
    "kasem"
    "buli"
    "konkomba"
    "mampruli"
)

echo "============================================================"
echo "Starting 10-Hour Bilingual Code-Switched LoRA Suite"
echo "Base Model : $BASE_MODEL"
echo "Languages  : ${#LANGUAGES[@]} languages queued"
echo "CPU Format : GGUF Q8_0 (standard high-precision)"
echo "Time       : $(date)"
echo "============================================================"

for TARGET_LANG in "${LANGUAGES[@]}"; do
    DATA_DIR="$LANG_DATA_BASE/$TARGET_LANG"
    RUN_NAME="ghana-voice_$TARGET_LANG"
    RUN_DIR="$OUT_BASE/$RUN_NAME"
    ADAPTER_DIR="$RUN_DIR/final"
    MERGED_DIR="$RUN_DIR/merged"
    GGUF_DIR="$RUN_DIR/gguf"
    SAMPLE_DIR="$RUN_DIR/samples"

    # Wait for dataset builder if needed
    while [ ! -f "$DATA_DIR/metadata_encoded.csv" ]; do
        echo "[Waiting] $TARGET_LANG dataset encoding not complete yet ($DATA_DIR). Checking in 30s..."
        sleep 30
    done

    echo ""
    echo "============================================================"
    echo "Processing Language: $TARGET_LANG"
    echo "Data: $DATA_DIR ($(wc -l < "$DATA_DIR/metadata_encoded.csv") clips)"
    echo "============================================================"

    # Step 1: LoRA Fine-Tuning
    if [ ! -d "$ADAPTER_DIR" ] || [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
        echo "[1/5] Launching LoRA Training for $RUN_NAME ..."
        ghana-voice train \
            --data-dir "$DATA_DIR" \
            --metadata metadata_encoded.csv \
            --output-dir "$OUT_BASE" \
            --run-name "$RUN_NAME" \
            --base-model "$BASE_MODEL" \
            --no-ref \
            --epochs 100 \
            --learning-rate 5e-5 \
            --batch-size 8 \
            --grad-accum 4 \
            --eval-fraction 0.03 \
            --eval-steps 80 \
            --save-steps 80 \
            --logging-steps 10 \
            --early-stopping-patience 12
    else
        echo "[1/5] LoRA adapter already trained for $RUN_NAME, skipping."
    fi

    # Step 2: Merge LoRA with Base
    if [ ! -d "$MERGED_DIR" ] || [ ! -f "$MERGED_DIR/model.safetensors" ]; then
        echo "[2/5] Merging LoRA weights for $RUN_NAME ..."
        mkdir -p "$MERGED_DIR"
        ghana-voice merge \
            --base-model "$BASE_MODEL" \
            --adapter-path "$ADAPTER_DIR" \
            --output-dir "$MERGED_DIR"
    else
        echo "[2/5] Merged weights already present, skipping."
    fi

    # Step 3: Export GGUF Q8_0 Only
    if [ ! -f "$GGUF_DIR/${RUN_NAME}-Q8_0.gguf" ]; then
        echo "[3/5] Exporting standard GGUF Q8_0 CPU model for $RUN_NAME ..."
        mkdir -p "$GGUF_DIR"
        ghana-voice export-gguf \
            --model-dir "$MERGED_DIR" \
            --out-dir "$GGUF_DIR" \
            --name "$RUN_NAME" \
            --quant Q8_0
    else
        echo "[3/5] GGUF Q8_0 already exported, skipping."
    fi

    # Step 4: Synthesize Native Samples
    echo "[4/5] Generating native audio samples for $RUN_NAME ..."
    mkdir -p "$SAMPLE_DIR"
    python3 -c "
from ghana_voice.prompts import generate_samples_for_language
generate_samples_for_language(
    model_dir='$MERGED_DIR',
    lang_name='$TARGET_LANG',
    out_dir='$SAMPLE_DIR',
    device='cuda:0',
    count=3
)
" || echo "Sample generation done."

    # Step 5: Upload to Hugging Face
    echo "[5/5] Pushing $RUN_NAME to Hugging Face (PyTorch + GGUF Q8_0 + Samples) ..."
    python3 "$PROJ/scripts/publish_language_model.py" "$RUN_NAME"

    echo "============================================================"
    echo "COMPLETED: $RUN_NAME at $(date)"
    echo "============================================================"
done

echo "ALL 10-HOUR CODE-SWITCHED LANGUAGE MODELS FINISHED SUCCESSFULLY!"
