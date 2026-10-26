#!/usr/bin/env bash
# Autonomous Training Suite for 70/30 Sample-Balanced Asante Twi:
# - 70% pure Asante Twi samples (clear-high-slow)
# - 30% brisk Ghanaian English samples (clear-high-slow)
# - Pure sample-level mixing (NO audio splicing, independent natural sentences)

set -euo pipefail
PROJ="/mnt/volume_d2wey28/projects/vieneu-ghana"
cd "$PROJ"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

DATA_DIR="$PROJ/dataset_70_30_asante_twi"
RUN_NAME="ghana-voice_asante_twi_70_30"
OUT_BASE="$PROJ/output/lora_language_models"
BASE_MODEL="$PROJ/output/neu-tts-ghana-all-ipa-noref-converged/final"

RUN_DIR="$OUT_BASE/$RUN_NAME"
ADAPTER_DIR="$RUN_DIR/final"
MERGED_DIR="$RUN_DIR/merged"
GGUF_DIR="$RUN_DIR/gguf"
SAMPLE_DIR="$RUN_DIR/samples"

# 1. Wait for dataset builder to finish writing metadata
while [ ! -f "$DATA_DIR/metadata_encoded.csv" ]; do
    echo "[Waiting] 70/30 dataset builder in progress... checking in 15s"
    sleep 15
done

echo "============================================================"
echo "Starting 70/30 Sample-Balanced Asante Twi Training"
echo "Voice Persona : clear-high-slow (215-261 Hz female voice)"
echo "Data Mix      : 70% Twi / 30% English (Standalone Sentences)"
echo "Data Directory: $DATA_DIR ($(wc -l < "$DATA_DIR/metadata_encoded.csv") clips)"
echo "Run Name      : $RUN_NAME"
echo "============================================================"

# Step 1: LoRA Fine-Tuning
if [ ! -d "$ADAPTER_DIR" ] || [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
    ghana-voice train \
        --data-dir "$DATA_DIR" \
        --metadata metadata_encoded.csv \
        --output-dir "$OUT_BASE" \
        --run-name "$RUN_NAME" \
        --base-model "$BASE_MODEL" \
        --no-ref \
        --epochs 50 \
        --learning-rate 5e-5 \
        --batch-size 8 \
        --grad-accum 4 \
        --eval-fraction 0.03 \
        --eval-steps 80 \
        --save-steps 80 \
        --logging-steps 10 \
        --early-stopping-patience 10
fi

# Step 2: Merge LoRA with Base
echo "Merging LoRA weights..."
mkdir -p "$MERGED_DIR"
ghana-voice merge \
    --base-model "$BASE_MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    --output-dir "$MERGED_DIR"

# Step 3: Export standard GGUF Q8_0
echo "Exporting GGUF Q8_0..."
mkdir -p "$GGUF_DIR"
ghana-voice export-gguf \
    --model-dir "$MERGED_DIR" \
    --out-dir "$GGUF_DIR" \
    --name "$RUN_NAME" \
    --quant Q8_0

# Step 4: Synthesize Native Samples (Both PyTorch GPU and GGUF Q8_0 CPU)
echo "Synthesizing test audio samples in both PyTorch (GPU) and GGUF Q8_0 (CPU)..."
mkdir -p "$SAMPLE_DIR"
python3 -c "
from ghana_voice.prompts import generate_samples_for_language
generate_samples_for_language(
    model_dir='$MERGED_DIR',
    lang_name='asante_twi',
    out_dir='$SAMPLE_DIR',
    gguf_path='$GGUF_DIR/${RUN_NAME}-Q8_0.gguf',
    device='cuda:0',
    count=3
)
"

# Step 5: Upload to Hugging Face
echo "Pushing $RUN_NAME to Hugging Face Hub (PyTorch + GGUF Q8_0 + Samples)..."
python3 "$PROJ/scripts/publish_language_model.py" "$RUN_NAME"

echo "============================================================"
echo "70/30 ASANTE TWI MODEL COMPLETED SUCCESSFULLY!"
echo "Samples saved in: $SAMPLE_DIR"
echo "============================================================"
