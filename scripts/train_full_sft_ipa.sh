#!/usr/bin/env bash
# Full Parameter SFT (100% trainable, NO LoRA) on the original base model
# pnnbao-ump/VieNeu-TTS-0.3B with silence-pruned, word-separated IPA (no ref audio).

set -euo pipefail
PROJ="/mnt/volume_d2wey28/projects/vieneu-ghana"
cd "$PROJ"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Neu-TTS Ghana: Full SFT (552M params, NO LoRA) on Base Model"
echo "Base Model: fantaxy/neutts-air (Original Upstream Pretrained Base)"
echo "Dataset   : $PROJ/dataset/metadata_ipa_noref.csv (68,842 samples, 43 langs)"
echo "Mode      : Zero-Ref Direct TTS (No Ref Audio)"
echo "Phonetics : Word-Separated IPA (Punctuation Preserved, No Tags)"
echo "Objective : Online ASR CER via Meta OmniASR 300M CTC"
echo "Time      : $(date)"
echo "============================================================"

META="$PROJ/dataset/metadata_ipa_noref.csv"
if [ ! -f "$META" ]; then
    echo "ERROR: $META not found! Exiting."
    exit 1
fi

TOTAL_ROWS=$(wc -l < "$META")
echo "Dataset verified with $TOTAL_ROWS samples! Launching Full SFT ..."

ghana-voice-trainer train \
    --data-dir "$PROJ/dataset" \
    --metadata metadata_ipa_noref.csv \
    --output-dir "$PROJ/output" \
    --run-name "neu-tts-ghana-full-sft" \
    --base-model "fantaxy/neutts-air" \
    --full-sft \
    --no-ref \
    --epochs 40 \
    --learning-rate 3e-5 \
    --batch-size 8 \
    --grad-accum 4 \
    --eval-fraction 0.02 \
    --eval-steps 250 \
    --save-steps 250 \
    --logging-steps 25 \
    --early-stopping-patience 10 \
    --eval-cer

echo "============================================================"
echo "Full Parameter SFT Completed Successfully!"
echo "Model weights saved -> $PROJ/output/neu-tts-ghana-full-sft/final"
echo "Finished at: $(date)"
echo "============================================================"
