# Ghana Voice 🇬🇭🎙️

**Ghana Voice** is a unified Python package and CLI toolkit for **inference, fine-tuning, voice conversion, and model management** of Ghanaian and African text-to-speech models built on [`ghanaopenai/ghana-voice-base`](https://huggingface.co/ghanaopenai/ghana-voice-base) and [NeuCodec](https://huggingface.co/neuphonic/neucodec).

---

## Foundation Models & Naming Convention

- **Multilingual Base Model:** [`ghanaopenai/ghana-voice-base`](https://huggingface.co/ghanaopenai/ghana-voice-base) (244M parameters, trained across 42 African languages in direct zero-reference mode).
- **Language-Level Models (Real Speech):** `ghanaopenai/ghana-voice_{lang_code}`
- **Language-Level Models (Voice Converted / Persona):** `ghanaopenai/ghana-voice_{lang_code}_{voice_name}` (e.g. `ghana-voice_twi_clear-high-slow`)

All models are published under the official [**Ghana Voice Collection**](https://huggingface.co/collections/ghanaopenai/ghana-voice-6ada4ff31e447e9db5288d08).

---

## Fine-Tuning Recommendations: LoRA & Audio Volume

When fine-tuning from `ghanaopenai/ghana-voice-base` to create dedicated language models or personalized voices:

1. **Recommended Audio Volume:** **Minimum 5 to 10 hours** (~3,000 to 6,000 clean, digit-free speech clips).
2. **LoRA Fine-Tuning:**
   - Always fine-tune with **LoRA (r=16, alpha=32)**.
   - Freezes 100% of the base model weights and trains only **6.4M adapter parameters (2.56%)**.
   - Preserves foundation phonetic stability while specializing dialect timbre and tone contours.
3. **Unified Single-Speaker Voice Conversion (Africa Female VC)**:
   - All language fine-tuning uses a consistent single-speaker voice persona to guarantee crisp, expressive, non-robotic zero-reference synthesis.
   - Users choose from **21 African female voices** (e.g., `clear-high-slow` for native Twi female voice, `warm-high` for Hausa, `bright-high` for Yoruba) via `--vc-voice <voice_name>` (default: `clear-high-slow`).
   - English is optional: `--english-ratio 0.3` adds VC-converted Ghanaian English as separate whole clips (30% of your data's hours), converted into the same voice. Leave it at the default `0` to train on your own data only.

To inspect all 21 available target voices from the command line:
```bash
ghana-voice list-vc-voices
```

---

## Installation

```bash
git clone https://github.com/GhanaOpenAI/ghana-voice.git
cd ghana-voice
pip install -r requirements.txt
pip install -e .
```

---

## 🔊 Speech Synthesis & Inference (CLI, Python API, Web UI, REST)

### 1. Python API (`GhanaVoice`)

```python
from ghana_voice import GhanaVoice

# Auto-resolves the language model from HF (ghanaopenai/ghana-voice_asante_twi), else ghana-voice-base.
# Runs the PyTorch weights, recommended for production (GPU if available, else CPU):
voice = GhanaVoice(lang="asante_twi")

# Zero-reference synthesis directly from natural text:
audio_array = voice.speak("Akwaaba, yɛbɔ mpae sɛ Onyankopɔn bɛhyira wo.", output_file="akwaaba.wav")
```

### 2. Command Line Interface (`ghana-voice speak`)

```bash
ghana-voice speak \
  --lang asante_twi \
  --text "Akwaaba, yɛbɔ mpae sɛ Onyankopɔn bɛhyira wo." \
  --out twi.wav
```

Text is converted to word-separated IPA with africa-g2p, the same format the models were trained on,
and spoken in sentence chunks: long text is split at sentence ends, and short sentences are merged
with their neighbours, because very short prompts make the model repeat itself.

> **PyTorch vs GGUF**: use the PyTorch weights in production. The GGUF files (`--gguf`, or
> `GhanaVoice(..., use_gguf=True)`) are less reliable; use them with care, only where resources are
> constrained, and compare against PyTorch output.

### 3. Interactive Web UI (`ghana-voice web`)

```bash
ghana-voice web --port 7860
```

### 4. OpenAI-Compatible REST API Server (`ghana-voice serve`)

```bash
ghana-voice serve --port 8000
```

---

## 🛠️ Fine-Tuning the Base Model with LoRA

### 1. Prepare Dataset (`prep`)
`--input-dir` holds your audio plus a `metadata.csv` of `filename|text` lines.
```bash
# Your data only, converted into the voice you pick:
ghana-voice prep \
  --input-dir path/to/recordings \
  --out-dir datasets/my_data \
  --lang twi \
  --vc-voice clear-high-slow

# Same, plus VC-converted Ghanaian English worth 30% of your data's hours:
ghana-voice prep \
  --input-dir path/to/recordings \
  --out-dir datasets/my_data \
  --lang twi \
  --vc-voice clear-high-slow \
  --english-ratio 0.3
```
`--english-ratio` requires `--vc-voice`, so every clip shares one voice. English clips come from
[`ghanaopenai/ghana-speech-female-22khz`](https://huggingface.co/datasets/ghanaopenai/ghana-speech-female-22khz)
(`clear-high-slow`) and are converted again when you pick another voice.

Seed-VC pins its own torch/transformers versions, so install `africa-female-vc` in a separate
environment and point ghana-voice at it; conversion then runs in a worker process:
```bash
export GHANA_VOICE_VC_PYTHON=/path/to/seedvc-env/bin/python
```

To build a dataset straight from the published VC speech instead of your own recordings
(already encoded, so skip `encode`):
```bash
ghana-voice build-dataset --lang asante_twi --out-dir datasets/asante_twi --hours 10 --english-ratio 0.3
```

### 2. NeuCodec Encoding (`encode`)
```bash
ghana-voice encode \
  --data-dir datasets/my_data \
  --batch-size 32
```

### 3. LoRA Training (`train`)
```bash
ghana-voice train \
  --base-model ghanaopenai/ghana-voice-base \
  --data-dir datasets/my_data \
  --output-dir output \
  --run-name my-twi-model \
  --no-ref \
  --epochs 40 \
  --learning-rate 5e-5 \
  --batch-size 8 \
  --grad-accum 4 \
  --eval-fraction 0.03 \
  --eval-steps 80 \
  --save-steps 80 \
  --early-stopping-patience 10
```
Early stopping and the released checkpoint follow eval loss. `--eval-cer` adds an OmniASR CER check,
but it is only meaningful for languages the ASR model reads well (e.g. Twi).

### 4. Export GGUF for CPU (`export-gguf`)
```bash
ghana-voice export-gguf \
  --model-dir output/my-twi-model/final \
  --out-dir output/my-twi-model/gguf \
  --quant Q8_0
```
> **Note on GGUF**: GGUF exports (Q8_0, Q4_K_M) are for resource-constrained setups and are less reliable
> than the PyTorch weights; use them with care. Prefer Q8_0 over Q4_K_M, and use PyTorch in production.

---

## Example Fine-Tunes

Nine language models fine-tuned from `ghana-voice-base` on 10 h of voice-converted speech each
(LoRA, eval-loss early stopping), as worked examples of the recipe above:
[Asante Twi](https://huggingface.co/ghanaopenai/ghana-voice_asante_twi),
[Akuapem Twi](https://huggingface.co/ghanaopenai/ghana-voice_akuapem_twi),
[Anyin](https://huggingface.co/ghanaopenai/ghana-voice_anyin),
[Avatime](https://huggingface.co/ghanaopenai/ghana-voice_avatime),
[Bassar](https://huggingface.co/ghanaopenai/ghana-voice_bassar),
[Bimoba](https://huggingface.co/ghanaopenai/ghana-voice_bimoba),
[Birifor](https://huggingface.co/ghanaopenai/ghana-voice_birifor),
[Bissa](https://huggingface.co/ghanaopenai/ghana-voice_bissa) and
[Buli](https://huggingface.co/ghanaopenai/ghana-voice_buli).

## Known Limitations

- **Very short inputs.** The models were trained on 2.5-16 s clips and do not reliably stop after a
  one- to three-word phrase: on the nsanku-TTS benchmark the Akuapem Twi model read full sentences
  well (Bible sentences: CER 0.32, SpeechBERTScore 0.71) but babbled on short finance phrases such as
  "Medɔ wo". `GhanaVoice.synthesize` merges short sentences with their neighbours to avoid this; a
  phrase on its own has nothing to merge with. Fine-tunes that need short phrases should include
  short utterances in their training data.
- **Base model runaways.** `ghana-voice-base` keeps talking past the end of the sentence more often
  than the fine-tunes (about a third of Dangme benchmark sentences); fine-tune for production use.
- **Encode at 16 kHz.** NeuCodec's encoder expects 16 kHz audio (`ghana-voice encode` does this).
  Encoding audio at any other rate gives codes at the wrong rate and models that never stop.

## License
Apache-2.0. Built by the GhanaOpenAI team.
