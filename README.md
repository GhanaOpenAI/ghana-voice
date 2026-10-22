# Ghana Voice 🇬🇭🎙️

**Ghana Voice** is a unified Python package and CLI toolkit for **inference, fine-tuning, and model management** of Ghanaian and African text-to-speech models built on [`ghanaopenai/ghana-voice-base`](https://huggingface.co/ghanaopenai/ghana-voice-base) and [NeuCodec](https://huggingface.co/neuphonic/neucodec).

---

## Foundation Base Model & Naming Convention

- **Multilingual Base Model:** [`ghanaopenai/ghana-voice-base`](https://huggingface.co/ghanaopenai/ghana-voice-base) (244M parameters, trained across 42 African languages in direct zero-reference mode).
- **Language-Level Models:** `ghanaopenai/ghana-voice_{lang_code}`:
  - `ghanaopenai/ghana-voice_twi` (Asante & Akuapem Twi)
  - `ghanaopenai/ghana-voice_fat` (Fante)
  - `ghanaopenai/ghana-voice_gaa` (Ga)
  - `ghanaopenai/ghana-voice_ewe` (Ewe)
  - `ghanaopenai/ghana-voice_dag` (Dagbani)
  - `ghanaopenai/ghana-voice_dga` (Dagaare)
  - `ghanaopenai/ghana-voice_gur` (Frafra / Farefare)
  - `ghanaopenai/ghana-voice_nzi` (Nzema)
  - `ghanaopenai/ghana-voice_aha` (Ahanta)
  - `ghanaopenai/ghana-voice_sfw` (Sehwi)
  - `ghanaopenai/ghana-voice_ada` (Dangme)
  - `ghanaopenai/ghana-voice_eng` (Ghanaian English)

---

## Fine-Tuning Recommendations: LoRA & Data Volume

When fine-tuning from `ghanaopenai/ghana-voice-base` to create dedicated language models or personalized voices:

1. **Recommended Audio Volume:** **Minimum 5 to 10 hours** (~3,000 to 6,000 clean, digit-free speech clips).
   - *Why:* The base model already possesses general acoustic and cross-lingual phonetic capabilities. Providing 5–10 hours ensures sufficient lexical variety and vocal consistency, preventing memorization and overfitting.
2. **LoRA Fine-Tuning:**
   - Always fine-tune with **LoRA (r=16, alpha=32)**.
   - Freezes 100% of the base model weights and trains only **6.4M adapter parameters (2.56%)**.
   - Preserves foundation phonetic stability while specializing dialect timbre and tone contours.

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

# Auto-resolves model from HF (ghanaopenai/ghana-voice_twi or ghana-voice-base):
voice = GhanaVoice(lang="twi", device="cpu")

# Zero-reference synthesis directly from natural text:
audio_array = voice.speak("Akwaaba, yɛbɔ mpae sɛ Onyankopɔn bɛhyira wo.", output_file="akwaaba.wav")
```

### 2. Command Line Interface (`ghana-voice speak`)

```bash
# Direct CPU synthesis:
ghana-voice speak \
  --lang twi \
  --text "Akwaaba, yɛbɔ mpae sɛ Onyankopɔn bɛhyira wo." \
  --device cpu \
  --out twi.wav
```

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
```bash
ghana-voice prep \
  --input-dir path/to/recordings \
  --out-dir datasets/my_data \
  --lang twi \
  --min-dur 3.0 \
  --max-dur 15.0
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
  --epochs 100 \
  --learning-rate 5e-5 \
  --batch-size 8 \
  --grad-accum 4 \
  --early-stopping-patience 10 \
  --eval-cer
```

### 4. Export GGUF for CPU (`export-gguf`)
```bash
ghana-voice export-gguf \
  --model-dir output/my-twi-model/final \
  --out-dir output/my-twi-model/gguf \
  --quant Q4_K_M Q8_0
```

---

## License
Apache-2.0. Built by the GhanaOpenAI team.
