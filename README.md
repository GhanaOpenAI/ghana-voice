# Ghana Voice 🇬🇭🎙️

**Ghana Voice** is a unified Python package and CLI toolkit for **inference, fine-tuning, and model management** of Ghanaian and African text-to-speech models built on [`ghanaopenai/ghana-voice-base`](https://huggingface.co/ghanaopenai/ghana-voice-base) and [NeuCodec](https://huggingface.co/neuphonic/neucodec).

Whether you want to generate high-fidelity speech in Ghanaian languages directly or fine-tune custom voice models on your own audio, `ghana-voice` provides complete zero-reference and in-context capabilities.

---

## Foundation Models & Naming Convention

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

## Installation

```bash
git clone https://github.com/GhanaOpenAI/ghana-voice.git
cd ghana-voice
pip install -r requirements.txt
pip install -e .
```

---

## 🔊 Speech Synthesis & Inference

### 1. Python API (`GhanaVoice`)

```python
from ghana_voice import GhanaVoice

# Auto-resolves model:
# 1. Local path if provided
# 2. Hugging Face repo ghanaopenai/ghana-voice_{lang}
# 3. Base model ghanaopenai/ghana-voice-base
voice = GhanaVoice(lang="twi")

# Direct zero-reference synthesis from natural text:
audio_array = voice.speak("Akwaaba, yɛbɔ mpae sɛ Onyankopɔn bɛhyira wo.", output_file="akwaaba.wav")

# Synthesize other Ghanaian languages:
voice_ga = GhanaVoice(lang="ga")
voice_ga.speak("Nyɔŋmɔ ji mi nɔkwɛlɔ, nɔ ko nɔ ko ehiaŋ mi.", output_file="ga.wav")

voice_ewe = GhanaVoice(lang="ewe")
voice_ewe.speak("Yehowa nye nye kplɔla, naneke mehiãam o.", output_file="ewe.wav")
```

### 2. Command Line Interface (`ghana-voice speak`)

```bash
# Speak Twi:
ghana-voice speak \
  --lang twi \
  --text "Akwaaba, yɛbɔ mpae sɛ Onyankopɔn bɛhyira wo ne wo fiefoɔ nyinaa." \
  --out twi_sample.wav

# Speak Ga:
ghana-voice speak \
  --lang gaa \
  --text "Nyɔŋmɔ ji mi nɔkwɛlɔ, nɔ ko nɔ ko ehiaŋ mi." \
  --out ga_sample.wav

# Speak with a custom fine-tuned checkpoint:
ghana-voice speak \
  --model output/my-custom-model/final \
  --lang twi \
  --text "Me kɔ sukuu da biara na mepɛ sɛ mesua ade." \
  --out custom.wav
```

---

## 🛠️ Fine-Tuning the Base Model

Train new voice models or specialize existing models with Full Parameter SFT or LoRA:

### 1. Prepare Dataset (`prep`)
Filters out transcripts containing digits, strips dead silence with an 80ms onset/release buffer, and converts text to word-separated IPA:

```bash
ghana-voice prep \
  --input-dir path/to/recordings \
  --out-dir datasets/my_data \
  --lang twi \
  --min-dur 3.0 \
  --max-dur 15.0
```

### 2. NeuCodec Encoding (`encode`)
Encodes trimmed audio into discrete acoustic tokens:

```bash
ghana-voice encode \
  --data-dir datasets/my_data \
  --batch-size 32
```

### 3. Training (`train`)
Fine-tune the foundation model with Full Parameter SFT or LoRA:

```bash
ghana-voice train \
  --base-model ghanaopenai/ghana-voice-base \
  --data-dir datasets/my_data \
  --output-dir output \
  --run-name my-custom-model \
  --full-sft \
  --no-ref \
  --epochs 100 \
  --learning-rate 3e-5 \
  --batch-size 8 \
  --grad-accum 4 \
  --early-stopping-patience 10 \
  --eval-cer
```
*Note: Uses `JointPlateauEarlyStoppingCallback` to monitor both `eval_loss` and `eval_cer` simultaneously, stopping only when both metrics have plateaued.*

---

## Language Tiers & Data Requirements

Fine-tuning data requirements depend on the language tier:

### 🌟 Tier 1: Core Ghanaian Languages (Deeply Pre-tuned)
- **Data Requirement:** **2 to 3 hours** (~1,000–1,500 clean clips).
- Languages: `asante_twi`, `akuapem_twi`, `fante`, `ga`, `ewe`, `dagbani`, `dagaare`, `frafra`, `nzema`, `ahanta`, `sehwi`, `dangme`.

### 🌍 Tier 2: Extended Languages + Ghanaian English
- **Data Requirement:** **3 to 6 hours** (~1,500–3,000 clean clips).
- Languages: `ghana_english` (English), `kusaal`, `mampruli`, `gonja`, `kasem`, `buli`, `konkomba`, `hausa`, `bimoba`, `chumburung`, `fulfulde`, `lelemi`, `nkonya`, `ntrubo`, `paasaal`, `sekpele`, `selee`, `sisaala`, `siwu`, `tampulma`, `tem`, `tuwuli`, `vagla`, `deg`, `avatime`, `bassar`, `birifor`, `bissa`, `gikyode`, `kabiye`, `konni`, `nawuri`, `anyin`.

To inspect the full list from the command line:
```bash
ghana-voice list-langs
ghana-voice list-langs --tier 1
ghana-voice list-langs --tier 2
```

---

## License
Apache-2.0. Built by the GhanaOpenAI team.
