# Ghana Voice Trainer 🇬🇭🎙️

**Ghana Voice Trainer** is an end-to-end toolkit for training, fine-tuning, and evaluating Ghanaian and West African text-to-speech models built on [`ghanaopenai/neu-tts-ghana-base`](https://huggingface.co/ghanaopenai/neu-tts-ghana-base) and [NeuCodec](https://huggingface.co/neuphonic/neucodec).

Whether you want to clone your own voice or train a dedicated single-language model (Twi, Fante, Ga, Ewe, Dagbani, Ghanaian English, Kusaal, etc.), this toolkit handles the entire pipeline:
1. **Phonemization** via [africa-g2p](https://github.com/AfriSpeech/africa-g2p) into word-separated IPA phonetics (no intra-word spaces, preserved punctuation, zero language tags).
2. **Audio Encoding** into discrete acoustic codes via NeuCodec (GPU-batched, auto-resuming).
3. **Full SFT or LoRA Fine-Tuning** with dynamic per-batch sequence padding (3x faster), cosine decay scheduling, and **Joint Loss + CER Plateau Early Stopping**.
4. **Adapter Merging** into a standalone model ready for deployment.
5. **Speech Synthesis & Intelligibility Evaluation** with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) OmniASR 300M (measuring Character Error Rate).

---

## Foundation Base Model

All training builds upon our official Ghanaian foundation model:
- **Base Model:** [`ghanaopenai/neu-tts-ghana-base`](https://huggingface.co/ghanaopenai/neu-tts-ghana-base) (244M parameters, trained across 42 Ghanaian and West African languages in direct zero-reference mode).

---

## Language Tiers & Data Requirements

Because `ghanaopenai/neu-tts-ghana-base` has already undergone extensive multilingual zero-reference training, fine-tuning data requirements depend on the language tier:

### 🌟 Tier 1: Core Ghanaian Languages (Deeply Pre-tuned)
- **Data Requirement:** **2 to 3 hours** (~1,000–1,500 clean clips).
- These languages are already deeply anchored in the base foundation model weights and need minimal data to specialize:
  - **Asante Twi** (`asante_twi` / `twi`)
  - **Akuapem Twi** (`akuapem_twi` / `twi`)
  - **Fante** (`fante` / `fat`)
  - **Ga** (`ga` / `gaa`)
  - **Ewe** (`ewe` / `ewe`)
  - **Dagbani** (`dagbani` / `dag`)
  - **Southern Dagaare** (`dagaare` / `dga`)
  - **Frafra / Farefare** (`frafra` / `gur`)
  - **Nzema** (`nzema` / `nzi`)
  - **Ahanta** (`ahanta` / `aha`)
  - **Sehwi** (`sehwi` / `sfw`)
  - **Dangme** (`dangme` / `ada`)

### 🌍 Tier 2: Extended Ghanaian & Regional African Languages + Ghanaian English
- **Data Requirement:** **3 to 6 hours** (~1,500–3,000 clean clips).
- Languages requiring slightly more audio for acoustic and tonal anchoring:
  - **Ghanaian English** (`ghana_english` / `eng`)
  - **Kusaal** (`kusaal` / `kus`)
  - **Mampruli** (`mampruli` / `maw`)
  - **Gonja** (`gonja` / `gjn`)
  - **Kasem** (`kasem` / `xsm`)
  - **Buli** (`buli` / `bwu`)
  - **Konkomba** (`konkomba` / `xon`)
  - **Hausa** (`hausa` / `hau`)
  - **Bimoba** (`bimoba` / `bim`)
  - **Chumburung** (`chumburung` / `ncu`)
  - **Fulfulde** (`fulfulde` / `ffm`)
  - **Lelemi** (`lelemi` / `lef`)
  - **Nkonya** (`nkonya` / `nko`)
  - **Ntrubo** (`ntrubo` / `ntr`)
  - **Paasaal** (`paasaal` / `sig`)
  - **Sekpele** (`sekpele` / `lip`)
  - **Selee** (`selee` / `snw`)
  - **Sisaala** (`sisaala` / `sil`)
  - **Siwu** (`siwu` / `akp`)
  - **Tampulma** (`tampulma` / `tpm`)
  - **Tem** (`tem` / `kdh`)
  - **Tuwuli** (`tuwuli` / `bov`)
  - **Vagla** (`vagla` / `vag`)
  - **Deg** (`deg` / `mzw`)
  - **Avatime** (`avatime` / `avn`)
  - **Bassar** (`bassar` / `bud`)
  - **Birifor** (`birifor` / `biv`)
  - **Bissa** (`bissa` / `bib`)
  - **Gikyode** (`gikyode` / `acd`)
  - **Kabiye** (`kabiye` / `kbp`)
  - **Konni** (`konni` / `kma`)
  - **Nawuri** (`nawuri` / `naw`)
  - **Anyin** (`anyin` / `any`)

To inspect the full list from the command line:
```bash
ghana-voice-trainer list-langs
# Or inspect specific tier:
ghana-voice-trainer list-langs --tier 1
ghana-voice-trainer list-langs --tier 2
```

---

## Installation

```bash
git clone https://github.com/GhanaOpenAI/ghana-voice-trainer.git
cd ghana-voice-trainer
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart: All-In-One Pipeline

Train a model on your dataset from start to finish with a single command:

```bash
# Fine-tune Asante Twi (Tier 1: 2-3h audio):
ghana-voice-trainer run \
  --input-dir path/to/twi_recordings \
  --data-dir datasets/twi \
  --lang twi \
  --run-name twi-voice-model \
  --epochs 30 \
  --learning-rate 3e-5

# Fine-tune Ghanaian English (Tier 2: 3-6h audio):
ghana-voice-trainer run \
  --input-dir path/to/english_recordings \
  --data-dir datasets/ghana_english \
  --lang eng \
  --run-name ghana-english-model \
  --epochs 40 \
  --learning-rate 3e-5
```

---

## Step-by-Step Pipeline

### 1. Prepare Dataset (`prep`)
Filters out transcripts containing digits, strips dead silence with an 80ms onset/release buffer, and converts text to word-separated IPA:

```bash
ghana-voice-trainer prep \
  --input-dir path/to/recordings \
  --out-dir datasets/my_data \
  --lang twi \
  --min-dur 3.0 \
  --max-dur 15.0
```

### 2. NeuCodec Encoding (`encode`)
Encodes trimmed audio into discrete acoustic tokens:

```bash
ghana-voice-trainer encode \
  --data-dir datasets/my_data \
  --batch-size 32
```

### 3. Training (`train`)
Fine-tune the foundation model with Full Parameter SFT or LoRA:

```bash
ghana-voice-trainer train \
  --base-model ghanaopenai/neu-tts-ghana-base \
  --data-dir datasets/my_data \
  --output-dir output \
  --run-name my-custom-model \
  --full-sft \
  --no-ref \
  --epochs 50 \
  --learning-rate 3e-5 \
  --batch-size 8 \
  --grad-accum 4 \
  --early-stopping-patience 8 \
  --eval-cer
```
*Note: Uses `JointPlateauEarlyStoppingCallback` to monitor both `eval_loss` and `eval_cer` simultaneously, stopping only when both metrics have plateaued.*

### 4. Synthesize Speech (`infer`)
Direct Zero-Reference text-to-speech synthesis:

```bash
ghana-voice-trainer infer \
  --model-path output/my-custom-model/final \
  --text "Akwaaba, yɛbɔ mpae sɛ Onyankopɔn bɛhyira wo." \
  --lang twi \
  --out-wav synthesized.wav
```

---

## License
Apache-2.0. Built by the GhanaOpenAI team.
