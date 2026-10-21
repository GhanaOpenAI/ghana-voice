# Ghana Voice Trainer 🇬🇭🎙️

**Ghana Voice Trainer** is an end-to-end toolkit for training, finetuning, and evaluating Ghanaian and African text-to-speech models built on [`ghanaopenai/neu-tts-ghana-base`](https://huggingface.co/ghanaopenai/neu-tts-ghana-base) and [NeuCodec](https://huggingface.co/neuphonic/neucodec).

Whether you want to clone your own voice or train a dedicated single-language model (Twi, Ewe, Ga, Dagbani, Fante, etc.), this toolkit handles the entire pipeline:
1. **Phonemization** via [africa-g2p](https://github.com/AfriSpeech/africa-g2p) into a shared universal grapheme space.
2. **Audio Encoding** into discrete acoustic codes via NeuCodec (GPU-batched, auto-resuming).
3. **LoRA Finetuning** with dynamic per-batch sequence padding (3x faster), cosine decay scheduling, and early stopping.
4. **Adapter Merging** into a standalone model ready for deployment.
5. **Speech Synthesis & Intelligibility Evaluation** with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) OmniASR 300M (measuring Character Error Rate).

---

## Installation

```bash
git clone https://github.com/GhanaOpenAI/ghana-voice-trainer.git
cd ghana-voice-trainer
pip install -r requirements.txt
pip install -e .
```

---

## Foundation Model

All training builds upon the Ghanaian foundation model:
- **Base Model:** [`ghanaopenai/neu-tts-ghana-base`](https://huggingface.co/ghanaopenai/neu-tts-ghana-base) (0.3B parameters, trained across 42 Ghanaian languages).

---

## Quickstart: All-In-One Pipeline

Train a model on your custom dataset from start to finish with a single command:

```bash
# From a local folder containing audio files (.wav) and a metadata.csv (filename|text):
ghana-voice-trainer run \
  --input-dir path/to/my_recordings \
  --data-dir datasets/my_voice \
  --lang twi \
  --run-name my-voice-model \
  --epochs 20 \
  --learning-rate 5e-5
```

Or from a Hugging Face dataset:
```bash
ghana-voice-trainer run \
  --hf-repo ghanaopenai/ghana-speech-female-22khz \
  --subset twi_twi \
  --data-dir datasets/twi_female \
  --lang twi \
  --run-name twi-female \
  --epochs 20 \
  --learning-rate 5e-5
```

---

## Step-by-Step Usage

### 1. Prepare Dataset (`prep`)
Convert audio files and phonemize text into universal graphemes:

```bash
ghana-voice-trainer prep \
  --input-dir path/to/recordings \
  --out-dir datasets/my_data \
  --lang twi \
  --min-dur 2.0 \
  --max-dur 15.0
```

### 2. NeuCodec Encoding (`encode`)
Encode all audio files into discrete NeuCodec codes:

```bash
ghana-voice-trainer encode \
  --data-dir datasets/my_data \
  --batch-size 16
```
*(Encoding automatically creates a progress checkpoint and resumes seamlessly if interrupted).*

### 3. LoRA Training (`train`)
Finetune the base model on your encoded dataset:

```bash
ghana-voice-trainer train \
  --base-model ghanaopenai/neu-tts-ghana-base \
  --data-dir datasets/my_data \
  --output-dir output \
  --run-name my-custom-model \
  --epochs 20 \
  --learning-rate 5e-5 \
  --batch-size 8 \
  --grad-accum 4 \
  --early-stopping-patience 4
```

### 4. Merge Adapter (`merge`)
Combine LoRA weights with the base model into a standalone model:

```bash
ghana-voice-trainer merge \
  --base-model ghanaopenai/neu-tts-ghana-base \
  --adapter-path output/my-custom-model/final \
  --output-dir output/my-custom-model/merged
```

### 5. Synthesize Speech (`infer`)
Generate speech using in-context voice cloning:

```bash
ghana-voice-trainer infer \
  --model-path output/my-custom-model/merged \
  --text "Onyankopɔn yɛ ɔdɔ na ɔdɔ obiara wɔ wiase." \
  --lang twi \
  --ref-audio path/to/sample_reference.wav \
  --ref-text "Reference sentence transcript spoken in the reference audio" \
  --out-wav synthesized.wav
```

### 6. CER Evaluation (`eval`)
Measure phonetic accuracy against ground truth using sherpa-onnx OmniASR 300M CTC:

```bash
ghana-voice-trainer eval \
  --checkpoint output/my-custom-model/final \
  --base-model ghanaopenai/neu-tts-ghana-base
```

---

## Long-Running / Detached Execution

For long training runs on remote VMs or servers (like an NVIDIA H200), use the detached launcher to ensure jobs keep running after terminal exit:

```bash
bash scripts/launch_detached.sh my_train_job \
  ghana-voice-trainer train \
    --data-dir datasets/my_data \
    --output-dir output \
    --run-name my_run

# Follow logs anytime:
tail -f logs/my_train_job.log
```

---

## Supported Languages

`ghana-voice-trainer` supports all Ghanaian and West African languages through [africa-g2p](https://github.com/AfriSpeech/africa-g2p), including:
- **Akan:** Twi (`twi`, `ak`, `tw`), Fante (`fat`)
- **Ga-Dangme:** Ga (`gaa`, `ga`), Dangme (`ada`)
- **Gbe:** Ewe (`ewe`, `ee`)
- **Gur:** Dagbani (`dag`), Farefare / Frafra (`gur`), Dagaare (`dga`)
- **Others:** English (`eng`, `en`), Hausa (`hau`), Nzema (`nzi`), Sehwi (`sfw`)

---

## License
Apache-2.0. Built by the GhanaOpenAI team.
