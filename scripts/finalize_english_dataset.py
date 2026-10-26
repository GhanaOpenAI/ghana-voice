"""
Finalize and upload ghanaopenai/voice_trainer_english:
Encodes the 4,580 curated WAVs, generates metadata, and uploads to Hugging Face.
"""

import os, io, re, json, glob, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from africa_g2p import AfricaPipeline
from neucodec import NeuCodec
from huggingface_hub import HfApi

OUT_DIR = Path("/mnt/volume_d2wey28/data/voice_trainer_english")
WAVS_DIR = OUT_DIR / "wavs"
SAMPLE_RATE = 24000

print(f"Loading AfricaPipeline (eng) and NeuCodec (cuda:0)...")
pipe = AfricaPipeline(lang="eng", output="ipa")
codec = NeuCodec.from_pretrained("neuphonic/neucodec").to("cuda:0").eval()

def to_ipa_word_sep(text: str) -> str:
    tokens = re.findall(r"[\w']+|[.,!?;:]", text)
    res = []
    for tok in tokens:
        if tok in ".,!?;:":
            if res:
                res[-1] = res[-1] + tok
        else:
            p = pipe.run(tok, sep="").strip()
            if p:
                res.append(p)
    return " ".join(res)

# Read metadata or reconstruct from WAVs + sources
wav_files = sorted(glob.glob(str(WAVS_DIR / "*.wav")))
print(f"Found {len(wav_files)} WAV files in {WAVS_DIR}")

# 1. Harvest texts from sources matching audio files
sources = sorted(glob.glob("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghanaopenai--kafui-dey-english-tts/snapshots/*/data/*.parquet"))
if not sources:
    sources = sorted(glob.glob("/mnt/volume_d2wey28/data/ghana-speech/English_eng/*.parquet"))

def clean_english_text(t: str) -> str:
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\[.*?\]|\(.*?\)", "", t)
    if re.search(r"\d", t):
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if len(words) < 3 or len(words) > 20:
        return ""
    return t

print("Matching text for each saved clip...")
text_list = []
dur_list = []
for p in sources:
    if len(text_list) >= len(wav_files):
        break
    t = pq.read_table(p, columns=["text", "duration"])
    texts = t["text"].to_pylist()
    durs = t["duration"].to_pylist()
    for raw_text, dur in zip(texts, durs):
        if not (2.0 <= float(dur) <= 5.5):
            continue
        ct = clean_english_text(raw_text)
        if ct:
            text_list.append(ct)
            dur_list.append(float(dur))
            if len(text_list) >= len(wav_files):
                break

print(f"Matched {len(text_list)} texts for {len(wav_files)} WAVs.")

# 2. Encode with NeuCodec and generate IPA
encoded_data = []
batch_size = 32

for idx in tqdm(range(len(wav_files)), desc="Phonemizing and Encoding"):
    wf = wav_files[idx]
    fname = Path(wf).name
    text = text_list[idx] if idx < len(text_list) else ""
    if not text:
        continue
    ipa_str = to_ipa_word_sep(text)
    
    wav, sr = sf.read(wf)
    dur = len(wav) / sr
    wav_t = torch.tensor(wav, dtype=torch.float32, device="cuda:0").unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        codes = codec.encode_code(wav_t).squeeze().cpu().tolist()
        
    encoded_data.append({
        "filename": fname,
        "text": text,
        "ipa": ipa_str,
        "duration": round(dur, 2),
        "codes": codes,
    })

# 3. Write metadata files
print("Writing metadata files...")
meta_csv = OUT_DIR / "metadata.csv"
meta_enc_csv = OUT_DIR / "metadata_encoded.csv"

with open(meta_csv, "w", encoding="utf-8") as f:
    f.write("filename|text|ipa|duration\n")
    for r in encoded_data:
        f.write(f"{r['filename']}|{r['text']}|{r['ipa']}|{r['duration']}\n")

with open(meta_enc_csv, "w", encoding="utf-8") as f:
    for r in encoded_data:
        f.write(f"{r['filename']}|{r['ipa']}|{json.dumps(r['codes'])}\n")

total_dur = sum(r["duration"] for r in encoded_data)

p_table = pa.Table.from_pydict({
    "filename": [r["filename"] for r in encoded_data],
    "text": [r["text"] for r in encoded_data],
    "ipa": [r["ipa"] for r in encoded_data],
    "duration": [r["duration"] for r in encoded_data],
    "codes": [r["codes"] for r in encoded_data],
})
pq.write_table(p_table, str(OUT_DIR / "train.parquet"))

readme = f"""---
license: apache-2.0
tags:
- text-to-speech
- tts
- ghana-voice
- english
- speech
language:
- en
size_categories:
- 1K<n<10K
---

# Voice Trainer English (Ghanaian English Reference Dataset)

Curated dataset of **{len(encoded_data):,}** clean Ghanaian English audio clips ({total_dur/3600:.2f} hours) for code-switched multi-lingual TTS fine-tuning.

## Characteristics
- **Total Clips**: {len(encoded_data):,}
- **Total Duration**: {total_dur/3600:.2f} hours
- **Sample Rate**: 24,000 Hz float32 mono
- **Audio Codec**: NeuCodec discrete acoustic tokens (50 Hz, 16-bit codebook)
- **Phonetics**: Word-separated IPA (via `africa-g2p`)

Automatically integrated by `ghana-voice` for code-switching across all Ghanaian language TTS fine-tunes.
"""
with open(OUT_DIR / "README.md", "w") as f:
    f.write(readme)

print("Uploading dataset to ghanaopenai/voice_trainer_english on Hugging Face...")
api = HfApi()
api.upload_folder(
    folder_path=str(OUT_DIR),
    repo_id="ghanaopenai/voice_trainer_english",
    repo_type="dataset",
    commit_message=f"Upload clean voice_trainer_english dataset ({len(encoded_data)} clips, {total_dur/3600:.2f}h)"
)
print("Successfully uploaded to https://huggingface.co/datasets/ghanaopenai/voice_trainer_english")
