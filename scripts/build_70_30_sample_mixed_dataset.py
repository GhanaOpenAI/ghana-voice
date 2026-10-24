"""
Build 10-hour 70/30 sample-balanced dataset for Asante Twi:
- 70% clean single-speaker Asante Twi (clear-high-slow, standalone sentences)
- 30% clean single-speaker Ghanaian English (clear-high-slow, standalone sentences)
- NO audio splicing, NO sentence-level concatenation. Every sample is an independent sentence.
"""

import io, os, re, json, glob, random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
from africa_g2p import AfricaPipeline
from neucodec import NeuCodec

OUT_DIR = Path("/mnt/volume_d2wey28/projects/vieneu-ghana/dataset_70_30_asante_twi")
WAVS_DIR = OUT_DIR / "wavs"
WAVS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = 24000
DIGIT_RE = re.compile(r"\d")

print("1. Loading AfricaPipeline (eng) and NeuCodec (cuda:0)...")
pipe_eng = AfricaPipeline(lang="eng", output="ipa")
codec = NeuCodec.from_pretrained("neuphonic/neucodec").to("cuda:0").eval()

def to_ipa_word_sep(text: str) -> str:
    tokens = re.findall(r"[\w']+|[.,!?;:]", text)
    res = []
    for tok in tokens:
        if tok in ".,!?;:":
            if res:
                res[-1] = res[-1] + tok
        else:
            p = pipe_eng.run(tok, sep="").strip()
            if p:
                res.append(p)
    return " ".join(res)

# 1. Load 2,800 clean Asante Twi clips (70%)
print("2. Loading 2,800 clean Asante Twi clips (70%)...")
twi_samples = []
with open("/mnt/volume_d2wey28/projects/vieneu-ghana/dataset_synthetic/asante_twi/metadata_encoded.csv", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 3:
            twi_samples.append({
                "filename": parts[0],
                "ipa": parts[1].strip(),
                "codes": json.loads(parts[2]),
                "wav_src": str(Path("/mnt/volume_d2wey28/projects/vieneu-ghana/dataset_synthetic/asante_twi/wavs") / parts[0]),
            })

random.seed(42)
random.shuffle(twi_samples)
twi_selected = twi_samples[:2800]
print(f"Selected {len(twi_selected)} clean Asante Twi clips.")

# 2. Extract and encode 1,200 clean Ghanaian English clips (30%)
print("3. Harvesting and encoding 1,200 clean Ghanaian English clips in clear-high-slow (30%)...")
eng_shards = sorted(glob.glob("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghanaopenai--ghana-speech-female-22khz/snapshots/d2292f00a4549d34c56fd511a97fd6885603fb75/ghana_english/*.parquet"))
eng_selected = []

for f in eng_shards:
    if len(eng_selected) >= 1200:
        break
    t = pq.read_table(f)
    rows = t.to_pydict()
    for txt, dur, audio_entry in zip(rows["text"], rows["duration"], rows["audio"]):
        if len(eng_selected) >= 1200:
            break
        dur = float(dur)
        if not (2.5 <= dur <= 15.0):
            continue
        clean_txt = (txt or "").strip()
        if not clean_txt or DIGIT_RE.search(clean_txt):
            continue
        words = clean_txt.split()
        if len(words) < 4 or len(words) > 75:
            continue
            
        audio_bytes = audio_entry["bytes"] if isinstance(audio_entry, dict) else audio_entry
        try:
            wav, sr = sf.read(io.BytesIO(audio_bytes))
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)
            if sr != SAMPLE_RATE:
                wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
            wav = np.clip(wav, -1.0, 1.0)
            
            wav_trim, _ = librosa.effects.trim(wav, top_db=30)
            trim_dur = len(wav_trim) / SAMPLE_RATE
            if not (2.0 <= trim_dur <= 15.0):
                continue
                
            ipa_str = to_ipa_word_sep(clean_txt)
            if not ipa_str:
                continue
                
            fn = f"synth_eng_{len(eng_selected):05d}.wav"
            fpath = WAVS_DIR / fn
            sf.write(str(fpath), wav_trim, SAMPLE_RATE)
            
            wav_t = torch.tensor(wav_trim, dtype=torch.float32, device="cuda:0").unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                codes = codec.encode_code(wav_t).squeeze().cpu().tolist()
                
            eng_selected.append({
                "filename": fn,
                "ipa": ipa_str,
                "codes": codes,
            })
            if len(eng_selected) % 200 == 0:
                print(f"  Harvested & encoded {len(eng_selected)} / 1200 English clips...")
        except Exception:
            continue

print(f"Total English clips encoded: {len(eng_selected)}")

# 3. Combine and copy Twi audio files
print("4. Copying Twi audio files and writing metadata...")
final_entries = []

for item in twi_selected:
    fn = item["filename"]
    fpath = WAVS_DIR / fn
    if not fpath.exists():
        import shutil
        shutil.copy(item["wav_src"], str(fpath))
    final_entries.append({
        "filename": fn,
        "ipa": item["ipa"],
        "codes": item["codes"],
    })

for item in eng_selected:
    final_entries.append({
        "filename": item["filename"],
        "ipa": item["ipa"],
        "codes": item["codes"],
    })

# Shuffle the combined dataset so English and Twi samples are interleaved
random.shuffle(final_entries)

meta_out = OUT_DIR / "metadata_encoded.csv"
with open(meta_out, "w", encoding="utf-8") as f:
    for item in final_entries:
        f.write(f"{item['filename']}|{item['ipa']}|{json.dumps(item['codes'])}\n")

n_twi = len(twi_selected)
n_eng = len(eng_selected)
total = len(final_entries)
print(f"Successfully generated 70/30 balanced dataset: {total} total clips ({n_twi} Twi = {n_twi/total:.1%}, {n_eng} English = {n_eng/total:.1%})")
print(f"Saved to: {meta_out}")
