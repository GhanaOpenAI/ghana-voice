"""
Build 10-hour bilingual code-switched synthetic dataset:
- Both Twi and Ghanaian English are in the IDENTICAL female voice (clear-high-slow).
- Ghanaian English is the brisk conversational broadcast corpus (3.5 - 4.2 wps).
- 40% pure Twi
- 30% brisk English BEFORE Twi (direct concat, no added silence)
- 30% brisk English AFTER Twi (direct concat, no added silence)
"""

import os, io, re, json, glob, random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
from africa_g2p import AfricaPipeline
from neucodec import NeuCodec

OUT_DIR = Path("/mnt/volume_d2wey28/projects/vieneu-ghana/dataset_synthetic_codeswitch/asante_twi")
WAVS_DIR = OUT_DIR / "wavs"
WAVS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = 24000

print("1. Loading AfricaPipeline and NeuCodec (cuda:0)...")
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

# 1. Load pre-encoded synthetic Twi clips
print("2. Loading pre-encoded synthetic Twi clips...")
twi_pool = []
with open("/mnt/volume_d2wey28/projects/vieneu-ghana/dataset_synthetic/asante_twi/metadata_encoded.csv", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 3:
            twi_pool.append({
                "filename": parts[0],
                "ipa": parts[1].strip(),
                "codes": json.loads(parts[2]),
                "wav_path": str(Path("/mnt/volume_d2wey28/projects/vieneu-ghana/dataset_synthetic/asante_twi/wavs") / parts[0]),
            })

print(f"Loaded {len(twi_pool)} synthetic Twi clips.")

# 2. Extract and encode brisk conversational English clips in clear-high-slow
print("3. Harvesting brisk conversational English in clear-high-slow...")
eng_shards = sorted(glob.glob("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghanaopenai--ghana-speech-female-22khz/snapshots/d2292f00a4549d34c56fd511a97fd6885603fb75/ghana_english/*.parquet"))
eng_pool = []
DIGIT_RE = re.compile(r"\d")

for f in eng_shards:
    if len(eng_pool) >= 2000:
        break
    t = pq.read_table(f)
    rows = t.to_pydict()
    for txt, dur, audio_entry in zip(rows["text"], rows["duration"], rows["audio"]):
        if len(eng_pool) >= 2000:
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
                
            wav_t = torch.tensor(wav_trim, dtype=torch.float32, device="cuda:0").unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                codes = codec.encode_code(wav_t).squeeze().cpu().tolist()
                
            eng_pool.append({
                "text": clean_txt,
                "ipa": ipa_str,
                "wav": wav_trim,
                "codes": codes,
                "duration": trim_dur,
            })
            if len(eng_pool) % 250 == 0:
                print(f"  Harvested {len(eng_pool)} brisk English clips in clear-high-slow...")
        except Exception:
            continue

print(f"Harvested total: {len(eng_pool)} brisk English clips in clear-high-slow.")

# 3. Create the 40/30/30 blend
print("4. Creating 40/30/30 bilingual single-speaker blend...")
random.seed(42)
random.shuffle(twi_pool)
random.shuffle(eng_pool)

n_total = len(twi_pool)
n_pure = int(0.40 * n_total)
n_eng_before = int(0.30 * n_total)
n_eng_after = n_total - n_pure - n_eng_before

pure_subset = twi_pool[:n_pure]
eng_before_subset = twi_pool[n_pure:n_pure+n_eng_before]
eng_after_subset = twi_pool[n_pure+n_eng_before:]

final_lines = []
total_dur = 0.0
idx = 0

# A. 40% Pure Twi
for item in pure_subset:
    fn = f"synth_bilingual_{idx:05d}_pure.wav"
    fpath = WAVS_DIR / fn
    wav, sr = sf.read(item["wav_path"])
    sf.write(str(fpath), wav, SAMPLE_RATE)
    final_lines.append(f"{fn}|{item['ipa']}|{json.dumps(item['codes'])}\n")
    total_dur += len(wav) / SAMPLE_RATE
    idx += 1

# B. 30% English BEFORE Twi: [eng_wav, twi_wav]
eng_idx = 0
for item in eng_before_subset:
    eng_item = eng_pool[eng_idx % len(eng_pool)]
    eng_idx += 1
    twi_wav, _ = sf.read(item["wav_path"])
    
    merged_wav = np.concatenate([eng_item["wav"], twi_wav])
    merged_ipa = f"{eng_item['ipa']} {item['ipa']}"
    merged_codes = eng_item["codes"] + item["codes"]
    
    fn = f"synth_bilingual_{idx:05d}_eng_before.wav"
    fpath = WAVS_DIR / fn
    sf.write(str(fpath), merged_wav, SAMPLE_RATE)
    final_lines.append(f"{fn}|{merged_ipa}|{json.dumps(merged_codes)}\n")
    total_dur += len(merged_wav) / SAMPLE_RATE
    idx += 1

# C. 30% English AFTER Twi: [twi_wav, eng_wav]
for item in eng_after_subset:
    eng_item = eng_pool[eng_idx % len(eng_pool)]
    eng_idx += 1
    twi_wav, _ = sf.read(item["wav_path"])
    
    merged_wav = np.concatenate([twi_wav, eng_item["wav"]])
    merged_ipa = f"{item['ipa']} {eng_item['ipa']}"
    merged_codes = item["codes"] + eng_item["codes"]
    
    fn = f"synth_bilingual_{idx:05d}_eng_after.wav"
    fpath = WAVS_DIR / fn
    sf.write(str(fpath), merged_wav, SAMPLE_RATE)
    final_lines.append(f"{fn}|{merged_ipa}|{json.dumps(merged_codes)}\n")
    total_dur += len(merged_wav) / SAMPLE_RATE
    idx += 1

meta_out = OUT_DIR / "metadata_encoded.csv"
with open(meta_out, "w", encoding="utf-8") as f:
    f.writelines(final_lines)

print(f"Successfully created bilingual single-speaker dataset: {len(final_lines)} clips ({total_dur/3600:.2f} hours)")
print(f"Saved to: {meta_out}")
