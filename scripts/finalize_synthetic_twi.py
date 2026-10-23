"""
Finalize synthetic single-speaker Asante Twi dataset:
Phonemizes and encodes the 4,177 single-speaker female WAVs with NeuCodec on cuda:0.
"""

import json, glob, re
from pathlib import Path
import soundfile as sf
import torch
from tqdm import tqdm
from africa_g2p import AfricaPipeline
from neucodec import NeuCodec
import pyarrow.parquet as pq

OUT_DIR = Path("/mnt/volume_d2wey28/projects/vieneu-ghana/dataset_synthetic/asante_twi")
WAVS_DIR = OUT_DIR / "wavs"
meta_out = OUT_DIR / "metadata_encoded.csv"

print("1. Loading AfricaPipeline (twi) and NeuCodec (cuda:0)...")
pipe = AfricaPipeline(lang="twi", output="ipa")
codec = NeuCodec.from_pretrained("neuphonic/neucodec").to("cuda:0").eval()

# Gather texts from harvest_cache/asante_twi
shards = sorted(glob.glob("/mnt/volume_d2wey28/data/harvest_cache/asante_twi/*.parquet"))
text_pool = []
DIGIT_RE = re.compile(r"\d")

for f in shards:
    t = pq.read_table(f)
    rows = t.to_pydict()
    for text, dur in zip(rows["text"], rows["duration"]):
        if not (2.5 <= float(dur) <= 14.0):
            continue
        text = (text or "").strip()
        if not text or DIGIT_RE.search(text):
            continue
        text_pool.append(text)

wav_files = sorted(glob.glob(str(WAVS_DIR / "*.wav")))
print(f"Found {len(wav_files)} WAV files and {len(text_pool)} matched texts.")

final_lines = []
batch_size = 32

for idx in tqdm(range(len(wav_files)), desc="NeuCodec Encoding"):
    wf = wav_files[idx]
    fname = Path(wf).name
    text = text_pool[idx] if idx < len(text_pool) else ""
    if not text:
        continue
    ipa_str = pipe.run(text, sep="").strip()
    if not ipa_str:
        continue
        
    wav, sr = sf.read(wf)
    wav_t = torch.tensor(wav, dtype=torch.float32, device="cuda:0").unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        codes = codec.encode_code(wav_t).squeeze().cpu().tolist()
        
    final_lines.append(f"{fname}|{ipa_str}|{json.dumps(codes)}\n")

with open(meta_out, "w", encoding="utf-8") as f:
    f.writelines(final_lines)

print(f"Successfully wrote {len(final_lines)} encoded clips to {meta_out}!")
