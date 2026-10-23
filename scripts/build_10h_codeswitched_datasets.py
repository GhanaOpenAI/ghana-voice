#!/usr/bin/env python3
"""
Build 10-Hour Bilingual Code-Switched Datasets for Ghanaian Languages:
- 40% Pure Ghanaian Language speech.
- 30% English BEFORE Ghanaian Language: [eng_wav, lang_wav] (NO added silence, natural word space).
- 30% English AFTER Ghanaian Language: [lang_wav, eng_wav] (NO added silence, natural word space).

English data is automatically pulled from /mnt/volume_d2wey28/data/voice_trainer_english
(or downloaded from ghanaopenai/voice_trainer_english).
"""

import argparse
import glob
import io
import json
import logging
import os
import random
import re
from pathlib import Path
from tqdm import tqdm

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from africa_g2p import AfricaPipeline
from neucodec import NeuCodec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_codeswitched")

DIGIT_RE = re.compile(r"\d")
OUTPUT_BASE = Path("/mnt/volume_d2wey28/projects/vieneu-ghana/dataset_10h_codeswitch")
ENGLISH_DIR = Path("/mnt/volume_d2wey28/data/voice_trainer_english")
SAMPLE_RATE = 24000

REAL_LANGUAGE_SOURCES = [
    {
        "name": "asante_twi",
        "iso": "twi",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Asante_Twi_twi"],
    },
    {
        "name": "akuapem_twi",
        "iso": "twi",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Akuapem_Twi_twi"],
    },
    {
        "name": "fante",
        "iso": "fat",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Fante_fat"],
    },
    {
        "name": "ewe",
        "iso": "ewe",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Ewe_ewe"],
    },
    {
        "name": "dagbani",
        "iso": "dag",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Dagbani_dag"],
    },
    {
        "name": "dagaare",
        "iso": "dga",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Dagaare_dga"],
    },
    {
        "name": "frafra",
        "iso": "gur",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Ninkare_gur"],
    },
    {
        "name": "nzema",
        "iso": "nzi",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Nzema_nzi"],
    },
    {
        "name": "ahanta",
        "iso": "aha",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Ahanta_aha"],
    },
    {
        "name": "sehwi",
        "iso": "sfw",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Sehwi_sfw"],
    },
    {
        "name": "dangme",
        "iso": "ada",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Dangme_ada"],
    },
    {
        "name": "ga",
        "iso": "gaa",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Ga_gaa"],
    },
    {
        "name": "kusaal",
        "iso": "kus",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Kusaal_kus"],
    },
    {
        "name": "mampruli",
        "iso": "maw",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Mampruli_maw"],
    },
    {
        "name": "gonja",
        "iso": "gjn",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Gonja_gjn"],
    },
    {
        "name": "kasem",
        "iso": "xsm",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Kasem_xsm"],
    },
    {
        "name": "buli",
        "iso": "bwu",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Buli_bwu"],
    },
    {
        "name": "konkomba",
        "iso": "xon",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Konkomba_xon"],
    },
    {
        "name": "hausa",
        "iso": "hau",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Hausa_hau"],
    },
    {
        "name": "bimoba",
        "iso": "bim",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Bimoba_bim"],
    },
    {
        "name": "chumburung",
        "iso": "ncu",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Chumburung_ncu"],
    },
    {
        "name": "lelemi",
        "iso": "lef",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Lelemi_lef"],
    },
    {
        "name": "nkonya",
        "iso": "nko",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Nkonya_nko"],
    },
    {
        "name": "ntrubo",
        "iso": "ntr",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Ntrubo_ntr"],
    },
    {
        "name": "paasaal",
        "iso": "sig",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Paasaal_sig"],
    },
    {
        "name": "sekpele",
        "iso": "lip",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Sekpele_lip"],
    },
    {
        "name": "selee",
        "iso": "snw",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Selee_snw"],
    },
    {
        "name": "sisaala",
        "iso": "sil",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Sisaala_Tumulung_sil"],
    },
    {
        "name": "siwu",
        "iso": "akp",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Siwu_akp"],
    },
    {
        "name": "tampulma",
        "iso": "tpm",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Tampulma_tpm"],
    },
    {
        "name": "tem",
        "iso": "kdh",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Tem_kdh"],
    },
    {
        "name": "tuwuli",
        "iso": "bov",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Tuwuli_bov"],
    },
    {
        "name": "vagla",
        "iso": "vag",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Vagla_vag"],
    },
    {
        "name": "deg",
        "iso": "mzw",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Deg_mzw"],
    },
    {
        "name": "avatime",
        "iso": "avn",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Avatime_avn"],
    },
    {
        "name": "bassar",
        "iso": "bud",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Bassar_Ntcham_bud"],
    },
    {
        "name": "birifor",
        "iso": "biv",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Birifor_Southern_biv"],
    },
    {
        "name": "bissa",
        "iso": "bib",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Bissa_bib"],
    },
    {
        "name": "gikyode",
        "iso": "acd",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Gikyode_acd"],
    },
    {
        "name": "kabiye",
        "iso": "kbp",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Kabiye_kbp"],
    },
    {
        "name": "konni",
        "iso": "kma",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Konni_kma"],
    },
    {
        "name": "nawuri",
        "iso": "naw",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Nawuri_naw"],
    },
    {
        "name": "anyin",
        "iso": "any",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Anyin_any"],
    },
    {
        "name": "fulfulde",
        "iso": "ffm",
        "dirs": ["/mnt/volume_d2wey28/data/ghana-speech/Fulfulde_Maasina_ffm"],
    },
]


def load_english_pool():
    """Load pre-encoded English reference clips from voice_trainer_english."""
    print(f"Loading English reference data from {ENGLISH_DIR} ...")
    enc_meta = ENGLISH_DIR / "metadata_encoded.csv"
    if not enc_meta.exists():
        from huggingface_hub import snapshot_download
        print("Downloading voice_trainer_english from Hugging Face ...")
        snapshot_download(repo_id="ghanaopenai/voice_trainer_english", repo_type="dataset", local_dir=str(ENGLISH_DIR))

    pool = []
    with open(enc_meta, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                pool.append({
                    "filename": parts[0],
                    "ipa": parts[1].strip(),
                    "codes": json.loads(parts[2]),
                    "wav_path": str(ENGLISH_DIR / "wavs" / parts[0]),
                })
    print(f"Loaded {len(pool)} English reference clips.")
    return pool


def trim_silence_safe(wav: np.ndarray, sr: int = 24000, top_db: int = 30, pad_ms: int = 80) -> np.ndarray:
    pad = int((pad_ms / 1000.0) * sr)
    _, idx = librosa.effects.trim(wav, top_db=top_db, frame_length=512, hop_length=128)
    s = max(0, idx[0] - pad)
    e = min(len(wav), idx[1] + pad)
    if e - s < int(0.5 * sr):
        return wav
    return wav[s:e]


def build_language_dataset(item, english_pool, codec, args):
    lang_name = item["name"]
    iso = item["iso"]
    out_dir = OUTPUT_BASE / lang_name
    out_meta = out_dir / "metadata_encoded.csv"
    out_wavs = out_dir / "wavs"

    if out_meta.exists() and out_meta.stat().st_size > 50000:
        print(f"[{lang_name}] Already built ({out_meta.stat().st_size / 1e6:.1f} MB). Skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_wavs.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Building 10h Bilingual Code-Switched Dataset: {lang_name.upper()}")
    print(f"Ratio: 40% pure {lang_name}, 30% eng+lang, 30% lang+eng (direct concat)")
    print(f"=======================================================")

    pipe = AfricaPipeline(lang=iso, output="ipa")

    # 1. Harvest pure language clips
    lang_clips = []
    total_lang_dur = 0.0
    RAW_LANG_TARGET = 7.0 * 3600.0  # 7 hours raw yields 10.0h bilingual code-switched

    for d_str in item["dirs"]:
        if total_lang_dur >= RAW_LANG_TARGET:
            break
        d_path = Path(d_str)
        if not d_path.exists():
            continue
        parquets = sorted(glob.glob(f"{d_path}/*.parquet"))
        for pq_file in parquets:
            if total_lang_dur >= RAW_LANG_TARGET:
                break
            try:
                t = pq.read_table(pq_file)
            except Exception:
                continue

            texts = t["text"].to_pylist()
            durs = t["duration"].to_pylist() if "duration" in t.column_names else [None] * len(texts)
            audios = t["audio"].to_pylist()

            for text, dur, audio_obj in zip(texts, durs, audios):
                if total_lang_dur >= RAW_LANG_TARGET:
                    break
                if not text or DIGIT_RE.search(text):
                    continue
                if dur is not None and not (2.0 <= dur <= 10.0):
                    continue

                try:
                    raw_bytes = audio_obj["bytes"] if isinstance(audio_obj, dict) else audio_obj
                    wav, sr = sf.read(io.BytesIO(raw_bytes))
                except Exception:
                    continue

                if sr != SAMPLE_RATE:
                    wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)

                wav_clean = trim_silence_safe(wav, sr=SAMPLE_RATE, top_db=30, pad_ms=80)
                clean_dur = len(wav_clean) / SAMPLE_RATE
                if not (2.0 <= clean_dur <= 10.0):
                    continue

                try:
                    ipa_text = pipe.run(text.strip(), sep="").strip()
                except Exception:
                    continue
                if not ipa_text:
                    continue

                # Encode with NeuCodec
                wav_t = torch.tensor(wav_clean, dtype=torch.float32, device=args.device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    codes = codec.encode_code(wav_t).squeeze().cpu().tolist()

                lang_clips.append({
                    "text": text,
                    "ipa": ipa_text,
                    "codes": codes,
                    "wav": wav_clean,
                    "duration": clean_dur,
                })
                total_lang_dur += clean_dur
                if len(lang_clips) % 200 == 0:
                    print(f"  Harvested {len(lang_clips)} {lang_name} clips ({total_lang_dur/3600:.2f}h / 7.0h)")

    print(f"Total harvested {lang_name} clips: {len(lang_clips)} ({total_lang_dur/3600:.2f} hours)")

    # 2. Shuffle and partition into 40% pure, 30% eng_before, 30% eng_after
    random.seed(42)
    random.shuffle(lang_clips)
    random.shuffle(english_pool)

    n_total = len(lang_clips)
    n_pure = int(0.40 * n_total)
    n_eng_before = int(0.30 * n_total)
    n_eng_after = n_total - n_pure - n_eng_before

    pure_subset = lang_clips[:n_pure]
    eng_before_subset = lang_clips[n_pure:n_pure+n_eng_before]
    eng_after_subset = lang_clips[n_pure+n_eng_before:]

    final_records = []
    final_dur = 0.0
    idx = 0

    print(f"Synthesizing bilingual code-switched combinations (NO added silences)...")

    # A. 40% Pure Language
    for item in pure_subset:
        fn = f"{lang_name}_{idx:05d}_pure.wav"
        fpath = out_wavs / fn
        sf.write(str(fpath), item["wav"], SAMPLE_RATE)
        final_records.append(f"{fn}|{item['ipa']}|{json.dumps(item['codes'])}\n")
        final_dur += item["duration"]
        idx += 1

    # B. 30% English BEFORE Language: [eng_wav, lang_wav]
    eng_idx = 0
    for item in eng_before_subset:
        eng_item = english_pool[eng_idx % len(english_pool)]
        eng_idx += 1
        eng_wav, _ = sf.read(eng_item["wav_path"])
        
        merged_wav = np.concatenate([eng_wav, item["wav"]])
        merged_ipa = f"{eng_item['ipa']} {item['ipa']}"
        merged_codes = eng_item["codes"] + item["codes"]
        merged_dur = len(merged_wav) / SAMPLE_RATE

        fn = f"{lang_name}_{idx:05d}_eng_before.wav"
        fpath = out_wavs / fn
        sf.write(str(fpath), merged_wav, SAMPLE_RATE)
        final_records.append(f"{fn}|{merged_ipa}|{json.dumps(merged_codes)}\n")
        final_dur += merged_dur
        idx += 1

    # C. 30% English AFTER Language: [lang_wav, eng_wav]
    for item in eng_after_subset:
        eng_item = english_pool[eng_idx % len(english_pool)]
        eng_idx += 1
        eng_wav, _ = sf.read(eng_item["wav_path"])
        
        merged_wav = np.concatenate([item["wav"], eng_wav])
        merged_ipa = f"{item['ipa']} {eng_item['ipa']}"
        merged_codes = item["codes"] + eng_item["codes"]
        merged_dur = len(merged_wav) / SAMPLE_RATE

        fn = f"{lang_name}_{idx:05d}_eng_after.wav"
        fpath = out_wavs / fn
        sf.write(str(fpath), merged_wav, SAMPLE_RATE)
        final_records.append(f"{fn}|{merged_ipa}|{json.dumps(merged_codes)}\n")
        final_dur += merged_dur
        idx += 1

    # 3. Write final metadata
    with open(out_meta, "w", encoding="utf-8") as f:
        f.writelines(final_records)

    print(f"[{lang_name}] Dataset Complete: {len(final_records)} samples ({final_dur/3600:.2f} hours)")
    print(f"Saved to: {out_meta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lang", default=None)
    args = parser.parse_args()

    english_pool = load_english_pool()
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(args.device).eval()

    langs = REAL_LANGUAGE_SOURCES
    if args.lang:
        langs = [l for l in REAL_LANGUAGE_SOURCES if l["name"] == args.lang]

    for item in langs:
        build_language_dataset(item, english_pool, codec, args)


if __name__ == "__main__":
    main()
