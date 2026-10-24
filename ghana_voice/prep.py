"""
Dataset preparation for ghana-voice.

Supports:
1. Local directory with audio files (.wav, .flac, .mp3) + metadata.csv (filename|text).
2. Hugging Face dataset (e.g. ghanaopenai/ghana-speech-female-22khz, split='train').
3. Optional Voice Conversion via africa-female-vc (e.g. --vc-voice clear-high-slow).
"""

import argparse
import collections
import hashlib
import io
import os
import re
import soundfile as sf
from typing import Optional
from tqdm import tqdm

from .constants import SAMPLE_RATE_16K
from .phonemize import phonemize_ipa, resolve_lang_code

_MAX_STEM = 160


def safe_name(sample_id: str) -> str:
    """Generate a collision-resistant, path-safe filename stem under 175 bytes."""
    raw = str(sample_id)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", raw).strip("_")
    if safe != raw or len(safe) > _MAX_STEM:
        safe = safe[:_MAX_STEM]
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"{safe}_{digest}"
    return safe


def prep_local_folder(
    input_dir: str,
    metadata_file: str,
    out_dir: str,
    lang: str,
    min_dur: float = 2.0,
    max_dur: float = 15.0,
    per_lang: int = 0,
    vc_voice: Optional[str] = None,
):
    raw_dir = os.path.join(out_dir, "raw_audio")
    os.makedirs(raw_dir, exist_ok=True)
    meta_out = os.path.join(out_dir, "metadata.csv")
    code = resolve_lang_code(lang)

    meta_in_path = os.path.join(input_dir, metadata_file)
    if not os.path.exists(meta_in_path):
        raise FileNotFoundError(f"Input metadata not found: {meta_in_path}")

    lines_to_write = []
    seen = set()
    kept = 0

    with open(meta_in_path, "r", encoding="utf-8") as f:
        rows = [ln.strip() for ln in f if ln.strip()]

    print(f"Preparing {len(rows)} local samples for {lang} ({code}) ...")
    if vc_voice:
        print(f"[VC Enabled] Converting audio into target voice: '{vc_voice}' ...")
        from .vc import convert_audio

    for line in tqdm(rows):
        if per_lang and kept >= per_lang:
            break
        parts = line.split("|") if "|" in line else line.split(",", 1)
        if len(parts) < 2:
            continue
        rel_audio, raw_text = parts[0].strip(), parts[1].strip()
        audio_src = os.path.join(input_dir, rel_audio)
        if not os.path.exists(audio_src):
            continue

        try:
            info = sf.info(audio_src)
            dur = info.duration
            if dur < min_dur or dur > max_dur:
                continue

            phones = phonemize_ipa(raw_text, code)
            if not phones.strip():
                continue

            stem = safe_name(os.path.splitext(os.path.basename(rel_audio))[0])
            out_filename = f"{code}_{stem}.wav"
            if out_filename in seen:
                continue
            seen.add(out_filename)

            out_audio = os.path.join(raw_dir, out_filename)
            if not os.path.exists(out_audio):
                arr, sr = sf.read(audio_src)
                if vc_voice:
                    arr = convert_audio(arr, voice_name=vc_voice, sr=sr, target_sr=16000)
                    sr = 16000
                sf.write(out_audio, arr, sr)

            lines_to_write.append(f"{out_filename}|{phones}\n")
            kept += 1
        except Exception:
            continue

    with open(meta_out, "w", encoding="utf-8") as f:
        f.writelines(lines_to_write)

    print(f"Dataset prepared: {kept} samples -> {meta_out}")


def prep_hf_dataset(
    hf_repo: str,
    subset: str,
    split: str,
    out_dir: str,
    lang: str,
    min_dur: float = 2.0,
    max_dur: float = 15.0,
    per_lang: int = 0,
    vc_voice: Optional[str] = None,
):
    from datasets import Audio, load_dataset

    raw_dir = os.path.join(out_dir, "raw_audio")
    os.makedirs(raw_dir, exist_ok=True)
    meta_out = os.path.join(out_dir, "metadata.csv")
    code = resolve_lang_code(lang)

    print(f"Loading HuggingFace dataset: {hf_repo} (subset: {subset}, split: {split}) ...")
    if vc_voice:
        print(f"[VC Enabled] Converting audio into target voice: '{vc_voice}' ...")
        from .vc import convert_audio

    ds = load_dataset(hf_repo, subset, split=split)
    ds = ds.cast_column("audio", Audio(decode=False))

    lines_to_write = []
    seen = set()
    kept = 0

    for sample in tqdm(ds, desc=f"Processing {subset}"):
        if per_lang and kept >= per_lang:
            break
        try:
            dur = sample.get("duration")
            if dur is None or dur < min_dur or dur > max_dur:
                continue
            text = (sample.get("text") or "").strip()
            if not text:
                continue

            phones = phonemize_ipa(text, code)
            if not phones.strip():
                continue

            stem = safe_name(sample.get("id", f"sample_{kept}"))
            filename = f"{code}_{stem}.wav"
            if filename in seen:
                continue
            seen.add(filename)

            out_audio = os.path.join(raw_dir, filename)
            if not os.path.exists(out_audio):
                arr, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]))
                if vc_voice:
                    arr = convert_audio(arr, voice_name=vc_voice, sr=sr, target_sr=16000)
                    sr = 16000
                sf.write(out_audio, arr, sr)

            lines_to_write.append(f"{filename}|{phones}\n")
            kept += 1
        except Exception:
            continue

    with open(meta_out, "w", encoding="utf-8") as f:
        f.writelines(lines_to_write)

    print(f"Prepared {kept} samples -> {meta_out}")
