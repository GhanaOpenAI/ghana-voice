"""
Build a fine-tuning dataset from africa-female-vc converted speech only.

Source: ghanaopenai/ghana-speech-female-22khz (all audio converted to the
`clear-high-slow` voice). Human recordings are never used, so the fine-tuned
model hears a single consistent voice.

The language is filled up to `hours`. Optionally, VC-converted Ghanaian
English is added as separate whole clips (never spliced onto language clips)
up to `english_ratio` x the language hours. Filtering, silence trimming, IPA and
NeuCodec encoding follow the recipe that produced the 10h LoRA samples.
"""

import argparse
import glob
import io
import json
import os
import re
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from .constants import ALL_SUPPORTED_LANGUAGES, NEUCODEC_REPO
from .phonemize import ipa_from_pipe as to_ipa

VC_DATASET_REPO = "ghanaopenai/ghana-speech-female-22khz"
VC_DATASET_VOICE = "clear-high-slow"
ENGLISH_LANG = "ghana_english"

# Subset folder names used in the VC dataset (older local caches use the first name).
VC_SUBSETS = {
    "asante_twi": ["asante_twi", "twi_twi"],
    "akuapem_twi": ["akuapem_twi", "twi_twi"],
    "fante": ["fante", "fante_fat"],
    "ga": ["ga", "ga_gaa"],
    "ewe": ["ewe", "ewe_ewe"],
    "dagaare": ["dagaare", "dagaare_dga"],
    "frafra": ["frafra", "frafra_gur"],
    "ahanta": ["ahanta", "ahanta_aha"],
    "nzema": ["nzema", "nzema_nzi"],
    "sehwi": ["sehwi", "sehwi_sfw"],
    "dangme": ["dangme", "dangme_ada"],
    "ikposo": ["ikposo"],
    ENGLISH_LANG: [ENGLISH_LANG],
}

DIGIT_RE = re.compile(r"\d")
SR = 16000


def trim_silence_safe(wav: np.ndarray, sr: int = SR, top_db: int = 30, pad_ms: int = 80) -> np.ndarray:
    pad = int((pad_ms / 1000.0) * sr)
    _, idx = librosa.effects.trim(wav, top_db=top_db, frame_length=512, hop_length=128)
    s, e = idx
    if e - s < int(0.5 * sr):
        return wav
    return wav[max(0, s - pad):min(len(wav), e + pad)]


def resolve_vc_shards(lang: str, vc_root: str = None) -> list:
    """Parquet shards for `lang` from a local VC cache, else downloaded from the Hub."""
    subsets = VC_SUBSETS.get(lang, [lang])
    if vc_root:
        for sub in subsets:
            shards = sorted(glob.glob(os.path.join(vc_root, sub, "*.parquet")))
            if shards:
                return shards
        raise FileNotFoundError(f"No VC shards for '{lang}' under {vc_root} (tried {subsets})")

    from huggingface_hub import list_repo_files, snapshot_download
    files = list_repo_files(VC_DATASET_REPO, repo_type="dataset")
    for sub in subsets:
        if any(f.startswith(f"{sub}/") and f.endswith(".parquet") for f in files):
            local = snapshot_download(VC_DATASET_REPO, repo_type="dataset", allow_patterns=[f"{sub}/*"])
            return sorted(glob.glob(os.path.join(local, sub, "*.parquet")))
    raise FileNotFoundError(f"No VC subset for '{lang}' in {VC_DATASET_REPO} (tried {subsets})")


def collect_clips(shards: list, pipe, is_english: bool, target_seconds: float, prefix: str):
    collected = []
    total_seconds = 0.0
    for pq_file in shards:
        if total_seconds >= target_seconds:
            break
        try:
            table = pq.read_table(pq_file)
        except Exception:
            continue

        texts = table["text"].to_pylist()
        durs = table["duration"].to_pylist() if "duration" in table.column_names else [None] * len(texts)
        audios = table["audio"].to_pylist()

        for text, dur, audio_obj in zip(texts, durs, audios):
            if total_seconds >= target_seconds:
                break
            if not text or DIGIT_RE.search(text):
                continue
            if dur is not None and not (3.0 <= dur <= 15.0):
                continue

            try:
                raw_bytes = audio_obj["bytes"] if isinstance(audio_obj, dict) else audio_obj
                wav, sr = sf.read(io.BytesIO(raw_bytes))
            except Exception:
                continue
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)

            wav_clean = trim_silence_safe(wav, sr=SR, top_db=30, pad_ms=80)
            clean_dur = len(wav_clean) / SR
            if not (2.5 <= clean_dur <= 16.0):
                continue

            try:
                ipa_text = to_ipa(pipe, text, is_english=is_english)
            except Exception:
                continue
            if not ipa_text:
                continue

            fn = f"{prefix}_{len(collected):05d}.wav"
            collected.append((fn, ipa_text, wav_clean))
            total_seconds += clean_dur
    return collected, total_seconds


def encode_clips(collected: list, codec, device: str, batch_size: int) -> list:
    lines = []
    for idx in range(0, len(collected), batch_size):
        batch = collected[idx: idx + batch_size]
        wavs = [b[2].astype(np.float32) for b in batch]
        max_len = max(len(w) for w in wavs)
        arr = np.zeros((len(wavs), 1, max_len), dtype=np.float32)
        lengths = []
        for i, w in enumerate(wavs):
            arr[i, 0, :len(w)] = w
            lengths.append(len(w))

        t = torch.from_numpy(arr).to(device)
        with torch.no_grad():
            codes = codec.encode_code(t).squeeze(1).cpu().numpy()

        for i, b in enumerate(batch):
            frame_ratio = codes.shape[1] / max_len
            n_frames = max(1, int(round(lengths[i] * frame_ratio)))
            c = codes[i][:n_frames].astype(int).tolist()
            if c:
                lines.append(f"{b[0]}|{b[1]}|{json.dumps(c)}\n")
    return lines


def build_vc_dataset(lang: str, out_dir: str, hours: float = 10.0, english_ratio: float = 0.0,
                     vc_root: str = None, batch_size: int = 32, device: str = None) -> str:
    """Build `<out_dir>/metadata_encoded.csv` for `lang` from VC-converted audio only."""
    from africa_g2p import AfricaPipeline
    from neucodec import NeuCodec

    if lang not in ALL_SUPPORTED_LANGUAGES:
        raise ValueError(f"Unknown language '{lang}'. See `ghana-voice list-langs`.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir)
    out_meta = out_path / "metadata_encoded.csv"
    if out_meta.exists():
        raise FileExistsError(f"{out_meta} already exists; remove it or choose another --out-dir.")

    print(f"[{lang}] VC-only dataset: {hours:.1f}h language + {english_ratio:.0%} English top-up")
    iso = ALL_SUPPORTED_LANGUAGES[lang]["iso"]
    is_eng = iso == "eng"
    pipe = AfricaPipeline(lang=iso, output="ipa")
    shards = resolve_vc_shards(lang, vc_root)
    print(f"  Scanning {len(shards)} VC shards for {lang} ...")
    collected, lang_seconds = collect_clips(shards, pipe, is_eng, hours * 3600.0, lang)
    print(f"  {lang}: {len(collected)} clips ({lang_seconds / 3600:.2f} h)")
    if not collected:
        raise RuntimeError(f"No usable VC clips found for {lang}.")

    eng_clips, eng_seconds = [], 0.0
    if english_ratio > 0 and not is_eng:
        eng_pipe = AfricaPipeline(lang="eng", output="ipa")
        eng_shards = resolve_vc_shards(ENGLISH_LANG, vc_root)
        print(f"  Scanning {len(eng_shards)} VC shards for English top-up ...")
        eng_clips, eng_seconds = collect_clips(eng_shards, eng_pipe, True,
                                               english_ratio * lang_seconds, "eng")
        print(f"  english: {len(eng_clips)} clips ({eng_seconds / 3600:.2f} h)")

    print(f"Encoding with NeuCodec on {device} ...")
    codec = NeuCodec.from_pretrained(NEUCODEC_REPO).to(device).eval()
    lines = encode_clips(collected + eng_clips, codec, device, batch_size)

    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_meta, "w", encoding="utf-8") as f:
        f.writelines(lines)
    total = lang_seconds + eng_seconds
    print(f"  Saved {len(lines)} samples ({total / 3600:.2f} h, English {eng_seconds / max(total, 1e-9):.0%}) -> {out_meta}")
    return str(out_meta)


def add_english_topup(data_dir: str, english_ratio: float, vc_voice: str, vc_root: str = None) -> float:
    """
    Append VC-converted English clips to a dataset prepared by `prep`
    (`<data_dir>/raw_audio` + `metadata.csv`), before encoding.

    English is taken from the VC dataset and converted to `vc_voice` when that
    differs from the dataset voice, so every clip shares one voice. Returns the
    English seconds added.
    """
    if english_ratio <= 0:
        return 0.0
    if not vc_voice:
        raise ValueError("English top-up needs --vc-voice so the English clips match the voice of your data.")

    from africa_g2p import AfricaPipeline

    raw_dir = os.path.join(data_dir, "raw_audio")
    meta_path = os.path.join(data_dir, "metadata.csv")
    with open(meta_path, "r", encoding="utf-8") as f:
        rows = [ln for ln in f if ln.strip()]
    if any(r.startswith("eng_") for r in rows):
        print("English top-up already present in metadata.csv, skipping.")
        return 0.0

    lang_seconds = 0.0
    for r in rows:
        try:
            lang_seconds += sf.info(os.path.join(raw_dir, r.split("|", 1)[0])).duration
        except Exception:
            continue

    print(f"English top-up: {english_ratio:.0%} of {lang_seconds / 3600:.2f} h in voice '{vc_voice}' ...")
    eng_pipe = AfricaPipeline(lang="eng", output="ipa")
    clips, eng_seconds = collect_clips(resolve_vc_shards(ENGLISH_LANG, vc_root), eng_pipe, True,
                                       english_ratio * lang_seconds, "eng")

    convert = None
    if vc_voice != VC_DATASET_VOICE:
        from .vc import convert_audio
        convert = convert_audio

    lines = []
    for fn, ipa_text, wav in clips:
        if convert is not None:
            wav = convert(wav.astype(np.float32), voice_name=vc_voice, sr=SR, target_sr=SR)
        sf.write(os.path.join(raw_dir, fn), wav, SR)
        lines.append(f"{fn}|{ipa_text}\n")

    with open(meta_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  Added {len(lines)} English clips ({eng_seconds / 3600:.2f} h) -> {meta_path}")
    return eng_seconds


def main():
    ap = argparse.ArgumentParser(description="Build a VC-only fine-tuning dataset with English top-up.")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--english-ratio", type=float, default=0.0,
                    help="Mix in VC English: English hours as a fraction of language hours (default 0 = language only)")
    ap.add_argument("--vc-root", default=None,
                    help="Local copy of ghana-speech-female-22khz (subset folders); default downloads from the Hub")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    build_vc_dataset(a.lang, a.out_dir, a.hours, a.english_ratio, a.vc_root, a.batch_size, a.device)


if __name__ == "__main__":
    main()
