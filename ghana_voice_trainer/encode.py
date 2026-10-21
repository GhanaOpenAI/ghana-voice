"""
GPU-batched NeuCodec encoding with progress tracking and automatic resume.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import torch
from tqdm import tqdm

from .constants import NEUCODEC_REPO, NEUCODEC_HZ, SAMPLE_RATE_16K, CODEBOOK_SIZE


def encode_dataset(data_dir: str, metadata_file: str = "metadata.csv",
                   output_file: str = "metadata_encoded.csv",
                   batch_size: int = 16, num_workers: int = 8, limit: int = 0):
    from neucodec import NeuCodec

    meta_in = os.path.join(data_dir, metadata_file)
    meta_out = os.path.join(data_dir, output_file)
    raw_dir = os.path.join(data_dir, "raw_audio")
    progress_path = meta_out + ".progress"

    if not os.path.exists(meta_in):
        raise FileNotFoundError(f"Input metadata not found: {meta_in}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NeuCodec ({NEUCODEC_REPO}) on {device} ...")
    codec = NeuCodec.from_pretrained(NEUCODEC_REPO).to(device).eval()

    with open(meta_in, "r", encoding="utf-8") as f:
        rows = [ln.strip() for ln in f if ln.strip()]

    if limit > 0:
        rows = rows[:limit]

    done = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            done = {ln.strip() for ln in f if ln.strip()}
        print(f"Resuming: {len(done)} samples already encoded.")

    work = []
    for ln in rows:
        parts = ln.split("|")
        if len(parts) < 2 or parts[0] in done:
            continue
        audio_path = os.path.join(raw_dir, parts[0])
        if os.path.exists(audio_path):
            work.append((parts[0], parts[1], audio_path))

    print(f"Total samples to encode: {len(work)} (already done: {len(done)})")
    if not work:
        print("Nothing to encode.")
        return

    def load_wav(item):
        fn, text, path = item
        w, _ = librosa.load(path, sr=SAMPLE_RATE_16K, mono=True)
        return fn, text, w

    out_mode = "a" if done else "w"
    out_f = open(meta_out, out_mode, encoding="utf-8")
    prog_f = open(progress_path, "a", encoding="utf-8")

    written = 0
    skipped = 0

    def process_batch(batch):
        nonlocal written, skipped
        if not batch:
            return
        names = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        wavs = [b[2] for b in batch]
        max_len = max(len(w) for w in wavs)

        arr = np.zeros((len(wavs), 1, max_len), dtype=np.float32)
        lengths = []
        for i, w in enumerate(wavs):
            arr[i, 0, :len(w)] = w
            lengths.append(len(w))

        t = torch.from_numpy(arr).to(device)
        with torch.no_grad():
            codes = codec.encode_code(t)
        codes = codes.squeeze(1).cpu().numpy()

        for i, fn in enumerate(names):
            n_samples = lengths[i]
            frame_ratio = codes.shape[1] / max_len
            n_frames = max(1, int(round(n_samples * frame_ratio)))
            c = codes[i][:n_frames].astype(int).tolist()
            if not c or not all(0 <= x < CODEBOOK_SIZE for x in c):
                skipped += 1
                continue
            out_f.write(f"{fn}|{texts[i]}|{json.dumps(c)}\n")
            prog_f.write(f"{fn}\n")
            written += 1

        out_f.flush()
        prog_f.flush()

    batch = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for item in tqdm(pool.map(load_wav, work, chunksize=32), total=len(work), desc="Encoding audio"):
            batch.append(item)
            if len(batch) >= batch_size:
                process_batch(batch)
                batch = []

    process_batch(batch)
    out_f.close()
    prog_f.close()

    if os.path.exists(progress_path):
        os.remove(progress_path)

    print(f"Encoding complete! Saved {written + len(done)} encoded rows -> {meta_out} (skipped {skipped})")


def main():
    ap = argparse.ArgumentParser(description="NeuCodec encode prepared dataset")
    ap.add_argument("--data-dir", required=True, help="Directory containing metadata.csv and raw_audio/")
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--output", default="metadata_encoded.csv")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    encode_dataset(
        data_dir=args.data_dir,
        metadata_file=args.metadata,
        output_file=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
