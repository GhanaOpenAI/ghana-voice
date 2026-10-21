"""
Evaluation module: Transcribe synthesized speech with sherpa-onnx OmniASR 300M
and compute Character Error Rate (CER).
"""

import argparse
import glob
import json
import os
import re
import time
import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio

from .constants import (
    DEFAULT_BASE_MODEL,
    NEUCODEC_REPO,
    SAMPLE_RATE_16K,
    SAMPLE_RATE_24K,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    TOKEN_TEXT_START,
    TOKEN_TEXT_END,
    TOKEN_SPEECH_START,
    TOKEN_SPEECH_END,
)
from .phonemize import phonemize_text, clean_reference_phones

_ASR_MODEL_DEFAULT = (
    "/mnt/volume_d2wey28/models/sherpa/"
    "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05/model.onnx"
)
_ASR_TOKENS_DEFAULT = (
    "/mnt/volume_d2wey28/models/sherpa/"
    "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05/tokens.txt"
)


def load_asr(model_path: str, tokens_path: str):
    import sherpa_onnx
    if not os.path.exists(model_path) or not os.path.exists(tokens_path):
        raise FileNotFoundError(f"ASR model or tokens not found at: {model_path}")
    return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=model_path,
        tokens=tokens_path,
        num_threads=2,
        decoding_method="greedy_search",
    )


def transcribe(recognizer, wav_16k_np: np.ndarray) -> str:
    st = recognizer.create_stream()
    st.accept_waveform(SAMPLE_RATE_16K, wav_16k_np.astype(np.float32))
    recognizer.decode_stream(st)
    return (st.result.text or "").strip()


def evaluate(checkpoint_dir: str, base_model: str, asr_model: str, asr_tokens: str,
             test_samples: list, out_dir: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neucodec import NeuCodec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = load_asr(asr_model, asr_tokens)
    codec = NeuCodec.from_pretrained(NEUCODEC_REPO).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir).eval()

    os.makedirs(out_dir, exist_ok=True)
    speech_end_id = tokenizer.convert_tokens_to_ids(TOKEN_SPEECH_END)

    cers = []
    print(f"\nEvaluating {checkpoint_dir} on {len(test_samples)} test sentences ...")

    for item in test_samples:
        lang = item["lang"]
        name = item.get("name", lang)
        text = item["text"]
        ref_phones = clean_reference_phones(item["ref_phones"])
        ref_codes = item["ref_codes"]

        target_phones = phonemize_text(text, lang)
        ref_codes_str = "".join(f"<|speech_{c}|>" for c in ref_codes)
        combined = ref_phones.strip() + " " + target_phones.strip()
        chat = f"{TOKEN_TEXT_START}{combined}{TOKEN_TEXT_END}{TOKEN_SPEECH_START}{ref_codes_str}"

        inp = torch.tensor([tokenizer.encode(chat)], device=device)
        with torch.no_grad():
            out = model.generate(
                inp,
                eos_token_id=speech_end_id,
                max_new_tokens=220,
                do_sample=True,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                top_k=DEFAULT_TOP_K,
                repetition_penalty=DEFAULT_REPETITION_PENALTY,
                use_cache=True,
            )

        gen_ids = out[0, inp.shape[1]:].cpu().tolist()
        gen_str = tokenizer.decode(gen_ids, add_special_tokens=False)
        sp_ids = [int(n) for n in re.findall(r"<\|speech_(\d+)\|>", gen_str)]

        with torch.no_grad():
            wav_24k = codec.decode_code(torch.tensor(sp_ids, device=device).unsqueeze(0).unsqueeze(0)).squeeze().cpu().float()
            wav_16k = torchaudio.functional.resample(wav_24k, SAMPLE_RATE_24K, SAMPLE_RATE_16K).numpy()

        audio_path = os.path.join(out_dir, f"{name.lower()}.wav")
        sf.write(audio_path, wav_24k.numpy(), SAMPLE_RATE_24K)

        hyp = transcribe(rec, wav_16k)
        ref_clean = re.sub(r"[^\w\s]", "", text).lower().strip()
        hyp_clean = re.sub(r"[^\w\s]", "", hyp).lower().strip()
        cer = jiwer.cer(ref_clean, hyp_clean) if hyp_clean else 1.0

        print(f"  [{name:10s}] CER: {cer:6.2%}")
        print(f"    REF: {text}")
        print(f"    HYP: {hyp}")
        cers.append(cer)

    mean_cer = sum(cers) / len(cers) if cers else 1.0
    print(f"\nMean CER: {mean_cer:.2%}")
    return mean_cer


def main():
    ap = argparse.ArgumentParser(description="Evaluate Neu-TTS checkpoint using sherpa-onnx OmniASR 300M")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--asr-model", default=_ASR_MODEL_DEFAULT)
    ap.add_argument("--asr-tokens", default=_ASR_TOKENS_DEFAULT)
    ap.add_argument("--out-dir", default="eval_samples")
    args = ap.parse_args()

    # Default smoke test item
    sample = [{
        "lang": "twi",
        "name": "Twi",
        "text": "Onyankopɔn yɛ ɔdɔ na ɔdɔ obiara wɔ wiase.",
        "ref_phones": "n a d a w i d t u m i y e e k e s e e",
        "ref_codes": [100] * 150,
    }]
    evaluate(args.checkpoint, args.base_model, args.asr_model, args.asr_tokens, sample, args.out_dir)


if __name__ == "__main__":
    main()
