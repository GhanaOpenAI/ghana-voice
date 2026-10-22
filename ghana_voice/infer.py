"""
Inference & speech synthesis using Neu-TTS and NeuCodec.
"""

import argparse
import os
import re
import librosa
import numpy as np
import soundfile as sf
import torch

from .constants import (
    DEFAULT_BASE_MODEL,
    NEUCODEC_REPO,
    SAMPLE_RATE_16K,
    SAMPLE_RATE_24K,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    MAX_REF_CODES,
    TOKEN_TEXT_START,
    TOKEN_TEXT_END,
    TOKEN_SPEECH_START,
    TOKEN_SPEECH_END,
)
from .phonemize import phonemize_text, clean_reference_phones


def synthesize(model_path: str, text: str, ref_audio_path: str, ref_text: str,
               out_wav_path: str, lang: str = "twi", adapter_path: str = "",
               temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P,
               top_k: int = DEFAULT_TOP_K, repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
               max_new_tokens: int = 400, device: str = ""):

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from neucodec import NeuCodec

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading NeuCodec on {device} ...")
    codec = NeuCodec.from_pretrained(NEUCODEC_REPO).to(device).eval()

    print(f"Encoding reference audio: {ref_audio_path} ...")
    wav, sr = librosa.load(ref_audio_path, sr=SAMPLE_RATE_16K, mono=True)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_codes = codec.encode_code(wav_t).squeeze().cpu().numpy().tolist()
    ref_codes = [int(c) for c in ref_codes][:MAX_REF_CODES]

    print(f"Loading TTS model: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    if adapter_path:
        print(f"Applying LoRA adapter: {adapter_path} ...")
        model = PeftModel.from_pretrained(base_model, adapter_path).eval()
    else:
        model = base_model.eval()

    # Phonemize
    ref_phones = clean_reference_phones(phonemize_text(ref_text, lang))
    target_phones = phonemize_text(text, lang)

    # In-context prompt
    ref_codes_str = "".join(f"<|speech_{c}|>" for c in ref_codes)
    combined = ref_phones.strip() + " " + target_phones.strip()
    chat = f"{TOKEN_TEXT_START}{combined}{TOKEN_TEXT_END}{TOKEN_SPEECH_START}{ref_codes_str}"

    input_ids = torch.tensor([tokenizer.encode(chat)], device=device)
    speech_end_id = tokenizer.convert_tokens_to_ids(TOKEN_SPEECH_END)

    print("Generating speech tokens ...")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            eos_token_id=speech_end_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_cache=True,
        )

    gen_ids = output[0, input_ids.shape[1]:].cpu().tolist()
    gen_str = tokenizer.decode(gen_ids, add_special_tokens=False)
    sp_ids = [int(n) for n in re.findall(r"<\|speech_(\d+)\|>", gen_str)]

    print(f"Generated {len(sp_ids)} acoustic speech tokens. Decoding audio ...")
    sp_tensor = torch.tensor(sp_ids, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out_wav = codec.decode_code(sp_tensor).squeeze().cpu().float().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)), exist_ok=True)
    sf.write(out_wav_path, out_wav, SAMPLE_RATE_24K)
    print(f"Synthesized audio saved successfully -> {out_wav_path}")


def main():
    ap = argparse.ArgumentParser(description="Neu-TTS voice cloning speech synthesis")
    ap.add_argument("--model-path", default=DEFAULT_BASE_MODEL, help="Path or HF ID of merged model")
    ap.add_argument("--adapter-path", default="", help="Optional path to LoRA adapter")
    ap.add_argument("--text", required=True, help="Text to speak")
    ap.add_argument("--lang", default="twi", help="Language code (e.g. twi, ewe, ga, eng)")
    ap.add_argument("--ref-audio", required=True, help="Path to reference audio file (.wav, .flac)")
    ap.add_argument("--ref-text", required=True, help="Exact transcript of reference audio")
    ap.add_argument("--out-wav", default="output.wav", help="Output .wav path")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--rep-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    args = ap.parse_args()

    synthesize(
        model_path=args.model_path,
        text=args.text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        out_wav_path=args.out_wav,
        lang=args.lang,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.rep_penalty,
    )


if __name__ == "__main__":
    main()
