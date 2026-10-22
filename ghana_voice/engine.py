"""
GhanaVoice: High-Level Inference and Model Management for Ghanaian TTS.
"""

import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from africa_g2p import AfricaPipeline
from neucodec import NeuCodec

from .constants import (
    DEFAULT_BASE_MODEL,
    LANG_MODEL_PREFIX,
    NEUCODEC_REPO,
    SAMPLE_RATE_24K,
    TOKEN_TEXT_START,
    TOKEN_TEXT_END,
    TOKEN_SPEECH_START,
    TOKEN_SPEECH_END,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    resolve_lang_code,
    get_language_info,
)


class GhanaVoice:
    """
    High-level speech synthesis engine for Ghanaian and African languages.

    Supports:
    - Zero-Reference direct synthesis from natural text or IPA.
    - Automatic model resolution: language-level model (ghanaopenai/ghana-voice_{lang})
      falling back to the multilingual base (ghanaopenai/ghana-voice-base).
    - Audio export to 24 kHz high-fidelity WAV.
    """

    def __init__(
        self,
        lang: str = "twi",
        model: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.lang = lang
        self.lang_info = get_language_info(lang)
        self.iso = self.lang_info["iso"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        # Resolve model path
        self.model_id = self._resolve_model_id(model)
        print(f"[GhanaVoice] Loading {self.lang_info['display']} TTS from: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        ).eval()

        print("[GhanaVoice] Initializing NeuCodec 24kHz audio decoder...")
        self.codec = NeuCodec.from_pretrained(NEUCODEC_REPO).to(self.device).eval()

        print(f"[GhanaVoice] Initializing {self.iso} AfricaPipeline (word-separated IPA)...")
        self.pipe = AfricaPipeline(lang=self.iso, output="ipa")

        self.speech_end_id = self.tokenizer.convert_tokens_to_ids(TOKEN_SPEECH_END)

    def _resolve_model_id(self, model: Optional[str]) -> str:
        """Auto-resolve language-level model, local path, or base model."""
        if model:
            return model

        # Check for language-specific model repo: ghanaopenai/ghana-voice_{iso}
        lang_repo = f"{LANG_MODEL_PREFIX}{self.iso}"
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.repo_info(lang_repo)
            return lang_repo
        except Exception:
            # Fall back to base foundation model
            return DEFAULT_BASE_MODEL

    def to_ipa(self, text: str) -> str:
        """Convert natural text to word-separated IPA with preserved punctuation."""
        if self.iso in ("eng", "en"):
            tokens = re.findall(r"[\w']+|[.,!?;:]", text)
            res = []
            for tok in tokens:
                if tok in ".,!?;:":
                    if res:
                        res[-1] = res[-1] + tok
                else:
                    res.append(self.pipe.run(tok, sep="").strip())
            return " ".join(res)
        else:
            return self.pipe.run(text.strip(), sep="").strip()

    def synthesize(
        self,
        text: str,
        is_ipa: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = 500,
    ) -> np.ndarray:
        """
        Synthesize speech audio from text or pre-formatted IPA.
        Returns 24,000 Hz 1D float32 numpy array.
        """
        ipa_text = text.strip() if is_ipa else self.to_ipa(text)
        prompt = f"{TOKEN_TEXT_START}{ipa_text}{TOKEN_TEXT_END}{TOKEN_SPEECH_START}"
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                eos_token_id=self.speech_end_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                use_cache=True,
            )

        gen_ids = out[0, input_ids.shape[1]:].cpu().tolist()
        gen_str = self.tokenizer.decode(gen_ids, add_special_tokens=False)
        speech_ids = [int(n) for n in re.findall(r"<\|speech_(\d+)\|>", gen_str)]

        if not speech_ids:
            raise RuntimeError("Model generated no speech tokens. Check prompt length or input characters.")

        sp_t = torch.tensor(speech_ids, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            wav = self.codec.decode_code(sp_t).squeeze().cpu().float().numpy()

        return wav

    def speak(
        self,
        text: str,
        output_file: Optional[Union[str, Path]] = None,
        is_ipa: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Synthesize text and optionally write directly to a WAV file."""
        wav = self.synthesize(text, is_ipa=is_ipa, **kwargs)
        if output_file:
            sf.write(str(output_file), wav, SAMPLE_RATE_24K)
            print(f"[GhanaVoice] Saved audio ({len(wav)/SAMPLE_RATE_24K:.2f}s) -> {output_file}")
        return wav
