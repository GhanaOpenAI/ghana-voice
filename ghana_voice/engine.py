"""
GhanaVoice: High-Level Inference and Model Management for Ghanaian TTS.
Supports both CPU (GGUF + ONNX) and GPU (PyTorch safetensors + CUDA) inference.
"""

import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer
from africa_g2p import AfricaPipeline

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
    - Direct zero-reference synthesis from natural text or IPA.
    - CPU inference via GGUF (llama.cpp) + NeuCodec ONNX decoder.
    - GPU inference via PyTorch bfloat16 + CUDA NeuCodec.
    - Automatic model resolution: language-level model (ghanaopenai/ghana-voice_{lang})
      falling back to the multilingual base (ghanaopenai/ghana-voice-base).
    """

    def __init__(
        self,
        lang: str = "twi",
        model: Optional[str] = None,
        device: Optional[str] = None,
        use_gguf: Optional[bool] = None,
        gguf_type: str = "Q4_K_M",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.lang = lang
        self.lang_info = get_language_info(lang)
        self.iso = self.lang_info["iso"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.gguf_type = gguf_type

        # Decide whether to use GGUF for CPU execution
        if use_gguf is None:
            self.use_gguf = (self.device == "cpu")
        else:
            self.use_gguf = use_gguf

        self.model_id = self._resolve_model_id(model)
        print(f"[GhanaVoice] Language: {self.lang_info['display']} ({self.iso}) | Model: {self.model_id} | Device: {self.device}")

        # Always load the HF tokenizer for accurate token encoding
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.speech_end_id = self.tokenizer.convert_tokens_to_ids(TOKEN_SPEECH_END)

        # Initialize AfricaPipeline for word-separated IPA
        print(f"[GhanaVoice] Initializing {self.iso} AfricaPipeline (word-separated IPA)...")
        self.pipe = AfricaPipeline(lang=self.iso, output="ipa")

        # Load weights and audio decoder based on backend
        self.llm = None
        self.pytorch_model = None
        self.codec = None
        self.codec_onnx = None

        if self.use_gguf:
            self._init_gguf_backend()
        else:
            self._init_pytorch_backend()

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

    def _init_pytorch_backend(self):
        from transformers import AutoModelForCausalLM
        from neucodec import NeuCodec

        print(f"[GhanaVoice] Loading PyTorch model on {self.device} ...")
        self.pytorch_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        ).eval()

        print("[GhanaVoice] Initializing NeuCodec PyTorch decoder ...")
        self.codec = NeuCodec.from_pretrained(NEUCODEC_REPO).to(self.device).eval()

    def _init_gguf_backend(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            print("[GhanaVoice] llama-cpp-python not found, falling back to PyTorch CPU backend.")
            self.use_gguf = False
            self._init_pytorch_backend()
            return

        from huggingface_hub import hf_hub_download

        # Resolve GGUF filename
        model_name = Path(self.model_id).name
        gguf_filename = f"{model_name}-{self.gguf_type}.gguf"

        if os.path.exists(self.model_id) and self.model_id.endswith(".gguf"):
            gguf_path = self.model_id
        elif os.path.exists(os.path.join(self.model_id, gguf_filename)):
            gguf_path = os.path.join(self.model_id, gguf_filename)
        else:
            try:
                print(f"[GhanaVoice] Downloading {gguf_filename} from {self.model_id} ...")
                gguf_path = hf_hub_download(repo_id=self.model_id, filename=gguf_filename)
            except Exception:
                # Fallback to base model GGUF
                fallback_name = f"ghana-voice-base-{self.gguf_type}.gguf"
                print(f"[GhanaVoice] Falling back to {fallback_name} from {DEFAULT_BASE_MODEL} ...")
                gguf_path = hf_hub_download(repo_id=DEFAULT_BASE_MODEL, filename=fallback_name)

        print(f"[GhanaVoice] Loading GGUF weights on CPU: {gguf_path}")
        self.llm = Llama(model_path=gguf_path, n_ctx=2048, n_gpu_layers=0, verbose=False)

        # Load NeuCodec ONNX decoder for lightweight CPU synthesis if available
        try:
            from vieneu.utils import NeuCodecOnnx
            print("[GhanaVoice] Loading NeuCodec ONNX decoder on CPU ...")
            self.codec_onnx = NeuCodecOnnx.from_pretrained("neuphonic/neucodec-onnx-decoder-int8")
        except Exception:
            from neucodec import NeuCodec
            print("[GhanaVoice] Loading NeuCodec PyTorch CPU decoder ...")
            self.codec = NeuCodec.from_pretrained(NEUCODEC_REPO).to("cpu").eval()

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
        """Synthesize speech audio from text or pre-formatted IPA (24,000 Hz float32 array)."""
        ipa_text = text.strip() if is_ipa else self.to_ipa(text)
        prompt = f"{TOKEN_TEXT_START}{ipa_text}{TOKEN_TEXT_END}{TOKEN_SPEECH_START}"

        if self.use_gguf and self.llm is not None:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            out_ids = []
            for tid in self.llm.generate(
                prompt_ids,
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
            ):
                if tid == self.speech_end_id or tid == self.tokenizer.eos_token_id:
                    break
                out_ids.append(tid)
                if len(out_ids) >= max_new_tokens:
                    break
            gen_tokens = self.tokenizer.convert_ids_to_tokens(out_ids)
            speech_ids = [int(m.group(1)) for t in gen_tokens for m in [re.match(r"<\|speech_(\d+)\|>", t or "")] if m]
        else:
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            with torch.no_grad():
                out = self.pytorch_model.generate(
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

        # Decode speech tokens to 24 kHz waveform
        if self.codec_onnx is not None:
            codes_arr = np.array(speech_ids, dtype=np.int32)
            wav = np.asarray(self.codec_onnx.decode_code(codes_arr)).reshape(-1)
        else:
            sp_t = torch.tensor(speech_ids, device=self.device if self.codec else "cpu").unsqueeze(0).unsqueeze(0)
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
