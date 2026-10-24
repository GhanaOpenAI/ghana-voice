"""
Voice Conversion integration for ghana-voice using africa-female-vc.
Allows users to optionally convert real speech training audio into any of 21 African female voices
(e.g., 'clear-high-slow' for native Twi female voice, 'warm-high' for Hausa, etc.).

Seed-VC pins its own torch/transformers versions, so it usually cannot share the
training environment. Set GHANA_VOICE_VC_PYTHON to the Python of an environment
with africa-female-vc installed and conversion runs there in a worker process.
"""

import io
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("ghana_voice.vc")

# Cached voice converter instance
_CONVERTER_INSTANCE = None
_WORKER = None
_VC_DIFFUSION_STEPS = 30


class _VCWorker:
    """Seed-VC in a separate Python (GHANA_VOICE_VC_PYTHON), model loaded once."""

    def __init__(self, python: str, voice_name: str):
        self.voice_name = voice_name
        self.tmp = tempfile.TemporaryDirectory(prefix="ghana_voice_vc_")
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vc_worker.py")
        self.proc = subprocess.Popen(
            [python, worker, voice_name, str(_VC_DIFFUSION_STEPS)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        status = self.proc.stdout.readline().strip()
        if status != "ready":
            self.close()
            raise RuntimeError(f"Voice conversion worker failed to start ({python}): {status or 'no output'}")

    def convert(self, wav: np.ndarray, sr: int) -> tuple:
        src = os.path.join(self.tmp.name, "in.wav")
        dst = os.path.join(self.tmp.name, "out.wav")
        sf.write(src, wav, sr)
        self.proc.stdin.write(f"{src}\t{dst}\n")
        self.proc.stdin.flush()
        status = self.proc.stdout.readline().strip()
        if status != "ok":
            raise RuntimeError(f"Voice conversion failed: {status or 'worker exited'}")
        return sf.read(dst)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)
        self.tmp.cleanup()


def get_available_vc_voices() -> Dict[str, dict]:
    """Return all available reference voices from africa-female-vc."""
    try:
        from africa_female_vc.voices import list_voices
        return list_voices()
    except ImportError:
        # Fallback dictionary of canonical voices
        return {
            "clear-high-slow": {"recorded_in": "Twi", "pitch_hz": 261, "description": "Native Twi female voice, clear timbre, lively intonation (Recommended)"},
            "bright-high": {"recorded_in": "Yoruba", "pitch_hz": 220, "description": "Bright, energetic African female voice"},
            "clear-mid-steady": {"recorded_in": "Swahili", "pitch_hz": 209, "description": "Calm, steady, articulate narration voice"},
            "warm-high": {"recorded_in": "Hausa", "pitch_hz": 217, "description": "Warm, resonant tone with gentle cadence"},
            "bright-mid-quick": {"recorded_in": "Zulu", "pitch_hz": 215, "description": "Lively, dynamic, expressive tone"},
            "warm-low-steady": {"recorded_in": "Sepedi", "pitch_hz": 187, "description": "Deep, smooth, warm storytelling timbre"},
        }


def convert_audio(
    audio_arr_or_path: Union[str, Path, np.ndarray],
    voice_name: str = "clear-high-slow",
    sr: int = 16000,
    target_sr: int = 16000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """
    Convert an audio file or numpy array into a target African female voice.

    Args:
        audio_arr_or_path: Path to input audio or 1D float32 numpy array.
        voice_name: Target voice persona (default: 'clear-high-slow').
        sr: Sample rate if input is a numpy array.
        target_sr: Target sample rate for the converted output.
        device: 'cuda' or 'cpu'.

    Returns:
        1D float32 numpy array of the voice-converted audio.
    """
    # Load audio
    if isinstance(audio_arr_or_path, (str, Path)):
        wav, in_sr = sf.read(str(audio_arr_or_path))
    else:
        wav = audio_arr_or_path
        in_sr = sr
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    vc_python = os.environ.get("GHANA_VOICE_VC_PYTHON")
    if vc_python:
        global _WORKER
        if _WORKER is None or _WORKER.voice_name != voice_name:
            if _WORKER is not None:
                _WORKER.close()
            _WORKER = _VCWorker(vc_python, voice_name)
        converted_wav, out_sr = _WORKER.convert(np.asarray(wav, dtype=np.float32), in_sr)
        return _resample(converted_wav, out_sr, target_sr)

    try:
        from africa_female_vc.engine import VoiceConverter
    except ImportError:
        raise ImportError(
            "africa-female-vc is required for voice conversion. "
            "Install via: pip install git+https://github.com/AfriSpeech/africa-female-vc.git "
            "or set GHANA_VOICE_VC_PYTHON to a Python that has it."
        )

    global _CONVERTER_INSTANCE
    if _CONVERTER_INSTANCE is None or getattr(_CONVERTER_INSTANCE, "voice_name", None) != voice_name:
        logger.info(f"[GhanaVoice VC] Initializing VoiceConverter for '{voice_name}' on {device}...")
        _CONVERTER_INSTANCE = VoiceConverter(
            voice=voice_name,
            diffusion_steps=_VC_DIFFUSION_STEPS,
            install_deps=False,
        )
        _CONVERTER_INSTANCE.load()
        _CONVERTER_INSTANCE.voice_name = voice_name

    # Run conversion
    converted_wav, out_sr = _CONVERTER_INSTANCE.convert_array(np.asarray(wav, dtype=np.float32), sampling_rate=in_sr)
    return _resample(converted_wav, out_sr, target_sr)


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav
