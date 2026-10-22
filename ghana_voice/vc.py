"""
Voice Conversion integration for ghana-voice using africa-female-vc.
Allows users to optionally convert real speech training audio into any of 21 African female voices
(e.g., 'clear-high-slow' for native Twi female voice, 'warm-high' for Hausa, etc.).
"""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("ghana_voice.vc")

# Cached voice converter instance
_CONVERTER_INSTANCE = None


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
    try:
        from africa_female_vc.engine import VoiceConverter
    except ImportError:
        raise ImportError(
            "africa-female-vc is required for voice conversion. "
            "Install via: pip install git+https://github.com/AfriSpeech/africa-female-vc.git"
        )

    global _CONVERTER_INSTANCE
    if _CONVERTER_INSTANCE is None or getattr(_CONVERTER_INSTANCE, "voice_name", None) != voice_name:
        logger.info(f"[GhanaVoice VC] Initializing VoiceConverter for '{voice_name}' on {device}...")
        _CONVERTER_INSTANCE = VoiceConverter(
            voice=voice_name,
            diffusion_steps=30,
            device=device,
            install_deps=False,
        )
        _CONVERTER_INSTANCE.load()
        _CONVERTER_INSTANCE.voice_name = voice_name

    # Load audio
    if isinstance(audio_arr_or_path, (str, Path)):
        wav, in_sr = sf.read(str(audio_arr_or_path))
    else:
        wav = audio_arr_or_path
        in_sr = sr

    # Run conversion
    converted_wav = _CONVERTER_INSTANCE.convert_array(wav, sr=in_sr, target_sr=target_sr)
    return converted_wav
