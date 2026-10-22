"""
Ghana Voice: Text-to-speech foundation models and fine-tuning toolkit for Ghanaian and African languages.
"""

from .engine import GhanaVoice
from .train import train
from .constants import (
    DEFAULT_BASE_MODEL,
    LANG_MODEL_PREFIX,
    TIER1_LANGUAGES,
    TIER2_LANGUAGES,
    ALL_SUPPORTED_LANGUAGES,
    get_language_info,
)

__version__ = "0.2.0"
__all__ = [
    "GhanaVoice",
    "train",
    "DEFAULT_BASE_MODEL",
    "LANG_MODEL_PREFIX",
    "TIER1_LANGUAGES",
    "TIER2_LANGUAGES",
    "ALL_SUPPORTED_LANGUAGES",
    "get_language_info",
]
