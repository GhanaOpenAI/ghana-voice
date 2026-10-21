"""
Global constants and default parameters for ghana-voice-trainer.
"""

import string

# Canonical foundation model on Hugging Face
DEFAULT_BASE_MODEL = "ghanaopenai/neu-tts-ghana-base"

# NeuCodec parameters
NEUCODEC_REPO = "neuphonic/neucodec"
NEUCODEC_HZ = 50
CODEBOOK_SIZE = 65536
SAMPLE_RATE_16K = 16000
SAMPLE_RATE_24K = 24000

# Special prompt tokens for in-context voice cloning
TOKEN_TEXT_START = "<|TEXT_PROMPT_START|>"
TOKEN_TEXT_END = "<|TEXT_PROMPT_END|>"
TOKEN_SPEECH_START = "<|SPEECH_GENERATION_START|>"
TOKEN_SPEECH_END = "<|SPEECH_GENERATION_END|>"

# Maximum reference codes in prompt (~4 seconds of reference speech @ 50 Hz)
MAX_REF_CODES = 200

# Recommended generation parameters
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.2

# Punctuation character sets for cleaning reference prompts
PUNCT_SET = set(string.punctuation) | set("“”‘’–—…«»")

# Common language code aliases mapped to africa-g2p ISO-639-3 codes
LANG_ALIASES = {
    "tw": "twi",
    "ak": "twi",
    "akan": "twi",
    "asante": "twi",
    "akuapem": "twi",
    "fat": "fat",
    "fante": "fat",
    "ga": "gaa",
    "gaa": "gaa",
    "ada": "ada",
    "dangme": "ada",
    "ee": "ewe",
    "ewe": "ewe",
    "dag": "dag",
    "dagbani": "dag",
    "gur": "gur",
    "frafra": "gur",
    "farefare": "gur",
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "hau": "hau",
    "hausa": "hau",
    "nzi": "nzi",
    "nzema": "nzi",
    "sfw": "twi",       # Sehwi routed through Twi for Akan parity
    "sehwi": "twi",
    "dga": "dga",
    "dagaare": "dga",
}
