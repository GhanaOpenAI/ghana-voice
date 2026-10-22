"""
Global constants, Tier hierarchies, and default parameters for ghana-voice-trainer.
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

# Special prompt tokens for in-context voice cloning & zero-ref TTS
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

# =========================================================================
# LANGUAGE TIERS: Foundation Architecture & Data Requirements
# =========================================================================

# Tier 1: Core Ghanaian Languages (Deeply pre-trained in base model).
# Recommended training data for custom voice or specialization: 2 to 3 hours (~1,000–1,500 clips).
TIER1_LANGUAGES = {
    "asante_twi": {"display": "Asante Twi", "iso": "twi", "family": "Akan", "rec_hours": "2-3 hours"},
    "akuapem_twi": {"display": "Akuapem Twi", "iso": "twi", "family": "Akan", "rec_hours": "2-3 hours"},
    "fante": {"display": "Fante", "iso": "fat", "family": "Akan", "rec_hours": "2-3 hours"},
    "ga": {"display": "Ga", "iso": "gaa", "family": "Ga-Dangme", "rec_hours": "2-3 hours"},
    "ewe": {"display": "Ewe", "iso": "ewe", "family": "Gbe", "rec_hours": "2-3 hours"},
    "dagbani": {"display": "Dagbani", "iso": "dag", "family": "Gur", "rec_hours": "2-3 hours"},
    "dagaare": {"display": "Southern Dagaare", "iso": "dga", "family": "Gur", "rec_hours": "2-3 hours"},
    "frafra": {"display": "Frafra (Farefare)", "iso": "gur", "family": "Gur", "rec_hours": "2-3 hours"},
    "nzema": {"display": "Nzema", "iso": "nzi", "family": "Kwa", "rec_hours": "2-3 hours"},
    "ahanta": {"display": "Ahanta", "iso": "aha", "family": "Kwa", "rec_hours": "2-3 hours"},
    "sehwi": {"display": "Sehwi", "iso": "sfw", "family": "Kwa", "rec_hours": "2-3 hours"},
    "dangme": {"display": "Dangme", "iso": "ada", "family": "Ga-Dangme", "rec_hours": "2-3 hours"},
}

# Tier 2: Extended Ghanaian & Regional West African Languages + Ghanaian English.
# Recommended training data for custom voice or specialization: 3 to 6 hours (~1,500–3,000 clips).
TIER2_LANGUAGES = {
    "ghana_english": {"display": "Ghanaian English", "iso": "eng", "family": "English", "rec_hours": "3-6 hours"},
    "kusaal": {"display": "Kusaal", "iso": "kus", "family": "Gur", "rec_hours": "3-6 hours"},
    "mampruli": {"display": "Mampruli", "iso": "maw", "family": "Gur", "rec_hours": "3-6 hours"},
    "gonja": {"display": "Gonja", "iso": "gjn", "family": "Kwa", "rec_hours": "3-6 hours"},
    "kasem": {"display": "Kasem", "iso": "xsm", "family": "Gur", "rec_hours": "3-6 hours"},
    "buli": {"display": "Buli", "iso": "bwu", "family": "Gur", "rec_hours": "3-6 hours"},
    "konkomba": {"display": "Konkomba", "iso": "xon", "family": "Gur", "rec_hours": "3-6 hours"},
    "hausa": {"display": "Hausa", "iso": "hau", "family": "Chadic", "rec_hours": "3-6 hours"},
    "fulfulde": {"display": "Fulfulde", "iso": "ffm", "family": "Senegambian", "rec_hours": "3-6 hours"},
    "bimoba": {"display": "Bimoba", "iso": "bim", "family": "Gur", "rec_hours": "3-6 hours"},
    "chumburung": {"display": "Chumburung", "iso": "ncu", "family": "Kwa", "rec_hours": "3-6 hours"},
    "lelemi": {"display": "Lelemi", "iso": "lef", "family": "Kwa", "rec_hours": "3-6 hours"},
    "nkonya": {"display": "Nkonya", "iso": "nko", "family": "Kwa", "rec_hours": "3-6 hours"},
    "ntrubo": {"display": "Ntrubo", "iso": "ntr", "family": "Kwa", "rec_hours": "3-6 hours"},
    "paasaal": {"display": "Paasaal", "iso": "sig", "family": "Gur", "rec_hours": "3-6 hours"},
    "sekpele": {"display": "Sekpele", "iso": "lip", "family": "Kwa", "rec_hours": "3-6 hours"},
    "selee": {"display": "Selee", "iso": "snw", "family": "Kwa", "rec_hours": "3-6 hours"},
    "sisaala": {"display": "Sisaala", "iso": "sil", "family": "Gur", "rec_hours": "3-6 hours"},
    "siwu": {"display": "Siwu", "iso": "akp", "family": "Kwa", "rec_hours": "3-6 hours"},
    "tampulma": {"display": "Tampulma", "iso": "tpm", "family": "Gur", "rec_hours": "3-6 hours"},
    "tem": {"display": "Tem", "iso": "kdh", "family": "Gur", "rec_hours": "3-6 hours"},
    "tuwuli": {"display": "Tuwuli", "iso": "bov", "family": "Kwa", "rec_hours": "3-6 hours"},
    "vagla": {"display": "Vagla", "iso": "vag", "family": "Gur", "rec_hours": "3-6 hours"},
    "deg": {"display": "Deg", "iso": "mzw", "family": "Gur", "rec_hours": "3-6 hours"},
    "avatime": {"display": "Avatime", "iso": "avn", "family": "Kwa", "rec_hours": "3-6 hours"},
    "bassar": {"display": "Bassar (Ntcham)", "iso": "bud", "family": "Gur", "rec_hours": "3-6 hours"},
    "birifor": {"display": "Southern Birifor", "iso": "biv", "family": "Gur", "rec_hours": "3-6 hours"},
    "bissa": {"display": "Bissa", "iso": "bib", "family": "Mande", "rec_hours": "3-6 hours"},
    "gikyode": {"display": "Gikyode", "iso": "acd", "family": "Kwa", "rec_hours": "3-6 hours"},
    "kabiye": {"display": "Kabiye", "iso": "kbp", "family": "Gur", "rec_hours": "3-6 hours"},
    "konni": {"display": "Konni", "iso": "kma", "family": "Gur", "rec_hours": "3-6 hours"},
    "nawuri": {"display": "Nawuri", "iso": "naw", "family": "Kwa", "rec_hours": "3-6 hours"},
    "anyin": {"display": "Anyin", "iso": "any", "family": "Kwa", "rec_hours": "3-6 hours"},
}

ALL_SUPPORTED_LANGUAGES = {**TIER1_LANGUAGES, **TIER2_LANGUAGES}

# Common language code aliases mapped to canonical africa-g2p ISO-639-3 codes
LANG_ALIASES = {
    # Akan / Twi / Fante
    "tw": "twi",
    "ak": "twi",
    "aka": "twi",
    "akan": "twi",
    "asante": "twi",
    "asante_twi": "twi",
    "akuapem": "twi",
    "akuapem_twi": "twi",
    "fat": "fat",
    "fante": "fat",
    "fanti": "fat",
    # Ga-Dangme
    "ga": "gaa",
    "gaa": "gaa",
    "ada": "ada",
    "dangme": "ada",
    # Gbe
    "ee": "ewe",
    "ewe": "ewe",
    # Gur languages
    "dag": "dag",
    "dagbani": "dag",
    "dga": "dga",
    "dagaare": "dga",
    "gur": "gur",
    "frafra": "gur",
    "farefare": "gur",
    "kus": "kus",
    "kusaal": "kus",
    "maw": "maw",
    "mampruli": "maw",
    "xsm": "xsm",
    "kasem": "xsm",
    "bwu": "bwu",
    "buli": "bwu",
    "xon": "xon",
    "konkomba": "xon",
    "bim": "bim",
    "bimoba": "bim",
    "sil": "sil",
    "sisaala": "sil",
    "tpm": "tpm",
    "tampulma": "tpm",
    "kdh": "kdh",
    "tem": "kdh",
    "vag": "vag",
    "vagla": "vag",
    "mzw": "mzw",
    "deg": "mzw",
    "bud": "bud",
    "bassar": "bud",
    "biv": "biv",
    "birifor": "biv",
    "kbp": "kbp",
    "kabiye": "kbp",
    "kma": "kma",
    "konni": "kma",
    "sig": "sig",
    "paasaal": "sig",
    # Kwa languages
    "nzi": "nzi",
    "nzema": "nzi",
    "aha": "aha",
    "ahanta": "aha",
    "sfw": "sfw",
    "sehwi": "sfw",
    "gjn": "gjn",
    "gonja": "gjn",
    "ncu": "ncu",
    "chumburung": "ncu",
    "lef": "lef",
    "lelemi": "lef",
    "nko": "nko",
    "nkonya": "nko",
    "ntr": "ntr",
    "ntrubo": "ntr",
    "lip": "lip",
    "sekpele": "lip",
    "snw": "snw",
    "selee": "snw",
    "akp": "akp",
    "siwu": "akp",
    "bov": "bov",
    "tuwuli": "bov",
    "avn": "avn",
    "avatime": "avn",
    "acd": "acd",
    "gikyode": "acd",
    "naw": "naw",
    "nawuri": "naw",
    "any": "any",
    "anyin": "any",
    # Mande / Chadic / Senegambian / English
    "bib": "bib",
    "bissa": "bib",
    "hau": "hau",
    "hausa": "hau",
    "ffm": "ffm",
    "fulfulde": "ffm",
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "ghana_english": "eng",
}


def resolve_lang_code(lang: str) -> str:
    """Resolve user-supplied language name or alias to canonical africa-g2p code."""
    cleaned = lang.strip().lower()
    return LANG_ALIASES.get(cleaned, cleaned)


def get_language_info(lang: str) -> dict:
    """Return Tier, recommended hours, family, and display name for a language."""
    canonical = resolve_lang_code(lang)
    for k, v in TIER1_LANGUAGES.items():
        if k == lang.lower() or v["iso"] == canonical:
            return {"key": k, "tier": 1, **v}
    for k, v in TIER2_LANGUAGES.items():
        if k == lang.lower() or v["iso"] == canonical:
            return {"key": k, "tier": 2, **v}
    return {
        "key": lang.lower(),
        "tier": 2,
        "display": lang,
        "iso": canonical,
        "family": "African",
        "rec_hours": "3-6 hours",
    }
