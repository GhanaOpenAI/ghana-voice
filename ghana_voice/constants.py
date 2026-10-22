"""
Global constants, Tier hierarchies, and default parameters for ghana-voice.
"""

import string

# Canonical foundation model on Hugging Face
DEFAULT_BASE_MODEL = "ghanaopenai/ghana-voice-base"
LANG_MODEL_PREFIX = "ghanaopenai/ghana-voice_"

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
# LANGUAGE TIERS: Data Requirements & Fine-Tuning Recommendations
# =========================================================================
# For all language-level or custom voice fine-tuning from ghana-voice-base,
# we recommend a minimum of 5 to 10 hours of clean audio (~3,000–6,000 clips)
# trained with LoRA adaptation to prevent overfitting and retain base fluency.

REC_HOURS_TIER1 = "5-10 hours"
REC_HOURS_TIER2 = "5-10 hours"

# Tier 1: Core Ghanaian Languages (Deeply pre-trained in base model).
TIER1_LANGUAGES = {
    "asante_twi": {"display": "Asante Twi", "iso": "twi", "family": "Akan", "rec_hours": REC_HOURS_TIER1},
    "akuapem_twi": {"display": "Akuapem Twi", "iso": "twi", "family": "Akan", "rec_hours": REC_HOURS_TIER1},
    "fante": {"display": "Fante", "iso": "fat", "family": "Akan", "rec_hours": REC_HOURS_TIER1},
    "ga": {"display": "Ga", "iso": "gaa", "family": "Ga-Dangme", "rec_hours": REC_HOURS_TIER1},
    "ewe": {"display": "Ewe", "iso": "ewe", "family": "Gbe", "rec_hours": REC_HOURS_TIER1},
    "dagbani": {"display": "Dagbani", "iso": "dag", "family": "Gur", "rec_hours": REC_HOURS_TIER1},
    "dagaare": {"display": "Southern Dagaare", "iso": "dga", "family": "Gur", "rec_hours": REC_HOURS_TIER1},
    "frafra": {"display": "Frafra (Farefare)", "iso": "gur", "family": "Gur", "rec_hours": REC_HOURS_TIER1},
    "nzema": {"display": "Nzema", "iso": "nzi", "family": "Kwa", "rec_hours": REC_HOURS_TIER1},
    "ahanta": {"display": "Ahanta", "iso": "aha", "family": "Kwa", "rec_hours": REC_HOURS_TIER1},
    "sehwi": {"display": "Sehwi", "iso": "sfw", "family": "Kwa", "rec_hours": REC_HOURS_TIER1},
    "dangme": {"display": "Dangme", "iso": "ada", "family": "Ga-Dangme", "rec_hours": REC_HOURS_TIER1},
}

# Tier 2: Extended Ghanaian & Regional African Languages + Ghanaian English.
TIER2_LANGUAGES = {
    "ghana_english": {"display": "Ghanaian English", "iso": "eng", "family": "English", "rec_hours": REC_HOURS_TIER2},
    "kusaal": {"display": "Kusaal", "iso": "kus", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "mampruli": {"display": "Mampruli", "iso": "maw", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "gonja": {"display": "Gonja", "iso": "gjn", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "kasem": {"display": "Kasem", "iso": "xsm", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "buli": {"display": "Buli", "iso": "bwu", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "konkomba": {"display": "Konkomba", "iso": "xon", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "hausa": {"display": "Hausa", "iso": "hau", "family": "Chadic", "rec_hours": REC_HOURS_TIER2},
    "fulfulde": {"display": "Fulfulde", "iso": "ffm", "family": "Senegambian", "rec_hours": REC_HOURS_TIER2},
    "bimoba": {"display": "Bimoba", "iso": "bim", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "chumburung": {"display": "Chumburung", "iso": "ncu", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "lelemi": {"display": "Lelemi", "iso": "lef", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "nkonya": {"display": "Nkonya", "iso": "nko", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "ntrubo": {"display": "Ntrubo", "iso": "ntr", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "paasaal": {"display": "Paasaal", "iso": "sig", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "sekpele": {"display": "Sekpele", "iso": "lip", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "selee": {"display": "Selee", "iso": "snw", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "sisaala": {"display": "Sisaala", "iso": "sil", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "siwu": {"display": "Siwu", "iso": "akp", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "tampulma": {"display": "Tampulma", "iso": "tpm", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "tem": {"display": "Tem", "iso": "kdh", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "tuwuli": {"display": "Tuwuli", "iso": "bov", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "vagla": {"display": "Vagla", "iso": "vag", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "deg": {"display": "Deg", "iso": "mzw", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "avatime": {"display": "Avatime", "iso": "avn", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "bassar": {"display": "Bassar (Ntcham)", "iso": "bud", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "birifor": {"display": "Southern Birifor", "iso": "biv", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "bissa": {"display": "Bissa", "iso": "bib", "family": "Mande", "rec_hours": REC_HOURS_TIER2},
    "gikyode": {"display": "Gikyode", "iso": "acd", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "kabiye": {"display": "Kabiye", "iso": "kbp", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "konni": {"display": "Konni", "iso": "kma", "family": "Gur", "rec_hours": REC_HOURS_TIER2},
    "nawuri": {"display": "Nawuri", "iso": "naw", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
    "anyin": {"display": "Anyin", "iso": "any", "family": "Kwa", "rec_hours": REC_HOURS_TIER2},
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
        "rec_hours": "5-10 hours",
    }
