"""
Phonemization module powered by africa-g2p.
"""

from africa_g2p import AfricaPipeline, convert_lang, UNIVERSAL
from .constants import LANG_ALIASES, PUNCT_SET


def resolve_lang_code(lang: str) -> str:
    """Normalize language name or code to its africa-g2p ISO-639-3 code."""
    normalized = lang.strip().lower()
    return LANG_ALIASES.get(normalized, normalized)


def phonemize_text(text: str, lang: str) -> str:
    """
    Phonemize text into the shared universal grapheme space.
    """
    code = resolve_lang_code(lang)
    try:
        return convert_lang(text.strip(), code, UNIVERSAL, sep=" ")
    except Exception as exc:
        # Fallback to direct pipeline run if convert_lang fails
        pipe = AfricaPipeline(lang=code, output="grapheme")
        return pipe.run(text.strip(), sep=" ")


def clean_reference_phones(phones: str) -> str:
    """
    Strip all punctuation marks from reference phonemes so reference speech prompts
    are clean and do not inject acoustic pause/punctuation artifacts into the prompt.
    """
    toks = [t.strip("".join(PUNCT_SET)) for t in phones.split()]
    return " ".join(t for t in toks if t and not any(ch in PUNCT_SET for ch in t))
