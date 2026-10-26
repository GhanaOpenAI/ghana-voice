"""
Phonemization module powered by africa-g2p.
"""

import re

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


_IPA_PIPES = {}


def ipa_from_pipe(pipe, text: str, is_english: bool = False) -> str:
    """Word-separated IPA as used for base-model training and zero-ref inference."""
    if is_english:
        tokens = re.findall(r"[\w']+|[.,!?;:]", text)
        res = []
        for tok in tokens:
            if tok in ".,!?;:":
                if res:
                    res[-1] = res[-1] + tok
            else:
                res.append(pipe.run(tok, sep="").strip())
        return " ".join(res)
    return pipe.run(text.strip(), sep="").strip()


def phonemize_ipa(text: str, lang: str) -> str:
    """Phonemize text into word-separated IPA (the format fine-tuning data must use)."""
    code = resolve_lang_code(lang)
    if code not in _IPA_PIPES:
        _IPA_PIPES[code] = AfricaPipeline(lang=code, output="ipa")
    return ipa_from_pipe(_IPA_PIPES[code], text, is_english=(code == "eng"))


def clean_reference_phones(phones: str) -> str:
    """
    Strip all punctuation marks from reference phonemes so reference speech prompts
    are clean and do not inject acoustic pause/punctuation artifacts into the prompt.
    """
    toks = [t.strip("".join(PUNCT_SET)) for t in phones.split()]
    return " ".join(t for t in toks if t and not any(ch in PUNCT_SET for ch in t))
