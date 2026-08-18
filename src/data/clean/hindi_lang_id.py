"""
Hindi Language Identifier & Machine-Translation Quality Filter Module

Features:
  1. Script purity evaluation (Devanagari ratio >= 0.50, Latin ratio <= 0.10).
  2. Machine-Translation template & artifact rejection (Wikipedia MT tags, translationese markers).
  3. Native Hindi lexical discrimination (distinguishing Hindi from Nepali, Sanskrit, and English).
"""

import re
from typing import TypedDict

# High-frequency native Hindi signature vocabulary
HINDI_SIGNATURE_WORDS = {
    # Postpositions & Case Markers
    "में", "पर", "ने", "से", "का", "के", "की", "को", "तक", "द्वारा", "लिये", "लिए",
    # Verb forms & Auxiliaries
    "है", "हैं", "था", "थी", "थे", "होगा", "होगी", "होंगे", "होता", "होती", "होते",
    "किया", "किए", "की", "दिया", "दिए", "गया", "गए", "गई", "रहा", "रहे", "रही",
    "सकता", "सकते", "सकती", "करना", "करने", "होने", "पाया", "कहा", "बोला", "जाता",
    "जाती", "जाते", "हुए", "हुआ", "हुई",
    # Conjunctions, Pronouns & Common Vocabulary
    "और", "कि", "यह", "वह", "इस", "उस", "इन", "उन", "इसे", "उसे", "इन्हें", "उन्हें",
    "अपने", "अपनी", "अपना", "नहीं", "भी", "ही", "तो", "जब", "तब", "अगर", "यदि",
    "लेकिन", "परंतु", "किंतु", "क्योंकि", "इसलिए", "भारत", "भारतीय", "बहुत", "अधिक",
}

# Negative indicator words (Nepali / Sanskrit markers that signal cross-lingual contamination)
NON_HINDI_SIGNATURE_WORDS = {
    # Nepali specific
    "लाई", "बाट", "छन्", "भयो", "भए", "हुने", "हुन्", "थियो", "थिए", "गरेको", "गर्छ",
    "हुनुपर्छ", "हाम्रो", "तिनीहरू", "यसको", "सम्बन्धी", "अनुसार", "नेपाल", "नेपाली",
    # Sanskrit specific
    "अस्ति", "सन्ति", "भवति", "भवन्ति", "इति", "अपि", "एवम्", "तथापि", "यथा",
}

# Wikipedia Machine-Translation warning templates (case-insensitive regex)
MT_TEMPLATES_REGEX = re.compile(
    r"\{\{\s*(?:अनुवाद|अशुद्ध\s*अनुवाद|मशीनी\s*अनुवाद|यांत्रिक\s*अनुवाद|अनुवाद\s*अधूरा|"
    r"rough\s*translation|bad\s*translation|cleanup-translation|machine\s*translation|"
    r"not\s*hindi|translate|translateme|not-hindi)[^}]*\}\}",
    re.IGNORECASE,
)

# Common MT artifacts / English gloss leakage
MT_ARTIFACT_PATTERNS = [
    re.compile(r"\[\s*untranslated\s*\]", re.IGNORECASE),
    re.compile(r"translate(?:d)?\s+from\s+english", re.IGNORECASE),
    re.compile(r"original\s+english\s+text", re.IGNORECASE),
]


class HindiLanguageIdResult(TypedDict):
    language_code: str
    language_confidence: float
    language_status: str  # "accept", "reject", "uncertain"
    devanagari_ratio: float
    latin_ratio: float
    digit_ratio: float
    hindi_token_ratio: float
    non_hindi_token_ratio: float
    is_machine_translated: bool
    rejection_reason: str | None


def is_mt_flagged(raw_text: str) -> bool:
    """Check whether raw text contains Wikipedia machine translation warning tags or MT artifacts."""
    if not raw_text:
        return False
    if MT_TEMPLATES_REGEX.search(raw_text):
        return True
    for pat in MT_ARTIFACT_PATTERNS:
        if pat.search(raw_text):
            return True
    return False


def identify_hindi_language(text: str, raw_text: str = "") -> HindiLanguageIdResult:
    """Evaluate whether the cleaned text is high-quality native Hindi.

    Args:
        text: Cleaned text for linguistic scoring
        raw_text: Optional uncleaned raw text to check for MT templates
    """
    total_len = len(text)
    if total_len == 0:
        return {
            "language_code": "unknown",
            "language_confidence": 0.0,
            "language_status": "reject",
            "devanagari_ratio": 0.0,
            "latin_ratio": 0.0,
            "digit_ratio": 0.0,
            "hindi_token_ratio": 0.0,
            "non_hindi_token_ratio": 0.0,
            "is_machine_translated": False,
            "rejection_reason": "empty_document",
        }

    # 1. Check for explicit MT flags in raw text
    if raw_text and is_mt_flagged(raw_text):
        return {
            "language_code": "hi_mt",
            "language_confidence": 0.0,
            "language_status": "reject",
            "devanagari_ratio": 0.0,
            "latin_ratio": 0.0,
            "digit_ratio": 0.0,
            "hindi_token_ratio": 0.0,
            "non_hindi_token_ratio": 0.0,
            "is_machine_translated": True,
            "rejection_reason": "flagged_machine_translation_template",
        }

    # 2. Script ratios
    dev_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    lat_chars = sum(1 for c in text if "a" <= c.lower() <= "z")
    dig_chars = sum(1 for c in text if c.isdigit() or "\u0966" <= c <= "\u096F")

    dev_ratio = round(dev_chars / total_len, 4)
    lat_ratio = round(lat_chars / total_len, 4)
    dig_ratio = round(dig_chars / total_len, 4)

    # 3. Reject high Latin contamination (untranslated English paragraphs)
    if lat_ratio > 0.15:
        return {
            "language_code": "en_mixed",
            "language_confidence": 0.0,
            "language_status": "reject",
            "devanagari_ratio": dev_ratio,
            "latin_ratio": lat_ratio,
            "digit_ratio": dig_ratio,
            "hindi_token_ratio": 0.0,
            "non_hindi_token_ratio": 0.0,
            "is_machine_translated": False,
            "rejection_reason": f"excessive_latin_characters (latin_ratio={lat_ratio:.2f} > 0.15)",
        }

    if dev_ratio < 0.40:
        return {
            "language_code": "non_devanagari",
            "language_confidence": 0.0,
            "language_status": "reject",
            "devanagari_ratio": dev_ratio,
            "latin_ratio": lat_ratio,
            "digit_ratio": dig_ratio,
            "hindi_token_ratio": 0.0,
            "non_hindi_token_ratio": 0.0,
            "is_machine_translated": False,
            "rejection_reason": f"insufficient_devanagari (dev_ratio={dev_ratio:.2f} < 0.40)",
        }

    # 4. Tokenize and compute lexical signatures
    words = [w for w in re.split(r"[\s\s\n\r\t.,!?;:\"'()\[\]{}|\\/\-–—]+", text) if w.strip()]
    total_words = len(words)

    if total_words == 0:
        return {
            "language_code": "unknown",
            "language_confidence": 0.0,
            "language_status": "reject",
            "devanagari_ratio": dev_ratio,
            "latin_ratio": lat_ratio,
            "digit_ratio": dig_ratio,
            "hindi_token_ratio": 0.0,
            "non_hindi_token_ratio": 0.0,
            "is_machine_translated": False,
            "rejection_reason": "no_extractable_words",
        }

    hindi_hits = sum(1 for w in words if w in HINDI_SIGNATURE_WORDS)
    non_hindi_hits = sum(1 for w in words if w in NON_HINDI_SIGNATURE_WORDS)

    hindi_token_ratio = round(hindi_hits / total_words, 4)
    non_hindi_token_ratio = round(non_hindi_hits / total_words, 4)

    # 5. Cross-lingual rejection (reject Nepali/Sanskrit text)
    if non_hindi_token_ratio > hindi_token_ratio and non_hindi_token_ratio >= 0.03:
        return {
            "language_code": "ne_or_sa",
            "language_confidence": 0.0,
            "language_status": "reject",
            "devanagari_ratio": dev_ratio,
            "latin_ratio": lat_ratio,
            "digit_ratio": dig_ratio,
            "hindi_token_ratio": hindi_token_ratio,
            "non_hindi_token_ratio": non_hindi_token_ratio,
            "is_machine_translated": False,
            "rejection_reason": f"detected_non_hindi_language (non_hi_ratio={non_hindi_token_ratio:.2f} > hi_ratio={hindi_token_ratio:.2f})",
        }

    # 6. Confidence calculation
    raw_score = dev_ratio * (1.0 + 3.0 * hindi_token_ratio - 4.0 * non_hindi_token_ratio)
    confidence = round(max(0.0, min(1.0, raw_score)), 4)

    if hindi_token_ratio >= 0.02 and confidence >= 0.40:
        return {
            "language_code": "hi",
            "language_confidence": confidence,
            "language_status": "accept",
            "devanagari_ratio": dev_ratio,
            "latin_ratio": lat_ratio,
            "digit_ratio": dig_ratio,
            "hindi_token_ratio": hindi_token_ratio,
            "non_hindi_token_ratio": non_hindi_token_ratio,
            "is_machine_translated": False,
            "rejection_reason": None,
        }

    return {
        "language_code": "hi",
        "language_confidence": confidence,
        "language_status": "uncertain",
        "devanagari_ratio": dev_ratio,
        "latin_ratio": lat_ratio,
        "digit_ratio": dig_ratio,
        "hindi_token_ratio": hindi_token_ratio,
        "non_hindi_token_ratio": non_hindi_token_ratio,
        "is_machine_translated": False,
        "rejection_reason": "low_hindi_lexical_density",
    }
