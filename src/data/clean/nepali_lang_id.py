"""
Nepali Language Identifier Module

Empirically distinguishes Nepali Devanagari prose from Hindi, Sanskrit, English,
and other languages using script metrics combined with Nepali vs Hindi lexical signatures.
"""

import re
from typing import TypedDict

# High-frequency Nepali signature tokens (words & bound morphemes)
NEPALI_SIGNATURE_WORDS = {
    # Postpositions & Case Markers
    "लाई", "मा", "ले", "बाट", "देखि", "द्वारा", "भन्दा", "सँग", "को", "का", "की",
    # Verb forms & Auxiliaries
    "छ", "छन्", "छौ", "छु", "थियो", "थिए", "थी", "भयो", "भए", "हुने", "हुन्",
    "गरिएको", "गर्ने", "गरिने", "भन्ने", "गर्छ", "छैन", "भने", "गरेको", "हुन्छ",
    "छैनन्", "हुनुपर्छ", "गर्न", "भनेर", "रहेको", "रहेका", "रहेकी", "भई", "भएर",
    # Conjunctions, Pronouns & Common Vocabulary
    "अनि", "र", "तर", "वा", "यस", "यसको", "यसमा", "यसले", "त्यस", "त्यसको", "त्यसमा",
    "हामी", "हाम्रो", "तपाईं", "तिनीहरू", "सबै", "अन्य", "कुनै", "अनुसार", "नेपाल",
    "नेपाली", "सम्बन्धी", "अघि", "पछि", "पहिलो", "दोस्रो", "बारेमा", "बीच", "लागि",
}

# High-frequency Hindi signature tokens (negative indicators for Nepali)
HINDI_SIGNATURE_WORDS = {
    "है", "हैं", "था", "थे", "थी", "और", "कि", "ने", "से", "में", "पर", "यह", "वह",
    "इस", "उस", "किया", "दिया", "गया", "नहीं", "होता", "सकता", "बहुत", "अपने",
    "साथ", "तक", "तरह", "गए", "जाता", "रहा", "रहे", "रही", "होने", "करने", "होता",
    "होती", "भारत", "भारतीय", "कहना", "कहा", "जब", "तब", "अगर", "लेकिन", "इसलिए",
}


class LanguageIdResult(TypedDict):
    language_code: str
    language_confidence: float
    language_method: str
    language_status: str  # "accept", "reject", "uncertain"
    devanagari_ratio: float
    latin_ratio: float
    digit_ratio: float
    nepali_token_ratio: float
    hindi_token_ratio: float
    rejection_reason: str | None


def identify_nepali_language(text: str) -> LanguageIdResult:
    """Identify language of input text and evaluate Nepali composite score.

    Returns LanguageIdResult dictionary.
    """
    total_len = len(text)
    if total_len == 0:
        return {
            "language_code": "unknown",
            "language_confidence": 0.0,
            "language_method": "composite_lexical_v1",
            "language_status": "reject",
            "devanagari_ratio": 0.0,
            "latin_ratio": 0.0,
            "digit_ratio": 0.0,
            "nepali_token_ratio": 0.0,
            "hindi_token_ratio": 0.0,
            "rejection_reason": "empty_document",
        }

    # 1. Calculate script ratios
    dev_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    lat_chars = sum(1 for c in text if "a" <= c.lower() <= "z")
    dig_chars = sum(1 for c in text if c.isdigit() or "\u0966" <= c <= "\u096F")

    dev_ratio = round(dev_chars / total_len, 4)
    lat_ratio = round(lat_chars / total_len, 4)
    dig_ratio = round(dig_chars / total_len, 4)

    # 2. Tokenize words for lexical evaluation
    words = [w for w in re.split(r"[\s\s\n\r\t.,!?;:\"'()\[\]{}|\\/\-–—]+", text) if w.strip()]
    total_words = len(words)

    if total_words == 0:
        return {
            "language_code": "unknown",
            "language_confidence": 0.0,
            "language_method": "composite_lexical_v1",
            "language_status": "reject",
            "devanagari_ratio": dev_ratio,
            "latin_ratio": lat_ratio,
            "digit_ratio": dig_ratio,
            "nepali_token_ratio": 0.0,
            "hindi_token_ratio": 0.0,
            "rejection_reason": "no_extractable_words",
        }

    # Count signature tokens
    nepali_hits = sum(1 for w in words if w in NEPALI_SIGNATURE_WORDS)
    hindi_hits = sum(1 for w in words if w in HINDI_SIGNATURE_WORDS)

    nepali_token_ratio = round(nepali_hits / total_words, 4)
    hindi_token_ratio = round(hindi_hits / total_words, 4)

    # Calculate confidence score
    raw_score = dev_ratio * (1.0 + 3.0 * nepali_token_ratio - 4.0 * hindi_token_ratio)
    confidence = round(max(0.0, min(1.0, raw_score)), 4)

    # 3. Decision Tree Logic
    if dev_ratio < 0.35:
        if lat_ratio > 0.40:
            lang_code = "en"
        else:
            lang_code = "non_devanagari"
        return {
            "language_code": lang_code,
            "language_confidence": confidence,
            "language_method": "composite_lexical_v1",
            "language_status": "reject",
            "devanagari_ratio": dev_ratio,
            "latin_ratio": lat_ratio,
            "digit_ratio": dig_ratio,
            "nepali_token_ratio": nepali_token_ratio,
            "hindi_token_ratio": hindi_token_ratio,
            "rejection_reason": f"insufficient_devanagari (ratio={dev_ratio:.2f} < 0.35)",
        }

    # Check for Hindi dominance
    if hindi_token_ratio > nepali_token_ratio and hindi_token_ratio >= 0.04:
        return {
            "language_code": "hi",
            "language_confidence": round(min(1.0, hindi_token_ratio * 10), 4),
            "language_method": "composite_lexical_v1",
            "language_status": "reject",
            "devanagari_ratio": dev_ratio,
            "latin_ratio": lat_ratio,
            "digit_ratio": dig_ratio,
            "nepali_token_ratio": nepali_token_ratio,
            "hindi_token_ratio": hindi_token_ratio,
            "rejection_reason": f"detected_hindi_language (hi_ratio={hindi_token_ratio:.2f} > ne_ratio={nepali_token_ratio:.2f})",
        }

    # Check for Nepali acceptance
    if nepali_token_ratio >= 0.03 and confidence >= 0.40:
        return {
            "language_code": "ne",
            "language_confidence": confidence,
            "language_method": "composite_lexical_v1",
            "language_status": "accept",
            "devanagari_ratio": dev_ratio,
            "latin_ratio": lat_ratio,
            "digit_ratio": dig_ratio,
            "nepali_token_ratio": nepali_token_ratio,
            "hindi_token_ratio": hindi_token_ratio,
            "rejection_reason": None,
        }

    # Borderline / Uncertain cases
    if nepali_token_ratio >= 0.01 or dev_ratio >= 0.60:
        return {
            "language_code": "ne",
            "language_confidence": confidence,
            "language_method": "composite_lexical_v1",
            "language_status": "uncertain",
            "devanagari_ratio": dev_ratio,
            "latin_ratio": lat_ratio,
            "digit_ratio": dig_ratio,
            "nepali_token_ratio": nepali_token_ratio,
            "hindi_token_ratio": hindi_token_ratio,
            "rejection_reason": f"low_nepali_lexical_density (ne_ratio={nepali_token_ratio:.3f})",
        }

    return {
        "language_code": "unknown",
        "language_confidence": confidence,
        "language_method": "composite_lexical_v1",
        "language_status": "reject",
        "devanagari_ratio": dev_ratio,
        "latin_ratio": lat_ratio,
        "digit_ratio": dig_ratio,
        "nepali_token_ratio": nepali_token_ratio,
        "hindi_token_ratio": hindi_token_ratio,
        "rejection_reason": "failed_nepali_language_identification",
    }
