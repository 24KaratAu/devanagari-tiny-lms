"""Nepali counterpart of `clean_hindi_web`.

Only the boilerplate/teaser vocabulary differs between the two languages; URL
and entity stripping, NFC normalisation, script ratios and the quality gate are
language-neutral and are reused directly from `clean_hindi_web` rather than
duplicated, so a fix to one applies to both.
"""

from __future__ import annotations

from src.data.clean.clean_hindi_web import (  # noqa: F401
    clean_line, devanagari_ratio, latin_ratio, normalize_unicode, passes_quality,
)
from src.data.clean.clean_hindi_web import clean_web_text as _clean_web_text

# "Read also" teasers, which Nepali portals also glue to the end of a sentence.
TEASER_MARKERS = (
    "ALSO READ", "Also Read", "also read", "READ MORE", "Read More",
    "यो पनि पढ्नुहोस्", "यसलाई पनि पढ्नुहोस्", "यो पनि हेर्नुहोस्",
    "थप पढ्नुहोस्", "सम्बन्धित समाचार", "सम्बन्धित खबर",
    "अन्य समाचार", "पनि पढ्नुहोस्",
)

# Share prompts, credits, app nags and dateline furniture.
BOILERPLATE_MARKERS = (
    "सेयर गर्नुहोस्", "शेयर गर्नुहोस्", "प्रतिक्रिया दिनुहोस्",
    "प्रकाशित मिति", "अन्तिम अपडेट", "तस्बिर सौजन्य", "फोटो सौजन्य",
    "एजेन्सी", "एजेन्सीहरू", "हामीलाई फलो गर्नुहोस्", "फलो गर्नुहोस्",
    "डाउनलोड गर्नुहोस्", "सदस्यता", "विज्ञापन", "ताजा अपडेट",
)


def clean_web_text(text: str, min_line_chars: int = 15,
                   max_line_latin_ratio: float = 0.40) -> str:
    """Clean a Nepali web page body using the shared line-level pipeline."""
    return _clean_web_text(text, min_line_chars, max_line_latin_ratio,
                           teasers=TEASER_MARKERS,
                           boilerplate=BOILERPLATE_MARKERS)
