"""Cleaning helpers for manually crawled Hindi web pages.

Shared by the Phase-B crawler and any re-processing step so that the cleaning
applied to the remaining manual collection is identical to what is documented
in the Phase 1 report.
"""

from __future__ import annotations

import re
import unicodedata

DEVANAGARI_START, DEVANAGARI_END = "ऀ", "ॿ"

# Teaser markers. These frequently appear *inline*, glued to the end of a real
# sentence ("...साक्षी बनती है।ALSO READ: नवरात्रि..."), so we cut the line at the
# marker and keep the prefix rather than discarding the whole paragraph.
TEASER_MARKERS = (
    "ALSO READ", "Also Read", "also read", "READ MORE", "Read More",
    "READ ALSO", "ALSO WATCH", "WATCH VIDEO", "Watch Video",
    "यह भी पढ़ें", "ये भी पढ़ें", "इसे भी पढ़ें", "और पढ़ें",
    "पढ़ें पूरी खबर", "पूरी खबर पढ़ने के लिए", "यह भी देखें", "ये भी देखें",
)

# Furniture that Hindi news templates leave behind even after boilerplate
# extraction: share prompts, app install nags, bylines, credits.
BOILERPLATE_MARKERS = (
    "डाउनलोड करें", "ऐप डाउनलोड", "हमें फॉलो करें", "फॉलो करें",
    "शेयर करें", "व्हाट्सएप चैनल", "टेलीग्राम चैनल",
    "सब्सक्राइब", "विज्ञापन", "तस्वीर साभार", "फोटो साभार",
    "इनपुट", "एजेंसी इनपुट", "ताजा खबरें", "लाइव अपडेट",
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_CITATION_RE = re.compile(r"\[\d{1,3}\]")
_MULTISPACE_RE = re.compile(r"[ \t ]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if DEVANAGARI_START <= c <= DEVANAGARI_END) / len(text)


def latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if "a" <= c.lower() <= "z") / len(text)


def normalize_unicode(text: str) -> str:
    """Normalise to Unicode NFC.

    Devanagari admits two encodings of the same grapheme — क़ is either the
    precomposed U+0958 or क + U+093C (nukta) — and a tokenizer treats them as
    different characters. Note that for U+0958..U+095F NFC resolves toward the
    *decomposed* form, since those codepoints are Unicode composition
    exclusions; what matters is that both spellings converge on one.

    Measured on 8,000 documents / 42M characters from the Wikipedia dump and the
    web crawl, 0% of the corpus was non-NFC: MediaWiki and the news portals
    already serve NFC. This is therefore a guard against future sources rather
    than a correction, and it is idempotent on everything collected so far.
    """
    return unicodedata.normalize("NFC", text)


def clean_line(line: str) -> str:
    line = normalize_unicode(line)
    line = _URL_RE.sub(" ", line)
    line = _EMAIL_RE.sub(" ", line)
    line = _HTML_ENTITY_RE.sub(" ", line)
    line = _CITATION_RE.sub("", line)
    return _MULTISPACE_RE.sub(" ", line).strip()


def _cut_teasers(line: str, min_keep: int) -> str:
    """Truncate a line at the first "read also"-style teaser marker."""
    cut = len(line)
    for marker in TEASER_MARKERS:
        idx = line.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    if cut == len(line):
        return line
    return line[:cut].strip() if cut >= min_keep else ""


def clean_web_text(text: str, min_line_chars: int = 15,
                   max_line_latin_ratio: float = 0.40) -> str:
    """Line-level clean: drop navigation furniture and Latin-dominant lines.

    Latin-heavy lines are almost always template leftovers (timestamps, credits,
    social widgets). Dropping them per-line instead of per-document keeps
    otherwise-good Hindi articles that carry a footer.
    """
    kept: list[str] = []
    for raw_line in text.split("\n"):
        line = clean_line(raw_line)
        if not line:
            kept.append("")
            continue
        line = _cut_teasers(line, min_line_chars)
        if not line:
            continue
        if any(marker in line for marker in BOILERPLATE_MARKERS):
            continue
        if len(line) < min_line_chars and devanagari_ratio(line) < 0.5:
            continue
        if latin_ratio(line) > max_line_latin_ratio:
            continue
        kept.append(line)

    out = "\n".join(kept)
    out = _MULTINEWLINE_RE.sub("\n\n", out)
    return out.strip()


def passes_quality(text: str, min_chars: int = 200,
                   min_dev_ratio: float = 0.50,
                   max_latin_ratio: float = 0.10) -> bool:
    return (len(text) >= min_chars
            and devanagari_ratio(text) >= min_dev_ratio
            and latin_ratio(text) <= max_latin_ratio)
