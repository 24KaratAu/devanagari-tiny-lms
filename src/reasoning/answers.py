"""Parsing the answer back out of generated text.

This lives beside the generator rather than in ``src/eval`` because it is the
other half of one contract: ``templates.py`` decides that an answer is the text
following ``उत्तर:`` on its own line, and this module is what reads that
decision back. Keeping the pair together means a change to the emitted format
cannot drift away from the parser that consumes it -- and it keeps both free of
a torch dependency, so the format contract is testable without a GPU.
"""

import re
import unicodedata

from src.reasoning.entities import FRAMES

# Trailing sentence punctuation that must not decide a match. The Devanagari
# danda is the one that actually matters: models trained on running text emit it
# after a clause far more often than the templates do.
TRAILING = "।.,?!:;\"'()[]{}"


def normalise(text: str) -> str:
    """Canonical form for exact-match comparison.

    Devanagari admits more than one encoding of the same grapheme -- a composed
    character, or a base plus a combining mark -- so a byte comparison can call
    two identical-looking strings different and quietly under-report accuracy.
    NFC folds those together; the rest strips whitespace and edge punctuation,
    which the model may or may not emit and which is not part of the answer.

    Args:
        text: Raw predicted or gold answer.

    Returns:
        The normalised string.
    """
    t = unicodedata.normalize("NFC", text).strip()
    t = re.sub(r"\s+", " ", t)
    return t.strip(TRAILING).strip()


def extract_answer(continuation: str, language: str) -> str:
    """Pull the answer span out of a generated continuation.

    The *final* marker is used, not the first. Under the rationale answer style
    the model restates the premise chain before concluding, and a few-shot
    prompt puts complete worked examples in front of the question; in both cases
    an earlier marker belongs to something other than this model's conclusion.

    Args:
        continuation: Everything the model produced after the prompt.
        language: Selects the answer prefix from ``FRAMES``.

    Returns:
        The normalised answer, or ``""`` when the model never emitted the
        prefix. That empty case is a real finding for the zero-shot baseline --
        it is reported as the format rate and counted as a miss, never retried.
    """
    prefix = FRAMES[language]["answer_prefix"]
    idx = continuation.rfind(prefix)
    if idx < 0:
        return ""
    return normalise(continuation[idx + len(prefix):].split("\n")[0])
