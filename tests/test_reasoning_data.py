"""Ground-truth and leakage tests for the synthetic reasoning corpus.

These deliberately do not import torch, so they run anywhere the generator
runs. The central test is :func:`test_rendered_premises_imply_the_label`, which
re-derives every answer by *parsing the rendered Devanagari back into an
ordering* rather than by reading the metadata. Checking the label against the
metadata would only prove the generator agrees with itself; a premise renderer
that states an inequality backwards would pass that and silently invert a
fraction of the corpus. Parsing the text is the only check that closes it.

Substring matching is done after entity names are removed from the sentence.
Several names contain the comparison words as substrings -- Nepali ``कमला`` and
Hindi ``कम्बल`` both contain ``कम`` ("less") -- so a naive scan would misread
the direction of any premise mentioning them.
"""

import random
import re

import pytest

from src.reasoning.entities import ATTRIBUTES, FRAMES, OBJECTS, PEOPLE, pool
from src.reasoning.generate import build_split, make_example, sample_values
from src.reasoning.templates import DEVANAGARI_DIGITS, FAMILIES, devanagari

LANGS = ("hindi", "nepali")
N = 400          # examples per language per test; enough to hit every family


def corpus(language, split="train", n=N, seed=7):
    """Generate a small deduplicated split for a test to inspect."""
    return build_split(language, split, n, random.Random(seed), set(), 0.25)


# A name may occur inside an ordinary word -- Hindi "राम" sits inside
# "किलोग्राम", the unit -- so a plain substring scan invents entity mentions.
# Every name in these templates begins a word, so requiring that the preceding
# character is not a Devanagari letter is both sufficient and exact.
_NOT_DEVA_BEFORE = r"(?<![\u0900-\u097F])"


def find_names(text, names):
    """Locate genuine entity mentions in ``text``.

    Args:
        text: Rendered premise or question text.
        names: Candidate entity surface forms.

    Returns:
        ``(position, name)`` pairs sorted by position, excluding matches that
        fall inside a longer word.
    """
    return sorted((m.start(), nm) for nm in names
                  for m in re.finditer(_NOT_DEVA_BEFORE + re.escape(nm), text))


def strip_names(sentence, names):
    """Remove every entity mention, longest first, so no name masks a keyword."""
    for nm in sorted(names, key=len, reverse=True):
        sentence = re.sub(_NOT_DEVA_BEFORE + re.escape(nm), " ", sentence)
    return sentence


def parse_premise(sentence, names, fr):
    """Recover ``(higher, lower)`` or ``(a, b, "equal")`` from one premise.

    Args:
        sentence: One rendered premise.
        names: Candidate entity surface forms.
        fr: ``FRAMES[language]``.

    Returns:
        ``(first_named, second_named, relation)`` where relation is ``">"``,
        ``"<"`` (first is the smaller) or ``"="``.
    """
    hits = find_names(sentence, names)
    assert len(hits) >= 2, f"premise names {hits} in {sentence!r}"
    a, b = hits[0][1], hits[1][1]
    bare = strip_names(sentence, names)
    if fr["relation_equal"] in bare:
        return a, b, "="
    if fr["relation_less"] in bare:
        return a, b, "<"
    return a, b, ">"


def _deva_to_int(text):
    """Parse a Devanagari numeral run into an int."""
    return int("".join(str(DEVANAGARI_DIGITS.index(c)) for c in text))


def ordering_from_premises(prompt, names, fr):
    """Recover the entity ordering from the rendered premises alone.

    Two premise modes need two recovery routes, and which one applies is read
    off the text rather than from the metadata:

      * **value premises** state one entity and one quantity per clause, so
        every name occurs exactly once and the ordering comes from the numerals;
      * **relative premises** state a pair per clause, so names recur and the
        ordering comes from topologically sorting the inequality edges.

    Args:
        prompt: The full prompt; only the first line holds premises.
        names: Entity surface forms in this example.
        fr: ``FRAMES[language]``.

    Returns:
        A list of ``set``s, largest rank first; tied entities share a set.
    """
    line = prompt.split("\n")[0].replace(fr["question_prefix"], "", 1)

    positions = find_names(line, names)
    # A quantity in the premises is what distinguishes the two modes: relative
    # premises never state one. (Counting name occurrences does not work --
    # rel2_which has two names in one clause, exactly like a value premise.)
    if re.search(r"[०-९]", line):
        # Value mode: the numeral that follows each name, before the next one.
        values = {}
        for i, (pos, nm) in enumerate(positions):
            stop = positions[i + 1][0] if i + 1 < len(positions) else len(line)
            found = re.search(r"[०-९]+", line[pos:stop])
            assert found, f"no quantity after {nm!r} in {line!r}"
            values[nm] = _deva_to_int(found.group())
        groups = {}
        for nm, v in values.items():
            groups.setdefault(v, set()).add(nm)
        return [groups[v] for v in sorted(groups, reverse=True)]

    # Relative mode: every realisation of a relative premise ends in a danda.
    greater, equal = [], []
    for sentence in (x for x in line.split("।") if x.strip()):
        if len({nm for _, nm in find_names(sentence, names)}) < 2:
            continue
        a, b, rel = parse_premise(sentence, names, fr)
        if rel == "=":
            equal.append((a, b))
        else:
            greater.append((a, b) if rel == ">" else (b, a))

    parent = {nm: nm for nm in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in equal:
        parent[find(a)] = find(b)

    groups = {}
    for nm in names:
        groups.setdefault(find(nm), set()).add(nm)

    edges = {(find(a), find(b)) for a, b in greater if find(a) != find(b)}
    order, remaining = [], dict(groups)
    while remaining:
        tops = [g for g in remaining
                if not any(lo == g and hi in remaining for hi, lo in edges)]
        assert tops, f"cycle in premises: {edges}"
        g = tops[0]
        order.append(remaining.pop(g))
    return order


@pytest.mark.parametrize("language", LANGS)
def test_entity_pools_are_disjoint_between_train_and_test(language):
    """Held-out entities are the primary leakage control; verify they are held out."""
    for table in (PEOPLE, OBJECTS):
        tr = {n for n, _ in table[language]["train"]}
        te = {n for n, _ in table[language]["test"]}
        assert tr and te
        assert not (tr & te), f"{language}: {sorted(tr & te)} in both pools"


@pytest.mark.parametrize("language", LANGS)
def test_test_split_uses_only_held_out_entities(language):
    """No training entity may appear anywhere in the test split."""
    train_names = {n for n, _ in PEOPLE[language]["train"]} | \
                  {n for n, _ in OBJECTS[language]["train"]}
    for ex in corpus(language, "test", 200):
        assert not (set(ex.meta["entities"]) & train_names), ex.prompt


@pytest.mark.parametrize("language", LANGS)
def test_rendered_premises_imply_the_label(language):
    """Re-derive every answer from the rendered text, not from the metadata.

    This is the test that would catch an inverted premise renderer, a
    superlative pointing the wrong way, or an off-by-one in the rank question.
    """
    fr = FRAMES[language]
    checked = 0
    for ex in corpus(language, "train", N):
        names = ex.meta["entities"]
        groups = ordering_from_premises(ex.prompt, names, fr)
        flat = [nm for g in groups for nm in sorted(g)]
        assert set(flat) == set(names), (ex.prompt, groups)

        qform = ex.meta["question_kind"]
        question = ex.prompt.split("\n")[1]
        if qform in ("extreme", "which"):
            # The answer must be an extreme of the recovered ordering.
            assert ex.answer in (groups[0] | groups[-1]), (ex.prompt, groups)
            expected = groups[0] if ex.meta["direction"] == "max" else groups[-1]
            assert ex.answer in expected, (ex.prompt, ex.answer, groups)
        elif qform == "relation":
            a, b = ex.meta["asked"]
            ra = next(i for i, g in enumerate(groups) if a in g)
            rb = next(i for i, g in enumerate(groups) if b in g)
            want = (fr["relation_equal"] if ra == rb else
                    fr["relation_more"] if ra < rb else fr["relation_less"])
            assert ex.answer == want, (ex.prompt, ex.answer, want, groups)
        else:                                     # rank
            ordinal = next(k for k, v in fr["ordinals"].items() if v in question)
            assert ex.answer in groups[ordinal - 1], (ex.prompt, groups, ordinal)
        checked += 1
    assert checked == N


@pytest.mark.parametrize("language", LANGS)
def test_value_premises_state_the_recorded_quantities(language):
    """Every quantity in a value-premise example appears in its rendered text."""
    for ex in corpus(language, "train", N):
        if not ex.family.startswith("val"):
            continue
        for v in ex.meta["values"]:
            assert devanagari(v) in ex.prompt, (v, ex.prompt)
        assert not any(c.isdigit() and c not in DEVANAGARI_DIGITS
                       for c in ex.prompt), "ASCII digits leaked into the text"


@pytest.mark.parametrize("language", LANGS)
def test_every_family_and_attribute_is_reachable(language):
    """The corpus must exercise all ten families and all six attributes."""
    rows = corpus(language, "train", 600)
    assert {r.family for r in rows} == set(FAMILIES)
    assert {r.meta["attribute"] for r in rows} == set(ATTRIBUTES[language])


@pytest.mark.parametrize("language", LANGS)
def test_equality_is_reachable_as_an_answer(language):
    """"बराबर" must be producible, or the three-way relation task is really two-way."""
    rows = corpus(language, "train", 1200)
    answers = {r.answer for r in rows if r.meta["question_kind"] == "relation"}
    assert FRAMES[language]["relation_equal"] in answers
    assert FRAMES[language]["relation_more"] in answers
    assert FRAMES[language]["relation_less"] in answers


@pytest.mark.parametrize("language", LANGS)
def test_generation_is_deterministic_under_a_seed(language):
    """Same seed, same corpus -- the reproducibility claim in the report."""
    a = [e.prompt + e.answer_rationale for e in corpus(language, "train", 120, 3)]
    b = [e.prompt + e.answer_rationale for e in corpus(language, "train", 120, 3)]
    c = [e.prompt + e.answer_rationale for e in corpus(language, "train", 120, 4)]
    assert a == b
    assert a != c


@pytest.mark.parametrize("language", LANGS)
def test_held_out_phrasing_never_reaches_train_or_val(language):
    """Reserved surface realisations are a test-only resource."""
    for split in ("train", "val"):
        assert not any(e.meta["held_out_phrasing"]
                       for e in corpus(language, split, 200))


@pytest.mark.parametrize("language", LANGS)
def test_distinct_values_are_separated_by_the_minimum_gap(language):
    """Comparisons must not hinge on a one-unit difference."""
    from src.reasoning.generate import MIN_GAP
    rng = random.Random(11)
    for attribute in ATTRIBUTES[language]:
        for _ in range(50):
            vals = sorted(sample_values(attribute, language, 4, rng))
            gaps = [b - a for a, b in zip(vals, vals[1:]) if b != a]
            assert all(g >= MIN_GAP[attribute] for g in gaps), (attribute, vals)


@pytest.mark.parametrize("language", LANGS)
def test_prompts_are_deduplicated_across_splits(language):
    """One hash set spans all three splits, so no prompt may straddle them."""
    seen, rng = set(), random.Random(5)
    splits = {s: build_split(language, s, 150, rng, seen, 0.25)
              for s in ("test", "val", "train")}
    texts = {k: {e.prompt for e in v} for k, v in splits.items()}
    assert not (texts["train"] & texts["test"])
    assert not (texts["val"] & texts["test"])
    assert not (texts["train"] & texts["val"])
    for k, v in splits.items():
        assert len(texts[k]) == len(v), f"{k} contains duplicate prompts"


@pytest.mark.parametrize("language", LANGS)
def test_prompt_and_answer_carry_the_expected_markers(language):
    """Answer extraction keys off these markers, so their presence is a contract."""
    fr = FRAMES[language]
    for ex in corpus(language, "train", 100):
        assert ex.prompt.startswith(fr["question_prefix"])
        assert ex.prompt.endswith("\n")
        assert ex.answer_direct.startswith(fr["answer_prefix"])
        assert ex.answer_rationale.startswith(fr["rationale_prefix"])
        assert ex.answer_rationale.rstrip().endswith(ex.answer)
        assert fr["answer_prefix"] not in ex.prompt


@pytest.mark.parametrize("language", LANGS)
def test_answer_extraction_round_trips_every_generated_answer(language):
    """Whatever the generator emits, the parser must read back unchanged.

    This closes the loop between ``templates.py`` and ``answers.py``: an
    accuracy number is only meaningful if the scorer recovers the same string
    the generator called correct.
    """
    from src.reasoning.answers import extract_answer, normalise
    for ex in corpus(language, "train", 300):
        assert extract_answer(ex.answer_direct, language) == normalise(ex.answer)
        assert extract_answer(ex.answer_rationale, language) == normalise(ex.answer)


@pytest.mark.parametrize("language", LANGS)
def test_extraction_takes_the_final_marker_not_the_first(language):
    """Few-shot prompts and rationales both put earlier markers in the text."""
    from src.reasoning.answers import extract_answer
    fr = FRAMES[language]
    a = fr["answer_prefix"]
    assert extract_answer(f"{a} first\n{a} second", language) == "second"
    assert extract_answer("nothing here", language) == ""
    # A trailing danda is punctuation, not part of the answer.
    assert extract_answer(f"{a} राम।", language) == "राम"
