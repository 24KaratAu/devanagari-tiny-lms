"""Template families and surface realisations for the synthetic reasoning set.

Ten families span the range the brief asks for: direct two-entity comparison,
ordering over explicit quantities, and transitive chains where the two entities
the question asks about are *never* compared to each other in the premises, so
the answer can only come from chaining inequalities.

    family            n  premises   question              hops
    rel2_which        2  relative   which one             0
    val2_which        2  values     which one             0
    val2_relation     2  values     more / less / equal    0
    rel3_extreme      3  chain      largest / smallest     1
    rel3_pairwise     3  chain      endpoint relation      1
    val3_extreme      3  values     largest / smallest     0
    val4_rank         4  values     k-th in descending     0
    eq3_relation      3  chain+eq   endpoint relation      1
    rel4_extreme      4  chain      largest / smallest     2
    rel4_pairwise     4  chain      endpoint relation      2

``hops`` is the number of transitive steps the premises force. It is carried
into every example's metadata so the report can plot accuracy against reasoning
depth rather than reporting one pooled number.

Two independent axes of variation keep the set from being ten fixed strings:
the *attribute* (six per language, over two entity domains) and the *surface
realisation* (several phrasings per premise and question type). One realisation
of each type is marked ``held_out`` and is emitted only into the test split, so
the report can separate "unseen entities" from "unseen phrasing" as two
different generalisation demands.

Numerals are rendered in Devanagari, not ASCII. The brief requires natural
target-language text, and it also makes a measurable difference: neither
tokenizer merged multi-digit ASCII runs, but Hindi has 42 Devanagari-numeral
pieces against Nepali's 158, so the same quantity fragments differently in the
two models. Each example records its numeral token cost so §Phase 3 of the
report can test whether that asymmetry shows up in accuracy.
"""

from dataclasses import dataclass, field

from src.reasoning.entities import ATTRIBUTES, FRAMES

DEVANAGARI_DIGITS = "०१२३४५६७८९"


@dataclass
class Example:
    """One rendered reasoning item.

    Attributes:
        family: Family id, e.g. ``"rel4_pairwise"``.
        prompt: Premises and question. Ends with a newline; the model continues
            from here.
        answer: The bare answer span -- an entity name or a relation word.
        answer_direct: ``"उत्तर: <answer>"``.
        answer_rationale: The premise chain restated, then the same answer line.
        meta: Provenance and analysis fields (attribute, hops, held-out flags,
            latent values). Never shown to the model.
    """
    family: str
    prompt: str
    answer: str
    answer_direct: str
    answer_rationale: str
    meta: dict = field(default_factory=dict)


# --- small rendering helpers ------------------------------------------------

def devanagari(n: int) -> str:
    """Render an integer with Devanagari digits.

    Args:
        n: A non-negative integer.

    Returns:
        The same number written with U+0966..U+096F.
    """
    return "".join(DEVANAGARI_DIGITS[int(c)] for c in str(n))


def genitive(name: str, attr: dict, fr: dict) -> str:
    """Attach the possessive marker the attribute noun requires.

    Hindi agrees with the possessed noun and writes the clitic separately
    (``राम का वज़न``); Nepali attaches it (``रामको तौल``).

    Args:
        name: Entity surface form.
        attr: One entry from ``ATTRIBUTES[language]``.
        fr: ``FRAMES[language]``.

    Returns:
        The genitive phrase, without the attribute noun itself.
    """
    return f"{name}{fr['gen_join']}{attr['poss']}"


def than(name: str, fr: dict) -> str:
    """Render "than <name>" (``श्याम से`` / ``श्यामभन्दा``)."""
    return fr["than"].format(x=name)


def _measure(value: int, attr: dict) -> str:
    """A quantity with its unit, skipping the unit when the noun already is one."""
    num = devanagari(value)
    return f"{num} {attr['unit']}" if attr["unit"] else num


# --- premise realisations ---------------------------------------------------
# Each takes the pair (high, low) as (name, gender) tuples and returns a
# sentence asserting that `high` exceeds `low` on `attr`.

def _rel_plain(high, low, attr, fr):
    """``राम की ऊँचाई श्याम से अधिक है।``"""
    return (f"{genitive(high[0], attr, fr)} {attr['noun']} {than(low[0], fr)} "
            f"{fr['relation_more']} {attr['copula']}।")


def _rel_inverted(high, low, attr, fr):
    """``श्याम की ऊँचाई राम से कम है।`` -- same fact, stated from the other side."""
    return (f"{genitive(low[0], attr, fr)} {attr['noun']} {than(high[0], fr)} "
            f"{fr['relation_less']} {attr['copula']}।")


def _rel_adjectival(high, low, attr, fr):
    """``राम, श्याम से लम्बा है।`` -- predicate adjective, agrees with the subject."""
    adj = attr["more"][0 if high[1] == "m" else 1]
    return f"{high[0]}, {than(low[0], fr)} {adj} है।" if fr["gen_join"] == " " \
        else f"{high[0]} {than(low[0], fr)} {adj} छ।"


def _rel_alt_lexical(high, low, attr, fr):
    """Held out: the same frame with a synonym for "more" (``ज़्यादा`` / ``धेरै``)."""
    return (f"{genitive(high[0], attr, fr)} {attr['noun']} {than(low[0], fr)} "
            f"{fr['than_alt']} {attr['copula']}।")


def _rel_equal(a, b, attr, fr):
    """``राम की ऊँचाई श्याम के बराबर है।`` -- the equality edge."""
    if fr["gen_join"] == " ":
        return (f"{genitive(a[0], attr, fr)} {attr['noun']} {b[0]} के "
                f"{fr['relation_equal']} {attr['copula']}।")
    return (f"{genitive(a[0], attr, fr)} {attr['noun']} {b[0]}को "
            f"{fr['relation_equal']} {attr['copula']}।")


REL_PREMISES = [
    ("plain", _rel_plain, False),
    ("inverted", _rel_inverted, False),
    ("adjectival", _rel_adjectival, False),
    ("alt_lexical", _rel_alt_lexical, True),
]


def _val_plain(ent, value, attr, fr):
    """``राम की ऊँचाई १५० सेंटीमीटर है।``"""
    return (f"{genitive(ent[0], attr, fr)} {attr['noun']} "
            f"{_measure(value, attr)} {attr['copula']}।")


def _val_colon(ent, value, attr, fr):
    """Held out: ``राम की ऊँचाई: १५० सेंटीमीटर`` -- a listing rather than a clause."""
    return f"{genitive(ent[0], attr, fr)} {attr['noun']}: {_measure(value, attr)}"


VAL_PREMISES = [
    ("plain", _val_plain, False),
    ("colon", _val_colon, True),
]


# --- question realisations --------------------------------------------------

def _q_extreme_gen(attr, fr, ents, direction):
    """``किसकी ऊँचाई सबसे अधिक है?`` -- interrogative genitive over the attribute."""
    who = fr["who_gen"][attr["poss"]]
    deg = fr["relation_more"] if direction == "max" else fr["relation_less"]
    return f"{who} {attr['noun']} {fr['superlative']} {deg} {attr['copula']}?"


def _q_extreme_adj(attr, fr, ents, direction):
    """``सबसे लम्बा कौन है?`` -- superlative adjective, masculine interrogative."""
    adj = (attr["more"] if direction == "max" else attr["less"])[0]
    tail = "कौन है?" if fr["gen_join"] == " " else "को हो?"
    return f"{fr['superlative']} {adj} {tail}"


Q_EXTREME = [
    ("genitive", _q_extreme_gen, False),
    ("adjectival", _q_extreme_adj, True),
]


def _q_which(attr, fr, pair, direction):
    """``राम और मोहन में किसकी ऊँचाई अधिक है?`` -- pick one of two named entities."""
    who = fr["who_gen"][attr["poss"]]
    deg = fr["relation_more"] if direction == "max" else fr["relation_less"]
    return (f"{pair[0][0]} {fr['and']} {pair[1][0]}{fr['gen_join']}"
            f"{fr['among']} {who} {attr['noun']} {deg} {attr['copula']}?")


def _q_relation(attr, fr, pair, direction):
    """``राम की ऊँचाई मोहन से अधिक है, कम है या बराबर?`` -- answer is a relation."""
    a, b = pair
    if fr["gen_join"] == " ":
        return (f"{genitive(a[0], attr, fr)} {attr['noun']} {than(b[0], fr)} "
                f"{fr['relation_more']} {attr['copula']}, "
                f"{fr['relation_less']} {attr['copula']} या "
                f"{fr['relation_equal']}?")
    return (f"{genitive(a[0], attr, fr)} {attr['noun']} {than(b[0], fr)} "
            f"{fr['relation_more']} {attr['copula']}, "
            f"{fr['relation_less']} {attr['copula']} वा "
            f"{fr['relation_equal']}?")


def _q_rank(attr, fr, ents, k):
    """``ऊँचाई के घटते क्रम में दूसरे स्थान पर कौन है?``"""
    ordinal = fr["ordinals"][k]
    if fr["gen_join"] == " ":
        return f"{attr['noun']} {fr['descending']} {ordinal} स्थान पर कौन है?"
    return f"{attr['noun']}{fr['descending']} {ordinal} स्थानमा को हो?"


# --- family table -----------------------------------------------------------
# premise: "relative" states consecutive pairs of the latent ordering;
#          "values" states each entity's quantity outright.
# question: which of the four question renderers to use.
# hops: transitive steps the premises force before the answer follows.

FAMILIES = {
    "rel2_which":    {"n": 2, "premise": "relative", "question": "which",    "hops": 0},
    "val2_which":    {"n": 2, "premise": "values",   "question": "which",    "hops": 0},
    "val2_relation": {"n": 2, "premise": "values",   "question": "relation", "hops": 0},
    "rel3_extreme":  {"n": 3, "premise": "relative", "question": "extreme",  "hops": 1},
    "rel3_pairwise": {"n": 3, "premise": "relative", "question": "relation", "hops": 1},
    "val3_extreme":  {"n": 3, "premise": "values",   "question": "extreme",  "hops": 0},
    "val4_rank":     {"n": 4, "premise": "values",   "question": "rank",     "hops": 0},
    "eq3_relation":  {"n": 3, "premise": "equality", "question": "relation", "hops": 1},
    "rel4_extreme":  {"n": 4, "premise": "relative", "question": "extreme",  "hops": 2},
    "rel4_pairwise": {"n": 4, "premise": "relative", "question": "relation", "hops": 2},
}


def _pick_realisation(options: list, rng, allow_held_out: bool) -> tuple:
    """Choose a surface realisation, respecting the held-out reservation.

    Args:
        options: ``(name, fn, held_out)`` triples.
        rng: A ``random.Random``.
        allow_held_out: When False, realisations reserved for the test split are
            excluded. When True, one is chosen deliberately.

    Returns:
        The chosen ``(name, fn, held_out)`` triple.
    """
    pool = [o for o in options if o[2]] if allow_held_out else \
           [o for o in options if not o[2]]
    if not pool:                      # a type with no held-out variant
        pool = [o for o in options if not o[2]]
    return rng.choice(pool)


def _chain_text(order: list, values: list, fr: dict, relative: bool) -> str:
    """The rationale body: the latent ordering, written out.

    Args:
        order: Entities sorted from largest to smallest.
        values: Their quantities, in the same order.
        fr: ``FRAMES[language]``.
        relative: Render inequality edges between neighbours rather than
            ``entity = quantity`` pairs.

    Returns:
        e.g. ``राम > श्याम, श्याम > मोहन`` or ``श्याम = १६०, राम = १५०``.
    """
    if relative:
        parts = []
        for (a, va), (b, vb) in zip(zip(order, values), zip(order[1:], values[1:])):
            op = fr["eq"] if va == vb else fr["gt"]
            parts.append(f"{a[0]} {op} {b[0]}")
        return ", ".join(parts)
    return ", ".join(f"{e[0]} {fr['eq']} {devanagari(v)}"
                     for e, v in zip(order, values))


def render(family: str, language: str, attribute: str, entities: list,
           values: list, rng, held_out_phrasing: bool = False) -> Example:
    """Render one example of a family from an already-sampled scenario.

    The caller owns attribute, entity and value sampling (``generate.py``) so
    pool disjointness and value ranges stay in one place; this function owns
    nothing but surface form and the ground-truth answer.

    Premise order is shuffled before rendering. Without that the first premise
    would always mention the largest entity, and the task would be solvable
    from position alone rather than from the relation.

    Args:
        family: A key of ``FAMILIES``.
        language: ``"hindi"`` or ``"nepali"``.
        attribute: A key of ``ATTRIBUTES[language]``; its ``domain`` must match
            the pool ``entities`` was drawn from.
        entities: ``n`` ``(name, gender)`` tuples, already distinct.
        values: Their latent quantities, same order, larger meaning "more".
        rng: A ``random.Random``.
        held_out_phrasing: Draw the reserved surface realisations instead of the
            shared ones. Only the test split passes True.

    Returns:
        A fully populated ``Example``.
    """
    spec = FAMILIES[family]
    fr = FRAMES[language]
    attr = ATTRIBUTES[language][attribute]
    adjectival_ok = attr.get("adjectival", True)

    # Latent ordering, largest first. Ties occur only in eq3_relation.
    ranked = sorted(zip(entities, values), key=lambda t: -t[1])
    order = [e for e, _ in ranked]
    ovals = [v for _, v in ranked]

    # --- premises ----------------------------------------------------------
    if spec["premise"] == "values":
        pname, pfn, _ = _pick_realisation(VAL_PREMISES, rng, held_out_phrasing)
        premises = [pfn(e, v, attr, fr) for e, v in zip(entities, values)]
    else:
        opts = [o for o in REL_PREMISES if adjectival_ok or o[0] != "adjectival"]
        pname, pfn, _ = _pick_realisation(opts, rng, held_out_phrasing)
        premises = []
        for (a, va), (b, vb) in zip(zip(order, ovals), zip(order[1:], ovals[1:])):
            premises.append(_rel_equal(a, b, attr, fr) if va == vb
                            else pfn(a, b, attr, fr))
    rng.shuffle(premises)

    # --- question and ground truth -----------------------------------------
    qkind = spec["question"]
    direction = rng.choice(["max", "min"])
    if qkind == "extreme":
        opts = [o for o in Q_EXTREME if adjectival_ok or o[0] != "adjectival"]
        qname, qfn, _ = _pick_realisation(opts, rng, held_out_phrasing)
        question = qfn(attr, fr, order, direction)
        answer = (order[0] if direction == "max" else order[-1])[0]
        asked = [order[0], order[-1]]
    elif qkind == "which":
        qname = "which"
        pair = [order[0], order[-1]]
        rng.shuffle(pair)
        question = _q_which(attr, fr, pair, direction)
        answer = (order[0] if direction == "max" else order[-1])[0]
        asked = pair
    elif qkind == "relation":
        qname = "relation"
        pair = [order[0], order[-1]]
        rng.shuffle(pair)
        va = ovals[order.index(pair[0])]
        vb = ovals[order.index(pair[1])]
        question = _q_relation(attr, fr, pair, direction)
        answer = (fr["relation_equal"] if va == vb else
                  fr["relation_more"] if va > vb else fr["relation_less"])
        asked = pair
    else:                                             # rank
        qname = "rank"
        k = rng.choice([2, 3])
        question = _q_rank(attr, fr, order, k)
        answer = order[k - 1][0]
        asked = [order[k - 1]]

    prompt = (f"{fr['question_prefix']} " + " ".join(premises) + "\n"
              + question + "\n")
    chain = _chain_text(order, ovals, fr, spec["premise"] != "values")
    direct = f"{fr['answer_prefix']} {answer}"
    rationale = f"{fr['rationale_prefix']} {chain}\n{direct}"

    return Example(
        family=family, prompt=prompt, answer=answer,
        answer_direct=direct, answer_rationale=rationale,
        meta={
            "language": language, "attribute": attribute,
            "domain": attr["domain"], "hops": spec["hops"],
            "n_entities": spec["n"], "direction": direction,
            # Two independent axes, previously conflated: `question_kind` is
            # what is being asked (extreme / which / relation / rank);
            # `question_form` is which surface realisation asked it.
            "question_kind": qkind, "premise_form": pname,
            "question_form": qname,
            "held_out_phrasing": held_out_phrasing,
            "entities": [e[0] for e in entities],
            "values": list(values), "ordering": [e[0] for e in order],
            "asked": [e[0] for e in asked],
        },
    )
