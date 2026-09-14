"""Per-language entity inventories and attribute vocabulary.

Everything the generator needs that is *language-specific* lives here, so
``templates.py`` can stay about sentence shape and ``generate.py`` about
sampling. Hindi and Nepali share the Devanagari script but not their grammar,
so the two inventories are written independently rather than transliterated
from one another.

Leakage control starts in this file. Every inventory is split into a ``train``
pool and a disjoint ``test`` pool. The test split of the generated dataset
draws entity names *only* from the test pool, so a model cannot pass the test
set by memorising which name won which comparison during finetuning -- it has
to have learned the relation, because it has never seen the entities before.
Validation deliberately draws from the train pool: it selects checkpoints, and
selecting on out-of-distribution entities would make early stopping measure
generalisation instead of convergence.

Gender agreement. Hindi predicate adjectives inflect
(``लम्बा`` / ``लम्बी``), so every animate entity carries a gender tag and every
Hindi adjective is stored as a ``(masculine, feminine)`` pair. Nepali is
handled with the invariant ``-ओ`` form, which is the standard citation form and
is what the corpus overwhelmingly contains. Inanimate objects carry the
grammatical gender the language assigns them, which is not always the same in
the two languages -- another reason the inventories are not shared.
"""

# --- people -----------------------------------------------------------------
# (name, gender). Gender drives Hindi adjective agreement only; it is never
# itself the subject of a question, so the reasoning task stays gender-neutral.

PEOPLE = {
    "hindi": {
        "train": [
            ("राम", "m"), ("श्याम", "m"), ("मोहन", "m"), ("सोहन", "m"),
            ("अमित", "m"), ("रवि", "m"), ("विजय", "m"), ("अनिल", "m"),
            ("दीपक", "m"), ("राहुल", "m"), ("संजय", "m"), ("अजय", "m"),
            ("सुरेश", "m"), ("रमेश", "m"), ("महेश", "m"), ("नरेश", "m"),
            ("विकास", "m"), ("प्रमोद", "m"), ("आनंद", "m"), ("गोपाल", "m"),
            ("गीता", "f"), ("सीता", "f"), ("सुनीता", "f"), ("कविता", "f"),
            ("प्रीति", "f"), ("नेहा", "f"), ("मीरा", "f"), ("पूजा", "f"),
            ("अंजलि", "f"), ("रेखा", "f"), ("ममता", "f"), ("शोभा", "f"),
            ("उषा", "f"), ("लता", "f"), ("राधा", "f"), ("सरिता", "f"),
            ("निशा", "f"), ("वंदना", "f"), ("आशा", "f"), ("किरण", "f"),
        ],
        "test": [
            ("करण", "m"), ("हिमांशु", "m"), ("यश", "m"), ("नितिन", "m"),
            ("प्रवीण", "m"), ("तरुण", "m"), ("मनीष", "m"), ("सचिन", "m"),
            ("अरुण", "m"), ("भरत", "m"),
            ("तनुजा", "f"), ("ऋतु", "f"), ("शालिनी", "f"), ("अर्चना", "f"),
            ("स्वाति", "f"), ("कल्पना", "f"), ("ज्योति", "f"), ("हेमा", "f"),
            ("सुधा", "f"), ("बबीता", "f"),
        ],
    },
    "nepali": {
        "train": [
            ("राम", "m"), ("श्याम", "m"), ("हरि", "m"), ("बिनोद", "m"),
            ("प्रकाश", "m"), ("दीपक", "m"), ("राजु", "m"), ("बिकास", "m"),
            ("मनोज", "m"), ("सुरेश", "m"), ("गोपाल", "m"), ("नरेश", "m"),
            ("किशोर", "m"), ("भीम", "m"), ("सन्तोष", "m"), ("अनिल", "m"),
            ("रमेश", "m"), ("कृष्ण", "m"), ("पुरुषोत्तम", "m"), ("बाबुराम", "m"),
            ("सीता", "f"), ("गीता", "f"), ("सुनिता", "f"), ("कमला", "f"),
            ("सरिता", "f"), ("अनिता", "f"), ("मीना", "f"), ("राधिका", "f"),
            ("शान्ति", "f"), ("पार्वती", "f"), ("लक्ष्मी", "f"), ("बिमला", "f"),
            ("सरस्वती", "f"), ("गंगा", "f"), ("तारा", "f"), ("इन्दिरा", "f"),
            ("यशोदा", "f"), ("दुर्गा", "f"), ("माया", "f"), ("रीता", "f"),
        ],
        "test": [
            ("सुरेन्द्र", "m"), ("भुवन", "m"), ("केशव", "m"), ("पदम", "m"),
            ("नबिन", "m"), ("दिलीप", "m"), ("हेमन्त", "m"), ("युवराज", "m"),
            ("प्रेम", "m"), ("जीवन", "m"),
            ("निर्मला", "f"), ("रञ्जना", "f"), ("सम्झना", "f"), ("उर्मिला", "f"),
            ("कुसुम", "f"), ("सपना", "f"), ("अस्मिता", "f"), ("प्रतिमा", "f"),
            ("रोशनी", "f"), ("सृजना", "f"),
        ],
    },
}

# --- objects ----------------------------------------------------------------
# Inanimate entities, used by the price / weight / length attributes. The
# gender tag is the noun's grammatical gender in that language.

OBJECTS = {
    "hindi": {
        "train": [
            ("पुस्तक", "f"), ("कुर्सी", "f"), ("मेज़", "f"), ("बैग", "m"),
            ("घड़ी", "f"), ("कलम", "f"), ("जूता", "m"), ("थैला", "m"),
            ("छाता", "m"), ("चश्मा", "m"), ("तकिया", "m"), ("कम्बल", "m"),
            ("टोकरी", "f"), ("रस्सी", "f"), ("चटाई", "f"), ("बर्तन", "m"),
        ],
        "test": [
            ("साइकिल", "f"), ("पंखा", "m"), ("ताला", "m"), ("कुरता", "m"),
            ("संदूक", "m"), ("बाल्टी", "f"), ("झाड़ू", "f"), ("दरी", "f"),
        ],
    },
    "nepali": {
        "train": [
            ("पुस्तक", "f"), ("कुर्सी", "f"), ("टेबुल", "m"), ("झोला", "m"),
            ("घडी", "f"), ("कलम", "f"), ("जुत्ता", "m"), ("छाता", "m"),
            ("चस्मा", "m"), ("सिरानी", "f"), ("कम्बल", "m"), ("भाँडा", "m"),
            ("डोको", "m"), ("डोरी", "f"), ("गुन्द्री", "f"), ("बटुको", "m"),
        ],
        "test": [
            ("साइकल", "f"), ("पंखा", "m"), ("ताल्चा", "m"), ("दौरा", "m"),
            ("बाकस", "m"), ("बाल्टिन", "m"), ("कुचो", "m"), ("ऐना", "m"),
        ],
    },
}

# --- attributes -------------------------------------------------------------
# Each attribute fixes: the noun used in the premise, the measurement unit, the
# plausible value range, whether it applies to people or objects, and the two
# polarity adjectives. Hindi adjectives are (masculine, feminine) pairs; Nepali
# uses the invariant -ओ form.
#
# ``value_range`` is (low, high, step). Steps larger than 1 keep the rendered
# numbers realistic (nobody is 163.7 cm in a template) and keep the tokenised
# numerals short, which matters at a 8k vocabulary.

ATTRIBUTES = {
    "hindi": {
        "height": {
            "domain": "people", "noun": "ऊँचाई", "unit": "सेंटीमीटर",
            "poss": "की", "copula": "है",
            "value_range": (120, 190, 1),
            "more": ("लम्बा", "लम्बी"), "less": ("छोटा", "छोटी"),
        },
        "age": {
            "domain": "people", "noun": "उम्र", "unit": "वर्ष",
            "poss": "की", "copula": "है",
            "value_range": (6, 70, 1),
            "more": ("बड़ा", "बड़ी"), "less": ("छोटा", "छोटी"),
        },
        "weight": {
            "domain": "people", "noun": "वज़न", "unit": "किलोग्राम",
            "poss": "का", "copula": "है",
            "value_range": (20, 95, 1),
            "more": ("भारी", "भारी"), "less": ("हल्का", "हल्की"),
        },
        "marks": {
            "domain": "people", "noun": "अंक", "unit": "",
            "poss": "के", "copula": "हैं",
            "value_range": (10, 100, 1),
            "more": ("अधिक", "अधिक"), "less": ("कम", "कम"),
            "adjectival": False,
        },
        "price": {
            "domain": "objects", "noun": "कीमत", "unit": "रुपये",
            "poss": "की", "copula": "है",
            "value_range": (20, 2000, 5),
            "more": ("महँगा", "महँगी"), "less": ("सस्ता", "सस्ती"),
        },
        "length": {
            "domain": "objects", "noun": "लम्बाई", "unit": "सेंटीमीटर",
            "poss": "की", "copula": "है",
            "value_range": (10, 200, 1),
            "more": ("लम्बा", "लम्बी"), "less": ("छोटा", "छोटी"),
        },
    },
    "nepali": {
        "height": {
            "domain": "people", "noun": "उचाइ", "unit": "सेन्टिमिटर",
            "poss": "को", "copula": "छ",
            "value_range": (120, 190, 1),
            "more": ("अग्लो", "अग्लो"), "less": ("होचो", "होचो"),
        },
        "age": {
            "domain": "people", "noun": "उमेर", "unit": "वर्ष",
            "poss": "को", "copula": "छ",
            "value_range": (6, 70, 1),
            "more": ("जेठो", "जेठो"), "less": ("कान्छो", "कान्छो"),
        },
        "weight": {
            "domain": "people", "noun": "तौल", "unit": "किलोग्राम",
            "poss": "को", "copula": "छ",
            "value_range": (20, 95, 1),
            "more": ("गह्रौँ", "गह्रौँ"), "less": ("हलुका", "हलुका"),
        },
        "marks": {
            "domain": "people", "noun": "अंक", "unit": "",
            "poss": "का", "copula": "छन्",
            "value_range": (10, 100, 1),
            "more": ("बढी", "बढी"), "less": ("कम", "कम"),
            "adjectival": False,
        },
        "price": {
            "domain": "objects", "noun": "मूल्य", "unit": "रुपैयाँ",
            "poss": "को", "copula": "छ",
            "value_range": (20, 2000, 5),
            "more": ("महँगो", "महँगो"), "less": ("सस्तो", "सस्तो"),
        },
        "length": {
            "domain": "objects", "noun": "लम्बाइ", "unit": "सेन्टिमिटर",
            "poss": "को", "copula": "छ",
            "value_range": (10, 200, 1),
            "more": ("लामो", "लामो"), "less": ("छोटो", "छोटो"),
        },
    },
}

# --- fixed lexical frames ---------------------------------------------------
# The tokens that mark structure rather than content. They are identical in
# every example of a language, so the finetuned model can learn them as format
# rather than as reasoning -- which is exactly the point: answer extraction in
# src/eval/reasoning_eval.py keys off ANSWER_PREFIX.

FRAMES = {
    "hindi": {
        "gen_join": " ",                      # "राम की ऊँचाई" -- clitic, spaced
        "than": "{x} से",                     # "श्याम से अधिक"
        "than_alt": "ज़्यादा",                 # held-out lexical variant of "अधिक"
        "who_gen": {"की": "किसकी", "का": "किसका", "के": "किसके"},
        "superlative": "सबसे",
        "ordinals": {2: "दूसरे", 3: "तीसरे"},
        "descending": "के घटते क्रम में",
        "among": "में",
        "question_prefix": "प्रश्न:",
        "answer_prefix": "उत्तर:",
        "rationale_prefix": "तर्क:",
        "and": "और",
        "gt": ">", "lt": "<", "eq": "=",
        "relation_more": "अधिक", "relation_less": "कम", "relation_equal": "बराबर",
    },
    "nepali": {
        "gen_join": "",                       # "रामको उचाइ" -- clitic, attached
        "than": "{x}भन्दा",                   # "श्यामभन्दा बढी"
        "than_alt": "धेरै",                   # held-out lexical variant of "बढी"
        "who_gen": {"को": "कसको", "का": "कसका"},
        "superlative": "सबैभन्दा",
        "ordinals": {2: "दोस्रो", 3: "तेस्रो"},
        "descending": "को घट्दो क्रममा",
        "among": "मध्ये",
        "question_prefix": "प्रश्न:",
        "answer_prefix": "उत्तर:",
        "rationale_prefix": "तर्क:",
        "and": "र",
        "gt": ">", "lt": "<", "eq": "=",
        "relation_more": "बढी", "relation_less": "कम", "relation_equal": "बराबर",
    },
}


def pool(language: str, domain: str, split: str) -> list:
    """Return the entity pool for one language, domain and split.

    Args:
        language: ``"hindi"`` or ``"nepali"``.
        domain: ``"people"`` or ``"objects"``.
        split: ``"train"``, ``"val"`` or ``"test"``. Validation shares the
            train pool by design (see the module docstring); only ``"test"``
            reaches the held-out names.

    Returns:
        A list of ``(surface_form, gender)`` tuples.
    """
    table = PEOPLE if domain == "people" else OBJECTS
    return table[language]["test" if split == "test" else "train"]


def attributes_for(language: str, domain: str) -> list:
    """Names of the attributes that apply to a given entity domain."""
    return [k for k, v in ATTRIBUTES[language].items() if v["domain"] == domain]
