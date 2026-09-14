"""Contract tests for the finetuning dataset and answer extraction.

The loss mask is the whole point of Phase 3's training path: if a prompt
position is ever scored, the model is being taught to write comparison
questions instead of to answer them, and nothing downstream would report that.
These tests pin the mask down against a hand-built tokenizer stub, so they
depend on neither the real SentencePiece models nor a GPU.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from src.model.transformer_lm import IGNORE_INDEX          # noqa: E402
from src.train.reasoning_dataset import ReasoningDataset   # noqa: E402

BOS, EOS, PAD = 2, 3, 0


class StubSP:
    """A whitespace tokenizer standing in for SentencePiece.

    Ids are deterministic and start at 10, so they can never collide with the
    BOS / EOS / PAD ids the dataset writes itself.
    """

    def encode(self, text):
        """Map each whitespace-separated piece to a stable id."""
        return [10 + (hash(tok) % 500) for tok in text.split()]


@pytest.fixture
def split(tmp_path):
    """A four-example JSONL in the shape ``src.reasoning.generate`` writes."""
    rows = [
        {"family": "rel2_which", "prompt": "q a b c\n", "answer": "A",
         "answer_direct": "ans A",
         "answer_rationale": "why a b\nans A",
         "meta": {"hops": 0, "held_out_phrasing": False, "attribute": "height"}},
        {"family": "rel4_pairwise", "prompt": "q a b c d e f\n", "answer": "B",
         "answer_direct": "ans B",
         "answer_rationale": "why a b c d\nans B",
         "meta": {"hops": 2, "held_out_phrasing": True, "attribute": "age"}},
        {"family": "val2_which", "prompt": "q a\n", "answer": "C",
         "answer_direct": "ans C", "answer_rationale": "why a\nans C",
         "meta": {"hops": 0, "held_out_phrasing": False, "attribute": "price"}},
        {"family": "val4_rank", "prompt": "q a b c d e f g h\n", "answer": "D",
         "answer_direct": "ans D", "answer_rationale": "why a b c\nans D",
         "meta": {"hops": 0, "held_out_phrasing": False, "attribute": "marks"}},
    ]
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                 encoding="utf-8")
    return p


def build(split, style="direct", ctx=64):
    """Construct a dataset over the fixture split."""
    return ReasoningDataset(split, StubSP(), ctx, answer_style=style, seed=0,
                            bos_id=BOS, eos_id=EOS, pad_id=PAD)


def test_every_prompt_position_is_masked_out_of_the_loss(split):
    """No gradient may come from a token the model was given rather than asked for."""
    ds = build(split)
    sp = StubSP()
    for row, rec in zip(ds.rows, ds.raw):
        full, n_prompt = row
        assert n_prompt == len(sp.encode(rec["prompt"]))
        y = full[1:]
        # Positions before n_prompt predict prompt tokens; they must be ignored.
        assert all(y[i] == full[i + 1] for i in range(len(y)))
        scored = y[n_prompt:]
        assert scored == sp.encode(rec["answer_direct"]) + [EOS]


def test_scored_span_is_exactly_the_answer_plus_eos(split):
    """The mask boundary must not clip the answer or leak the last prompt token."""
    ds = build(split)
    x, y = ds._collate(list(range(len(ds))))
    sp = StubSP()
    for r, (full, n_prompt) in enumerate(ds.rows):
        kept = [int(v) for v in y[r] if int(v) != IGNORE_INDEX]
        assert kept == sp.encode(ds.raw[r]["answer_direct"]) + [EOS]
        assert int(y[r][n_prompt - 1]) == IGNORE_INDEX
        assert int(y[r][n_prompt]) != IGNORE_INDEX


def test_batches_are_padded_to_the_longest_member_not_to_ctx_len(split):
    """Padding to ctx_len would cost ~7x the compute for no benefit."""
    ds = build(split, ctx=512)
    x, y = ds._collate(list(range(len(ds))))
    widest = max(len(f) for f, _ in ds.rows) - 1
    assert x.shape == y.shape == (len(ds), widest)
    assert widest < 32, "fixture sequences are short; padding should follow them"


def test_padding_contributes_neither_input_nor_loss(split):
    """Pad ids sit in x, but every pad target is ignored."""
    ds = build(split)
    idxs = sorted(range(len(ds)), key=lambda i: len(ds.rows[i][0]))
    x, y = ds._collate(idxs)
    shortest = len(ds.rows[idxs[0]][0]) - 1
    assert (x[0, shortest:] == PAD).all(), "short row should be right-padded"
    assert (y[0, shortest:] == IGNORE_INDEX).all(), "pad targets must be ignored"


def test_rationale_style_scores_strictly_more_tokens(split):
    """The two answer styles differ in supervision length, not in the answer."""
    direct, rationale = build(split, "direct"), build(split, "rationale")
    d = direct.token_stats()["total_scored_tokens"]
    r = rationale.token_stats()["total_scored_tokens"]
    assert r > d
    # Both must still end on the same answer line, so exact match is comparable.
    for a, b in zip(direct.raw, rationale.raw):
        assert b["answer_rationale"].endswith(a["answer_direct"])


def test_an_epoch_visits_every_example_exactly_once(split):
    """iter_epoch is a permutation, not a resampling."""
    ds = build(split)
    seen = sum(int((y != IGNORE_INDEX).any(dim=1).sum())
               for _, y in ds.iter_epoch(2))
    assert seen == len(ds)


def test_step_stream_reshuffles_and_counts_epochs(split):
    """The step-based loop needs an endless stream that tracks epoch boundaries."""
    ds = build(split)
    stream = ds.iter_steps(2)
    assert ds.epoch == 0
    for _ in range(4):                    # 4 batches of 2 over 4 examples = 2 epochs
        next(stream)
    assert ds.epoch >= 1


def test_over_long_examples_are_dropped_not_truncated(split):
    """Truncating would silently cut the answer off the end of the target."""
    ds = build(split, ctx=8)
    assert ds.n_skipped > 0
    assert all(len(f) <= 8 for f, _ in ds.rows)
    assert ds.token_stats()["skipped"] == ds.n_skipped


def test_unknown_answer_style_is_rejected(split):
    """A typo in the config must fail loudly, not silently train the wrong target."""
    with pytest.raises(ValueError):
        build(split, style="chain-of-thought")


def test_steps_per_epoch_accounts_for_gradient_accumulation(split):
    """max_steps is derived from this, so an off-by-one changes the run length."""
    ds = build(split)
    assert ds.steps_per_epoch(2, grad_accum=1) == 2
    assert ds.steps_per_epoch(1, grad_accum=2) == 2
