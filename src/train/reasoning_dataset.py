"""Supervised dataset for Phase 3 finetuning: prompt in, answer scored.

Pretraining reads a flat ``uint16`` token stream and takes the loss on every
position, because every position is a legitimate next-token prediction. A
reasoning finetune cannot do that. Half of each sequence is the *question*,
which the model is given rather than asked to produce; training it to predict
the premises teaches it to write comparison questions, not to answer them. So
every prompt position is written to ``IGNORE_INDEX`` and the loss falls on the
answer span alone.

Sequence layout, before the causal shift::

    <bos>  प्रश्न: ... premises ... question ?  \\n   उत्तर: राम   <eos>
    |------------------ prompt ------------------|--- target ---|

    x = full[:-1]
    y = full[1:], with y[i] = IGNORE_INDEX for i < len(prompt)

``y[i]`` predicts ``full[i+1]``, so masking the first ``len(prompt)`` positions
of ``y`` leaves exactly the answer tokens and the terminating ``<eos>`` scored.
Because ``cross_entropy`` reduces over non-ignored elements only, the reported
loss is a mean over answer tokens and is directly comparable between the two
answer styles, which have very different lengths (~3 tokens direct, ~19 with a
rationale).

Padding. Examples are short -- 37 prompt tokens on average against a 512-token
context -- so batches are padded to the longest member rather than to
``ctx_len``, which cuts the compute per step by roughly 7x. Padding sits on the
right, and under a causal mask no real token can attend to a position after it,
so the pad ids are unreachable; their targets are ignored as well, so they
contribute nothing to the loss either. No attention-mask change is needed.
"""

import json
import random
from pathlib import Path

import torch

from src.model.transformer_lm import IGNORE_INDEX

ANSWER_STYLES = ("direct", "rationale")


class ReasoningDataset:
    """Tokenised reasoning examples with prompt-masked targets.

    Attributes:
        rows: ``(full_ids, n_prompt)`` pairs, one per retained example.
        n_skipped: Examples dropped for exceeding ``ctx_len``.
    """

    def __init__(self, jsonl_path, sp, ctx_len: int,
                 answer_style: str = "direct", seed: int = 42,
                 device: str = "cpu", bos_id: int = 2, eos_id: int = 3,
                 pad_id: int = 0):
        """Load and tokenise one split.

        Args:
            jsonl_path: A ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl``
                written by ``src.reasoning.generate``.
            sp: A loaded ``sentencepiece.SentencePieceProcessor`` for this
                language. The finetune keeps the pretraining tokenizer and
                vocabulary fixed, so this is the Phase 1 model unchanged.
            ctx_len: Model context length; longer examples are dropped.
            answer_style: ``"direct"`` scores ``उत्तर: <answer>``;
                ``"rationale"`` scores the restated chain and then the same
                answer line.
            seed: Shuffling seed.
            device: Where batches are materialised.
            bos_id: Beginning-of-sequence id.
            eos_id: End-of-sequence id; always the final scored token.
            pad_id: Padding id for ``x``.

        Raises:
            ValueError: On an unknown ``answer_style``.
            FileNotFoundError: If the split file is missing.
        """
        if answer_style not in ANSWER_STYLES:
            raise ValueError(f"answer_style must be one of {ANSWER_STYLES}")
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"reasoning split not found: {path}")

        self.path = path
        self.answer_style = answer_style
        self.ctx_len = ctx_len
        self.device = device
        self.pad_id, self.eos_id, self.bos_id = pad_id, eos_id, bos_id
        self._rng = random.Random(seed)
        self.epoch = 0

        key = "answer_direct" if answer_style == "direct" else "answer_rationale"
        self.rows, self.raw, self.n_skipped = [], [], 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                p_ids = sp.encode(rec["prompt"])
                a_ids = sp.encode(rec[key])
                full = [bos_id] + p_ids + a_ids + [eos_id]
                if len(full) > ctx_len:
                    self.n_skipped += 1
                    continue
                # y[i] predicts full[i+1]. The prompt occupies
                # full[1 : 1+len(p_ids)], so y[0 : len(p_ids)] are exactly the
                # prompt predictions and scoring starts at index len(p_ids).
                self.rows.append((full, len(p_ids)))
                self.raw.append(rec)

        if not self.rows:
            raise ValueError(f"{path} produced no usable examples")
        self._order = list(range(len(self.rows)))

    def __len__(self) -> int:
        """Number of retained examples."""
        return len(self.rows)

    def token_stats(self) -> dict:
        """Sequence-length statistics, for the finetuning log header."""
        lens = [len(f) for f, _ in self.rows]
        scored = [len(f) - 1 - n for f, n in self.rows]   # answer tokens + <eos>
        return {
            "examples": len(lens), "skipped": self.n_skipped,
            "mean_len": round(sum(lens) / len(lens), 2), "max_len": max(lens),
            "mean_scored_tokens": round(sum(scored) / len(scored), 2),
            "total_scored_tokens": sum(scored),
        }

    def _collate(self, idxs: list):
        """Build one padded batch from example indices.

        Args:
            idxs: Row indices to include.

        Returns:
            ``(x, y)`` long tensors of shape ``(B, T - 1)`` where ``T`` is the
            longest sequence in the batch. ``y`` holds ``IGNORE_INDEX`` on every
            prompt and pad position.
        """
        chunk = [self.rows[i] for i in idxs]
        width = max(len(f) for f, _ in chunk) - 1        # after the causal shift
        x = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
        y = torch.full((len(chunk), width), IGNORE_INDEX, dtype=torch.long)
        for r, (full, n_prompt) in enumerate(chunk):
            xi, yi = full[:-1], full[1:]
            x[r, :len(xi)] = torch.tensor(xi, dtype=torch.long)
            # Score only what comes after the prompt; pads stay IGNORE_INDEX.
            y[r, n_prompt:len(yi)] = torch.tensor(yi[n_prompt:], dtype=torch.long)
        return x.to(self.device), y.to(self.device)

    def iter_epoch(self, batch_size: int, shuffle: bool = True):
        """Yield every example exactly once, in batches.

        Args:
            batch_size: Examples per batch. The final batch may be smaller.
            shuffle: Reshuffle the order first. Validation passes False so the
                same examples land in the same batches across checkpoints.

        Yields:
            ``(x, y)`` batches.
        """
        order = list(self._order)
        if shuffle:
            self._rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            yield self._collate(order[i:i + batch_size])

    def iter_steps(self, batch_size: int):
        """Yield batches forever, reshuffling between epochs.

        The finetuning loop is step-based like pretraining's, so it needs a
        stream rather than a sequence of epochs. ``self.epoch`` counts completed
        passes and is logged.

        Yields:
            ``(x, y)`` batches.
        """
        while True:
            for batch in self.iter_epoch(batch_size, shuffle=True):
                yield batch
            self.epoch += 1

    def steps_per_epoch(self, batch_size: int, grad_accum: int = 1) -> int:
        """Optimisation steps in one pass over the data."""
        batches = (len(self.rows) + batch_size - 1) // batch_size
        return batches // max(grad_accum, 1)
