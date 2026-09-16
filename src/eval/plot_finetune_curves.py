"""Finetuning curves: training loss, validation loss, and validation exact match.

The brief asks each phase's report to carry its loss curves. Phase 2's come from
``plot_curves.py``, which reads a pretraining log; the finetuning logs have a
different shape -- an ``event`` header, ``val_em`` alongside ``val_loss``, no
``tokens_seen`` -- so they get their own plotter rather than a widened one.

The third panel is the reason this module exists. Validation loss and validation
exact match both improve monotonically to the end of every run (report §5.1),
which is what makes the divergence from *test* exact match so easy to miss: the
curves a practitioner would actually watch show no problem at all. Plotting the
two together, with the test result marked, puts that gap on one axis.

Runs whose log holds more than one training session -- a re-run appends -- are
split at each ``event: start`` and only the last session is drawn, so a repeated
cell does not produce a curve that jumps backwards.

Usage:
    python -m src.eval.plot_finetune_curves \
        --logs hindi/logs/finetune_direct.jsonl hindi/logs/finetune_gentle.jsonl \
        --labels "3 epochs @ 1e-4" "1 epoch @ 3e-5" \
        --language hindi --out report/figures/finetune_curves_hindi.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval import provenance

COLORS = ["#C2410C", "#0F766E", "#5B21B6", "#B45309"]


def load_log(path) -> tuple:
    """Read one finetuning log, keeping only its final training session.

    Args:
        path: A ``finetune_*.jsonl`` written by ``src.train.finetune``.

    Returns:
        ``(train_records, eval_records)``, each a list of dicts ordered by step.

    Raises:
        ValueError: If the file holds no training records at all.
    """
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    starts = [i for i, r in enumerate(records) if r.get("event") == "start"]
    if starts:
        records = records[starts[-1]:]
    train = [r for r in records if "train_loss" in r]
    evals = [r for r in records if "val_em" in r]
    if not train:
        raise ValueError(f"{path} contains no training records")
    return train, evals


def plot(logs: list, labels: list, language: str, out_path, stamp: str):
    """Draw the three-panel figure for one language.

    Args:
        logs: Paths to finetuning logs, one per run being compared.
        labels: Human-readable name per run, same order.
        language: For the title.
        out_path: PNG destination.
        stamp: Provenance line printed under the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for i, (path, label) in enumerate(zip(logs, labels)):
        train, evals = load_log(path)
        c = COLORS[i % len(COLORS)]
        axes[0].plot([r["step"] for r in train], [r["train_loss"] for r in train],
                     color=c, lw=1.1, alpha=0.85, label=label)
        if evals:
            axes[1].plot([r["step"] for r in evals], [r["val_loss"] for r in evals],
                         color=c, lw=1.6, marker="o", ms=3.5, label=label)
            axes[2].plot([r["step"] for r in evals], [r["val_em"] for r in evals],
                         color=c, lw=1.6, marker="o", ms=3.5, label=label)

    for ax, title, ylab in (
            (axes[0], "Training loss (answer tokens only)", "cross-entropy"),
            (axes[1], "Validation loss", "cross-entropy"),
            (axes[2], "Validation exact match", "exact match")):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("optimisation step")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[2].set_ylim(0, 1.0)

    fig.suptitle(f"{language.title()} — reasoning finetuning. Validation exact "
                 f"match rises throughout, on entities the model trained on; "
                 f"§5.1 compares it with test.", fontsize=10)
    fig.text(0.01, 0.005, stamp, fontsize=6, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    """Parse the CLI and write one language's finetuning-curve figure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--language", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.logs) != len(args.labels):
        ap.error("--logs and --labels must have the same length")

    stamp = provenance.stamp_text(provenance.build(
        "src.eval.plot_finetune_curves",
        inputs={Path(p).name: p for p in args.logs}))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(args.logs, args.labels, args.language, out, stamp)
    for p, l in zip(args.logs, args.labels):
        train, evals = load_log(p)
        print(f"  {l:20} {len(train):>3} train pts, {len(evals):>3} evals, "
              f"final train loss {train[-1]['train_loss']:.4f}, "
              f"best val EM {max((r['val_em'] for r in evals), default=0):.4f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
