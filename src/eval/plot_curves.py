"""Training and validation loss curves from the JSONL training logs.

Computation and plotting are kept in separate functions, per guideline 4 of the
brief: :func:`load_log` and :func:`curve_summary` read and reduce, the ``plot_*``
functions only draw. Every figure carries a title, axis labels, and a legend --
guideline 3 warns that plots missing them "may receive zero marks".

Usage:
    python -m src.eval.plot_curves \
        --logs hindi=hindi/logs/train.jsonl nepali=nepali/logs/train.jsonl \
        --outdir report/figures
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: no display on Colab or CI
import matplotlib.pyplot as plt

from src.eval import provenance

# One colour per model, reused across every figure so the legend reads the same
# way throughout the report.
COLORS = {"hindi": "#C2410C", "nepali": "#1D4ED8"}
LABELS = {"hindi": "Model H (Hindi)", "nepali": "Model L (Nepali)"}


def load_log(path) -> tuple[list, list]:
    """Split a training log into its train and validation records.

    Args:
        path: JSONL file written by the trainer, one record per logged step.

    Returns:
        (train_records, val_records), each ordered by step.
    """
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    train = sorted((r for r in rows if "train_loss" in r), key=lambda r: r["step"])
    val = sorted((r for r in rows if "val_loss" in r), key=lambda r: r["step"])
    return train, val


def curve_summary(train: list, val: list) -> dict:
    """Reduce a pair of curves to the figures quoted in the report.

    Returns:
        Final train loss, best validation loss and where it occurred,
        perplexity, mean throughput, peak memory, and total steps.
    """
    best = min(val, key=lambda r: r["val_loss"]) if val else None
    tps = [r["tokens_per_sec"] for r in train if "tokens_per_sec" in r]
    return {
        "steps": train[-1]["step"] if train else 0,
        "final_train_loss": train[-1]["train_loss"] if train else float("nan"),
        "best_val_loss": best["val_loss"] if best else float("nan"),
        "best_val_ppl": math.exp(best["val_loss"]) if best else float("nan"),
        "best_val_step": best["step"] if best else 0,
        "n_evals": len(val),
        "mean_tokens_per_sec": sum(tps) / len(tps) if tps else 0.0,
        "peak_mem_gib": max((r.get("peak_mem_gib", 0) for r in train), default=0),
        "tokens_seen": train[-1].get("tokens_seen", 0) if train else 0,
    }


def _smooth(values: list, window: int = 9) -> list:
    """Centred moving average, for a readable trend line over noisy per-step loss."""
    if window < 2 or len(values) < window:
        return values
    half = window // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def _stamp(fig, stamp: str = None):
    """Print the provenance line along the bottom of a figure.

    A figure that travels into a report loses its context; carrying the commit
    and the input digests in the image itself means the reader can always tell
    which run and which code produced it.
    """
    if stamp:
        fig.text(0.5, -0.045, stamp, ha="center", va="top",
                 fontsize=7, color="#6B7280")



def plot_loss_curves(logs: dict, out_path: Path, stamp: str = None):
    """Training and validation loss against step, both models on shared axes.

    Args:
        logs: {lang: (train_records, val_records)}.
        out_path: Destination PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for lang, (train, val) in logs.items():
        c, label = COLORS.get(lang, "#444"), LABELS.get(lang, lang)
        steps = [r["step"] for r in train]
        raw = [r["train_loss"] for r in train]
        axes[0].plot(steps, raw, color=c, alpha=0.22, linewidth=0.8)
        axes[0].plot(steps, _smooth(raw), color=c, linewidth=1.8,
                     label=f"{label} — training")
        axes[1].plot([r["step"] for r in val], [r["val_loss"] for r in val],
                     color=c, marker="o", markersize=3.2, linewidth=1.8,
                     label=f"{label} — validation")

    axes[0].set_title("Training loss\n(faint: per-logged-step; bold: 9-point moving average)")
    axes[1].set_title("Validation loss\n(every 500 steps on a fixed 2M-token prefix;\n"
                      "single-source and easier than the full split — see report)")
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-entropy loss (nats/token)")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(frameon=False)
    fig.suptitle("Pretraining loss — Model H (Hindi) vs Model L (Nepali), "
                 "25.4M parameters each", fontsize=13, y=1.00)
    fig.tight_layout()
    _stamp(fig, stamp)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_perplexity(logs: dict, out_path: Path, stamp: str = None,
                    intrinsic: dict = None):
    """Validation perplexity on a log scale, against the true full-split value.

    The curve is *not* the model's perplexity on the validation set. During
    training the loop scored a fixed 2,000,000-token prefix of the split, and the
    Phase 1 splits are written grouped by source, so that prefix covers
    Wikibooks, Wikiquote, the news crawl and part of the manual web crawl and
    never reaches ``sangraha`` -- 71.6% of Hindi validation documents and 77.5%
    of Nepali's. The slice is therefore both unrepresentative and easier than the
    split as a whole, and the curve sits about 30% below the truth.

    It is still the right thing to plot: it is a *consistent* measure across
    steps, which is all a training curve needs to be, and it is what checkpoint
    selection actually used. Plotting it against the full-split value measured
    afterwards is more honest than quietly substituting one for the other, so
    when ``intrinsic`` is supplied the true value is drawn as a marker at the
    final step with the gap annotated.

    Args:
        logs: {lang: (train_records, val_records)}.
        out_path: Destination PNG.
        stamp: Provenance line.
        intrinsic: Optional {lang: parsed intrinsic.json} for the full-split
            markers.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for lang, (_, val) in logs.items():
        c = COLORS.get(lang, "#444")
        ax.plot([r["step"] for r in val], [math.exp(r["val_loss"]) for r in val],
                color=c, marker="o", markersize=3.2, linewidth=1.8,
                label=f"{LABELS.get(lang, lang)} — 2M-token slice (used in training)")

        if intrinsic and lang in intrinsic:
            m = intrinsic[lang]["splits"]
            step = val[-1]["step"]
            full = m["val"]["perplexity"]
            ax.plot([step], [full], color=c, marker="*", markersize=17,
                    linestyle="none", markeredgecolor="black",
                    markeredgewidth=0.6,
                    label=f"{LABELS.get(lang, lang)} — full validation split")
            slice_ppl = math.exp(val[-1]["val_loss"])
            ax.annotate(f"+{100 * (full / slice_ppl - 1):.0f}%",
                        xy=(step, full), xytext=(-46, 2),
                        textcoords="offset points", fontsize=9, color=c)

    ax.axhline(8000, color="#6B7280", linestyle="--", linewidth=1,
               label="Uniform over 8,000-token vocabulary")
    ax.set_yscale("log")
    ax.set_title("Validation perplexity during pretraining\n"
                 "(curve: fixed 2M-token prefix — one-source, easier than the "
                 "split; star: true full-split value)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Perplexity (log scale)")
    ax.grid(alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    _stamp(fig, stamp)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lr_and_throughput(logs: dict, out_path: Path, stamp: str = None):
    """Learning-rate schedule and throughput, to evidence the run's stability."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for lang, (train, _) in logs.items():
        c, label = COLORS.get(lang, "#444"), LABELS.get(lang, lang)
        steps = [r["step"] for r in train]
        axes[0].plot(steps, [r["lr"] for r in train], color=c, linewidth=1.8, label=label)
        axes[1].plot(steps, _smooth([r["tokens_per_sec"] for r in train]),
                     color=c, linewidth=1.5, label=label)
        axes[2].plot(steps, _smooth([r["grad_norm"] for r in train]),
                     color=c, linewidth=1.5, label=label)

    axes[0].set_title("Learning-rate schedule\n(500-step linear warmup, then cosine decay)")
    axes[0].set_ylabel("Learning rate")
    axes[1].set_title("Throughput\n(9-point moving average)")
    axes[1].set_ylabel("Tokens / second")
    axes[2].set_title("Gradient norm before clipping\n(clip threshold 1.0)")
    axes[2].set_ylabel("Gradient L2 norm")
    axes[2].axhline(1.0, color="#6B7280", linestyle="--", linewidth=1,
                    label="Clip threshold")
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Optimisation diagnostics", fontsize=13, y=1.02)
    fig.tight_layout()
    _stamp(fig, stamp)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)




def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True,
                    help="lang=path/to/train.jsonl pairs")
    ap.add_argument("--outdir", default="report/figures")
    ap.add_argument("--intrinsic", nargs="*", default=[],
                    help="lang=path/to/intrinsic.json, to mark the true "
                         "full-split perplexity on the curve")
    args = ap.parse_args()

    logs, summaries, sources = {}, {}, {}
    for spec in args.logs:
        lang, path = spec.split("=", 1)
        sources[lang] = path
        logs[lang] = load_log(path)
        summaries[lang] = curve_summary(*logs[lang])

    # Every input must be registered in `sources` before the provenance block
    # is built, or it is absent from the record the figures are stamped with.
    intrinsic = {}
    for spec in args.intrinsic:
        lang, path = spec.split("=", 1)
        intrinsic[lang] = json.loads(Path(path).read_text(encoding="utf-8"))
        sources[f"{lang}_intrinsic"] = path

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    prov = provenance.build("src.eval.plot_curves", sources,
                            {"matplotlib": matplotlib.__version__})
    stamp = provenance.stamp_text(prov)
    plot_loss_curves(logs, out / "loss_curves.png", stamp)
    plot_perplexity(logs, out / "val_perplexity.png", stamp, intrinsic or None)
    plot_lr_and_throughput(logs, out / "optimisation_diagnostics.png", stamp)

    # Record both numbers side by side: the slice the training loop used for
    # checkpoint selection, and the full-split value the report quotes.
    for lang, d in intrinsic.items():
        if lang in summaries:
            summaries[lang]["full_val_ppl"] = d["splits"]["val"]["perplexity"]
            summaries[lang]["full_test_ppl"] = d["splits"]["test"]["perplexity"]
            summaries[lang]["full_test_bpb"] = d["splits"]["test"]["bits_per_byte"]
            summaries[lang]["full_test_bits_per_char"] = d["splits"]["test"]["bits_per_char"]
            summaries[lang]["slice_vs_full_ppl_gap_pct"] = 100.0 * (
                summaries[lang]["full_val_ppl"] / summaries[lang]["best_val_ppl"] - 1)

    payload = {"provenance": prov, "models": summaries}
    (out / "training_summary.json").write_text(json.dumps(payload, indent=2))
    for lang, s in summaries.items():
        print(f"{LABELS.get(lang, lang)}")
        print(f"  {s['steps']:,} steps, {s['tokens_seen']:,} tokens, "
              f"{s['n_evals']} validation passes")
        print(f"  final train {s['final_train_loss']:.4f} | "
              f"best val {s['best_val_loss']:.4f} (ppl {s['best_val_ppl']:.2f}) "
              f"at step {s['best_val_step']:,}")
        print(f"  {s['mean_tokens_per_sec']:,.0f} tok/s mean, "
              f"{s['peak_mem_gib']:.1f} GiB peak")
    print(f"\nwrote 3 figures + training_summary.json to {out}/")


if __name__ == "__main__":
    main()
