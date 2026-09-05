"""Baseline vs no-positional-embedding ablation: matched-step comparison.

The Phase 2 bonus asks what happens when the positional embedding is removed.
Answering it needs a *matched* comparison, not two numbers: the ablation runs
2,000 of the baseline's 14,329 steps, so every claim here is made at equal step
count, equal tokens seen, equal learning rate, equal seed, and equal batch
composition. The only difference between the two runs is
``pos_encoding: learned`` versus ``none``.

The learning rate is the subtle one. Cosine decay over 2,000 steps has annealed
to its floor by the end, where the baseline at step 2,000 is still at 97% of
peak -- a 10x difference that would show up as an ablation effect if left
uncontrolled. ``schedule.total_steps`` in the ablation config pins the schedule
to the baseline's horizon so the two learning-rate curves coincide over the
compared range; this module asserts that they actually did, rather than assuming
the config was honoured.

What the comparison should *not* be read as. Removing positional embeddings does
not make the model order-blind. Under a causal mask, the query at position ``i``
attends over exactly ``i + 1`` keys, so the count of visible tokens is itself a
weak monotonic position signal that no amount of ablation removes. The measured
gap is therefore "learned absolute positions versus whatever the causal mask
leaks", not "positions versus none".

Usage:
    python -m src.eval.compare_ablation \
        --baseline-log hindi/logs/train.jsonl \
        --ablation-log hindi/logs/train_nopos.jsonl \
        --baseline-attn hindi/eval/attention_stats.json \
        --ablation-attn hindi/eval/attention_stats_nopos.json \
        --language hindi --outdir report/figures \
        --out report/figures/ablation_nopos.json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval import provenance
from src.eval.plot_curves import load_log

BASE_C, ABL_C = "#C2410C", "#0F766E"


def align_steps(base: list, abl: list) -> tuple:
    """Restrict both runs to the step range they have in common.

    Args:
        base: Records from the baseline run.
        abl: Records from the ablation run.

    Returns:
        (base_trimmed, abl_trimmed), both covering steps <= the ablation's last.
    """
    if not abl:
        return [], []
    last = abl[-1]["step"]
    return [r for r in base if r["step"] <= last], abl


def matched_pairs(base: list, abl: list, key: str) -> list:
    """Pair records that share a step, so no comparison is made across steps.

    Args:
        base: Baseline records.
        abl: Ablation records.
        key: Field to extract, e.g. ``"val_loss"``.

    Returns:
        A list of (step, baseline_value, ablation_value) at steps present in both.
    """
    b = {r["step"]: r[key] for r in base if key in r}
    a = {r["step"]: r[key] for r in abl if key in r}
    return [(s, b[s], a[s]) for s in sorted(set(b) & set(a))]


def check_lr_aligned(base: list, abl: list, tol: float = 0.02) -> dict:
    """Verify the two runs really did follow the same learning-rate schedule.

    The ablation pins ``schedule.total_steps`` to the baseline's horizon so the
    curves coincide. If that had been missed, the ablation would train at a tenth
    of the baseline's learning rate by step 2,000 and the comparison would be
    meaningless. Checking is cheap; assuming is not.

    Args:
        base: Baseline train records.
        abl: Ablation train records.
        tol: Maximum tolerated relative difference at any shared step.

    Returns:
        ``{"aligned", "max_relative_diff", "n_compared", "worst_step"}``.
    """
    pairs = matched_pairs(base, abl, "lr")
    worst, worst_step = 0.0, None
    for s, b, a in pairs:
        if b > 0:
            d = abs(a - b) / b
            if d > worst:
                worst, worst_step = d, s
    return {"aligned": bool(pairs) and worst <= tol,
            "max_relative_diff": worst, "n_compared": len(pairs),
            "worst_step": worst_step}


def loss_summary(base_tr: list, abl_tr: list,
                 base_val: list, abl_val: list) -> dict:
    """Reduce the two runs to the numbers the report quotes.

    Returns:
        Final matched-step losses, the gap in nats and in perplexity, and the
        full matched validation series.
    """
    val_pairs = matched_pairs(base_val, abl_val, "val_loss")
    tr_pairs = matched_pairs(base_tr, abl_tr, "train_loss")
    out = {
        "n_val_points": len(val_pairs),
        "n_train_points": len(tr_pairs),
        "val_series": [{"step": s, "baseline": b, "ablation": a,
                        "gap_nats": a - b,
                        "baseline_ppl": math.exp(b), "ablation_ppl": math.exp(a)}
                       for s, b, a in val_pairs],
    }
    if val_pairs:
        s, b, a = val_pairs[-1]
        out.update(final_step=s, final_baseline_loss=b, final_ablation_loss=a,
                   final_gap_nats=a - b,
                   final_baseline_ppl=math.exp(b), final_ablation_ppl=math.exp(a),
                   final_ppl_ratio=math.exp(a) / math.exp(b),
                   # ln(vocab) is the loss of a model that has learned nothing;
                   # it turns "the gap is 0.4 nats" into "the gap is 7% of the
                   # distance from ignorance to the baseline".
                   uniform_loss=math.log(8000),
                   gap_as_share_of_learned=(a - b) / (math.log(8000) - b))
    return out


def attention_comparison(base_attn: dict, abl_attn: dict) -> dict:
    """Contrast per-head attention structure between the two models.

    The headline prediction is specific: a previous-token head cannot form
    without a position signal, because "previous" is exactly what the model can
    no longer identify. Per layer this reports the strongest previous-token head
    in each run, alongside mean distance and entropy.

    Args:
        base_attn: Parsed ``attention_stats.json`` for the baseline.
        abl_attn: The same for the ablation.

    Returns:
        Per-layer contrasts plus whole-model aggregates and the head counts.
    """
    def _layers(d):
        return d["layers"]

    rows = []
    for li, (bl, al) in enumerate(zip(_layers(base_attn), _layers(abl_attn))):
        bp = max(bl, key=lambda h: h["prev_token_mass"])
        ap = max(al, key=lambda h: h["prev_token_mass"])
        mean = lambda L, k: sum(h[k] for h in L) / len(L)
        rows.append({
            "layer": li,
            "baseline_max_prev_mass": bp["prev_token_mass"],
            "baseline_max_prev_head": bp["head"],
            "ablation_max_prev_mass": ap["prev_token_mass"],
            "ablation_max_prev_head": ap["head"],
            "baseline_mean_distance": mean(bl, "mean_distance"),
            "ablation_mean_distance": mean(al, "mean_distance"),
            "baseline_mean_entropy": mean(bl, "entropy_normalised"),
            "ablation_mean_entropy": mean(al, "entropy_normalised"),
        })

    flat = lambda d: [h for L in d["layers"] for h in L]
    agg = {}
    for name, d in (("baseline", base_attn), ("ablation", abl_attn)):
        hs = flat(d)
        agg[name] = {
            "max_prev_token_mass": max(h["prev_token_mass"] for h in hs),
            "mean_distance": sum(h["mean_distance"] for h in hs) / len(hs),
            "mean_entropy": sum(h["entropy_normalised"] for h in hs) / len(hs),
            "n_prev_token_heads": sum(h["prev_token_mass"] > 0.25 for h in hs),
            "n_local_heads": sum(h["mean_distance"] < 10 for h in hs),
            "classification_counts": d.get("classification_counts", {}),
        }
    return {"per_layer": rows, "aggregate": agg}


def plot_loss_comparison(base_tr, abl_tr, base_val, abl_val,
                         out_path: Path, language: str, stamp: str = None):
    """Training and validation loss, baseline against ablation, at matched steps."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (b, a, key, title) in zip(axes, [
            (base_tr, abl_tr, "train_loss", "Training loss"),
            (base_val, abl_val, "val_loss", "Validation loss")]):
        ax.plot([r["step"] for r in b], [r[key] for r in b],
                color=BASE_C, linewidth=1.8, label="Learned positions (baseline)")
        ax.plot([r["step"] for r in a], [r[key] for r in a],
                color=ABL_C, linewidth=1.8, linestyle="--",
                label="No positional embedding")
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-entropy loss (nats/token)")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(frameon=False)

    axes[1].axhline(math.log(8000), color="#6B7280", linestyle=":", linewidth=1)
    axes[1].annotate("ln(8000) = 8.99 — a model that has learned nothing",
                     xy=(0.02, 0.93), xycoords="axes fraction",
                     fontsize=8, color="#6B7280", va="top")

    # The whole point of the figure is the size of the final gap, and against a
    # y-axis stretched to ln(vocab) the two curves are visually indistinguishable
    # at the end. State the number rather than leaving it to be squinted at.
    vp = matched_pairs(base_val, abl_val, "val_loss")
    if vp:
        step, b, a = vp[-1]
        axes[1].annotate(
            f"final gap {a - b:+.4f} nats\n"
            f"ppl {math.exp(b):.2f} -> {math.exp(a):.2f} ({math.exp(a) / math.exp(b):.2f}x)",
            xy=(step, a), xytext=(-150, 58), textcoords="offset points",
            fontsize=9, color="#0F766E",
            arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1))

    fig.suptitle(f"Effect of removing positional embeddings — {language}, "
                 f"matched steps and learning-rate schedule", fontsize=13)
    fig.tight_layout()
    if stamp:
        fig.text(0.5, -0.03, stamp, ha="center", va="top",
                 fontsize=7, color="#6B7280")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prev_token_by_layer(cmp: dict, out_path: Path, language: str,
                             stamp: str = None):
    """Strongest previous-token head per layer, baseline against ablation."""
    rows = cmp["per_layer"]
    x = [r["layer"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [r["baseline_max_prev_mass"] for r in rows],
           width=w, color=BASE_C, label="Learned positions (baseline)")
    ax.bar([i + w / 2 for i in x], [r["ablation_max_prev_mass"] for r in rows],
           width=w, color=ABL_C, label="No positional embedding")
    ax.axhline(0.25, color="#6B7280", linestyle="--", linewidth=1,
               label="Threshold for a \"previous-token head\"")
    ax.set_title("Strongest previous-token head in each layer\n"
                 "(attention mass on position i-1, averaged over queries)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Previous-token attention mass")
    ax.set_xticks(x)
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    ax.legend(frameon=False)
    fig.suptitle(f"{language}: can a previous-token head form without positions?",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    if stamp:
        fig.text(0.5, -0.04, stamp, ha="center", va="top",
                 fontsize=7, color="#6B7280")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def markdown_report(res: dict, language: str) -> str:
    """Render the comparison as the tables the report embeds."""
    L = [f"### No-positional-embedding ablation — {language}", ""]
    lr = res["lr_alignment"]
    L += [f"Learning-rate schedules {'match' if lr['aligned'] else 'DO NOT MATCH'} "
          f"across {lr['n_compared']} shared steps "
          f"(max relative difference {lr['max_relative_diff']:.4f}).", ""]

    s = res["loss"]
    if "final_step" in s:
        L += [f"| At step {s['final_step']:,} | Learned positions | No positions |",
              "|:--|--:|--:|",
              f"| Validation loss (nats) | {s['final_baseline_loss']:.4f} | "
              f"{s['final_ablation_loss']:.4f} |",
              f"| Validation perplexity | {s['final_baseline_ppl']:.2f} | "
              f"{s['final_ablation_ppl']:.2f} |",
              f"| Gap | — | +{s['final_gap_nats']:.4f} nats "
              f"({s['final_ppl_ratio']:.2f}x perplexity) |", ""]

    a = res.get("attention")
    if a:
        b, ab = a["aggregate"]["baseline"], a["aggregate"]["ablation"]
        L += ["| Attention | Learned positions | No positions |", "|:--|--:|--:|",
              f"| Strongest previous-token head | {b['max_prev_token_mass']:.2f} | "
              f"{ab['max_prev_token_mass']:.2f} |",
              f"| Heads with prev-token mass > 0.25 | {b['n_prev_token_heads']} | "
              f"{ab['n_prev_token_heads']} |",
              f"| Heads with mean distance < 10 | {b['n_local_heads']} | "
              f"{ab['n_local_heads']} |",
              f"| Mean attention distance | {b['mean_distance']:.1f} | "
              f"{ab['mean_distance']:.1f} |",
              f"| Mean normalised entropy | {b['mean_entropy']:.3f} | "
              f"{ab['mean_entropy']:.3f} |", ""]
    return "\n".join(L)


def main():
    """Compare a baseline run against its no-positional-embedding ablation.

    Aligns the two runs to their shared step range, verifies the learning-rate
    schedules coincide, summarises the loss gap, optionally contrasts attention
    structure, and writes figures, a JSON and a Markdown table.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-log", required=True)
    ap.add_argument("--ablation-log", required=True)
    ap.add_argument("--baseline-attn", default=None)
    ap.add_argument("--ablation-attn", default=None)
    ap.add_argument("--language", required=True)
    ap.add_argument("--outdir", default="report/figures")
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    base_tr, base_val = load_log(args.baseline_log)
    abl_tr, abl_val = load_log(args.ablation_log)
    base_tr, abl_tr = align_steps(base_tr, abl_tr)
    base_val, abl_val = align_steps(base_val, abl_val)

    sources = {"baseline_log": args.baseline_log, "ablation_log": args.ablation_log}
    if args.baseline_attn and args.ablation_attn:
        sources["baseline_attn"] = args.baseline_attn
        sources["ablation_attn"] = args.ablation_attn

    prov = provenance.build("src.eval.compare_ablation", sources,
                            {"matplotlib": matplotlib.__version__})
    stamp = provenance.stamp_text(prov)

    res = {"provenance": prov, "language": args.language,
           "lr_alignment": check_lr_aligned(base_tr, abl_tr),
           "loss": loss_summary(base_tr, abl_tr, base_val, abl_val)}

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    plot_loss_comparison(base_tr, abl_tr, base_val, abl_val,
                         out / f"{args.language}_ablation_nopos_loss.png",
                         args.language, stamp)

    if args.baseline_attn and args.ablation_attn:
        b = json.loads(Path(args.baseline_attn).read_text())
        a = json.loads(Path(args.ablation_attn).read_text())
        res["attention"] = attention_comparison(b, a)
        plot_prev_token_by_layer(
            res["attention"], out / f"{args.language}_ablation_nopos_prevtoken.png",
            args.language, stamp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    md = markdown_report(res, args.language)
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
    print(md)

    lr = res["lr_alignment"]
    if not lr["aligned"]:
        print(f"\nWARNING: learning-rate schedules diverge by "
              f"{lr['max_relative_diff']:.3f} at step {lr['worst_step']} — the "
              f"comparison is confounded. Check schedule.total_steps.")


if __name__ == "__main__":
    main()
