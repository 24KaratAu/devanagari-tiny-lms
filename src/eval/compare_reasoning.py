"""Consolidate every reasoning evaluation into the final comparison tables.

``reasoning_eval`` scores one checkpoint under one setup and knows nothing about
the others. This module reads all of those JSON results -- each of which carries
its own ``setup`` block, so no labelling is passed on the command line and
mislabelling a run is not possible -- and produces the three things the brief's
Phase 3 deliverables ask for: pretrained versus finetuned accuracy, a comparison
between Model H and Model L, and an error analysis.

Three cuts are reported, because a single pooled exact match hides all of them:

  * **by reasoning depth** (0, 1, 2 transitive hops). A model that has learned
    the answer format but not the relation scores at chance on every depth; a
    model that has learned one-hop comparison but cannot chain shows a cliff
    between hop 1 and hop 2. Those are different failures and the pooled number
    calls them the same.
  * **by premise form.** Families whose premises state quantities require
    reading numerals; families that state inequalities directly do not. This is
    where the tokenizers differ -- Nepali learned 158 Devanagari-numeral pieces
    to Hindi's 42 -- so it is the cut where a tokenizer effect would surface.
  * **by phrasing.** A quarter of the test set uses surface realisations held
    out of training entirely. The gap between seen and held-out phrasing
    separates "learned the relation" from "learned the template".

The format rate is reported alongside accuracy throughout. A zero-shot
pretrained model usually fails because it never emits an answer at all, which is
a different thing from answering wrongly, and reporting only exact match would
present the two identically.

Usage:
    python -m src.eval.compare_reasoning \
        --inputs hindi/eval/reasoning_*.json nepali/eval/reasoning_*.json \
        --out report/figures/reasoning_comparison.json \
        --md  report/figures/reasoning_comparison.md \
        --figure report/figures/reasoning_accuracy.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval import provenance

MODE_ORDER = ["zeroshot", "fewshot", "finetuned"]
MODE_LABEL = {"zeroshot": "pretrained, zero-shot",
              "fewshot": "pretrained, few-shot",
              "finetuned": "finetuned"}
LANG_LABEL = {"hindi": "Model H (Hindi)", "nepali": "Model L (Nepali)"}
VALUE_FAMILIES = ("val2_which", "val2_relation", "val3_extreme", "val4_rank")
COLORS = {"hindi": "#C2410C", "nepali": "#0F766E"}


def load_results(paths: list) -> list:
    """Read result JSONs and key them by their own recorded setup.

    Args:
        paths: Files written by ``src.eval.reasoning_eval``.

    Returns:
        A list of ``(setup, result)`` pairs sorted by language, style, then
        mode, so table row order is deterministic.

    Raises:
        KeyError: If a file predates the ``setup`` block and cannot be labelled.
    """
    out = []
    for p in paths:
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        if "setup" not in r:
            raise KeyError(f"{p} has no setup block; regenerate it with the "
                           f"current src.eval.reasoning_eval")
        r["_path"] = str(p)
        out.append((r["setup"], r))
    return sorted(out, key=lambda t: (t[0]["language"], t[0]["answer_style"],
                                      MODE_ORDER.index(t[0]["mode"])))


def premise_split(result: dict) -> dict:
    """Accuracy on quantity-stating families against inequality-stating ones.

    Args:
        result: One ``reasoning_eval`` result.

    Returns:
        ``{"value_premises", "relation_premises"}`` mean accuracies, or None
        where a family group is absent.
    """
    fam = result.get("by_family", {})
    val = [v for k, v in fam.items() if k in VALUE_FAMILIES]
    rel = [v for k, v in fam.items() if k not in VALUE_FAMILIES]
    return {"value_premises": round(sum(val) / len(val), 4) if val else None,
            "relation_premises": round(sum(rel) / len(rel), 4) if rel else None}


def headline_table(rows: list) -> str:
    """The pretrained-vs-finetuned table, one line per run."""
    out = ["| Model | Answer style | Setup | Exact match | Format rate | "
           "0 hops | 1 hop | 2 hops |",
           "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for setup, r in rows:
        h = r.get("by_hops", {})
        out.append(
            f"| {LANG_LABEL[setup['language']]} | {setup['answer_style']} | "
            f"{MODE_LABEL[setup['mode']]}"
            + (f" (k={setup['shots']})" if setup["mode"] == "fewshot" else "")
            + f" | **{r['exact_match']:.4f}** | {r['format_rate']:.4f} | "
            + " | ".join(f"{h.get(str(k), float('nan')):.4f}" for k in (0, 1, 2))
            + " |")
    return "\n".join(out)


def generalisation_table(rows: list) -> str:
    """Seen versus held-out phrasing, and value versus relation premises."""
    out = ["| Model | Style | Setup | Seen phrasing | Held-out phrasing | "
           "Gap | Value premises | Relation premises |",
           "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for setup, r in rows:
        ph = r.get("by_phrasing", {})
        seen, held = ph.get("seen"), ph.get("held_out")
        gap = (f"{seen - held:+.4f}" if seen is not None and held is not None
               else "-")
        ps = premise_split(r)
        def f(x):
            """Format a rate, or a dash when the run did not produce one."""
            # bool is an int subclass, hence the explicit exclusion; a rate that
            # rounds to an exact 0 or 1 arrives as an int and must still print.
            return (f"{x:.4f}" if isinstance(x, (int, float))
                    and not isinstance(x, bool) else "-")
        out.append(f"| {LANG_LABEL[setup['language']]} | "
                   f"{setup['answer_style']} | {MODE_LABEL[setup['mode']]} | "
                   f"{f(seen)} | {f(held)} | {gap} | "
                   f"{f(ps['value_premises'])} | "
                   f"{f(ps['relation_premises'])} |")
    return "\n".join(out)


def family_table(rows: list) -> str:
    """Per-family accuracy, finetuned runs only -- the error-analysis grid."""
    ft = [(s, r) for s, r in rows if s["mode"] == "finetuned"]
    if not ft:
        return "_No finetuned runs supplied._"
    fams = sorted({k for _, r in ft for k in r.get("by_family", {})})
    head = "| Family | " + " | ".join(
        f"{LANG_LABEL[s['language']].split()[1]} {s['answer_style']}"
        for s, _ in ft) + " |"
    out = [head, "| :--- |" + " ---: |" * len(ft)]
    for fam in fams:
        out.append(f"| `{fam}` | " + " | ".join(
            f"{r['by_family'].get(fam, float('nan')):.4f}" for _, r in ft)
            + " |")
    return "\n".join(out)


def confusion_section(rows: list) -> str:
    """The most frequent wrong answers, per finetuned run."""
    parts = []
    for setup, r in rows:
        if setup["mode"] != "finetuned":
            continue
        parts.append(f"**{LANG_LABEL[setup['language']]}, "
                     f"{setup['answer_style']} answers** - most frequent "
                     f"errors (gold -> predicted):\n")
        for pair, n in r.get("confusions", [])[:8]:
            parts.append(f"- `{pair}` x {n}")
        parts.append("")
    return "\n".join(parts) if parts else "_No finetuned runs supplied._"


def plot_accuracy(rows: list, path: Path):
    """Grouped bars: exact match by reasoning depth, per model and setup.

    Args:
        rows: ``(setup, result)`` pairs.
        path: PNG destination.
    """
    runs = [(s, r) for s, r in rows if s["answer_style"] == "direct"] or rows
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, lang in zip(axes, ("hindi", "nepali")):
        sub = [(s, r) for s, r in runs if s["language"] == lang]
        if not sub:
            ax.set_axis_off()
            ax.set_title(f"{LANG_LABEL.get(lang, lang)} - no runs")
            continue
        depths = ["0", "1", "2"]
        width = 0.8 / len(sub)
        # One hue per language, shade increasing with how much help the model
        # got: zero-shot palest, finetuned solid. Two setups at the same alpha
        # are indistinguishable in the legend, so the ramp spans the runs shown.
        shades = ([0.35, 0.65, 1.0] if len(sub) == 3
                  else [0.4 + 0.6 * i / max(len(sub) - 1, 1)
                        for i in range(len(sub))])
        for i, (s, r) in enumerate(sub):
            vals = [r.get("by_hops", {}).get(d, 0.0) for d in depths]
            xs = [j + i * width for j in range(len(depths))]
            ax.bar(xs, vals, width * 0.92, label=MODE_LABEL[s["mode"]],
                   alpha=shades[i], color=COLORS[lang],
                   edgecolor="white", linewidth=0.6)
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(depths))])
        ax.set_xticklabels([f"{d} hop{'s' if d != '1' else ''}" for d in depths])
        ax.set_xlabel("Transitive steps the premises force")
        ax.set_ylabel("Exact match")
        ax.set_ylim(0, 1.05)
        ax.set_title(LANG_LABEL.get(lang, lang))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Reasoning exact match by depth: pretrained vs finetuned",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Parse the CLI, build every table, and write the JSON, Markdown and PNG."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()

    rows = load_results(args.inputs)
    print(f"loaded {len(rows)} evaluation results")
    for s, r in rows:
        print(f"  {s['language']:<7} {s['answer_style']:<9} {s['mode']:<10} "
              f"EM {r['exact_match']:.4f}  fmt {r['format_rate']:.4f}")

    md = "\n\n".join([
        "### Pretrained vs finetuned exact match",
        headline_table(rows),
        "### Generalisation: unseen phrasing, and premise form",
        generalisation_table(rows),
        "### Per-family accuracy (finetuned)",
        family_table(rows),
        "### Error analysis",
        confusion_section(rows),
    ]) + "\n"
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(md, encoding="utf-8")

    if args.figure:
        plot_accuracy(rows, Path(args.figure))
        print(f"  figure -> {args.figure}")

    payload = {
        "runs": [{"setup": s, "exact_match": r["exact_match"],
                  "format_rate": r["format_rate"], "by_hops": r.get("by_hops"),
                  "by_family": r.get("by_family"),
                  "by_phrasing": r.get("by_phrasing"),
                  "by_attribute": r.get("by_attribute"),
                  "premise_split": premise_split(r), "source": r["_path"]}
                 for s, r in rows],
        "provenance": provenance.build(
            "src.eval.compare_reasoning",
            inputs={Path(p).name: p for p in args.inputs}),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  tables -> {args.md}\n  json   -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
