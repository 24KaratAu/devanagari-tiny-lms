"""Pretrained vs finetuned attention, measured on comparative-reasoning prompts.

Phase 2 characterised each model's heads over held-out *corpus* windows. Phase 3
asks a different question: does finetuning on comparative reasoning change where
the heads look, and does it do so differently in the higher- and lower-resource
model. Both checkpoints are therefore run over the *same* reasoning prompts, and
every number reported is a difference between two runs that differ only in which
weights were loaded.

Why the prompts, not the corpus. A comparison over corpus windows would mostly
re-measure Phase 2. The interesting positions are the ones this task actually
creates: the second mention of an entity, where the model must bind it back to
its first mention, and the question tokens, where it must reach back to whichever
premise is relevant. Those positions do not exist in running text.

Two comparability rules are enforced rather than assumed:

  * **Same input.** One prompt list is tokenised once and reused for both
    checkpoints, so the batches are identical objects, not merely similar sets.
  * **No cross-phase distance comparison.** Mean attention distance grows with
    sequence length, and these prompts are ~37 tokens against Phase 2's 512.
    The distances here are compared pretrained-vs-finetuned only; the JSON
    records the prompt length distribution so the constraint is visible to a
    reader who might otherwise put the two tables side by side.

Usage:
    python -m src.eval.attention_reasoning \
        --pretrained hindi/checkpoints/checkpoint_best.pt \
        --finetuned  hindi/checkpoints_ft_direct/finetune_best.pt \
        --model-config hindi/configs/model.yaml \
        --tokenizer tokenizers/hindi/hindi_bpe_8k.model \
        --test data/hindi/reasoning/test.jsonl --language hindi \
        --outdir report/figures --out hindi/eval/attention_reasoning.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.eval import provenance
from src.eval.attention_analysis import (classify_head, find_devanagari_font,
                                         head_statistics)
from src.model.transformer_lm import TransformerLM, TransformerLMConfig
from src.train.checkpoint import load_checkpoint

PRE_C, FT_C = "#C2410C", "#0F766E"


def prompt_batches(sp, prompts: list, batch_size: int, device: str) -> list:
    """Tokenise prompts and group them into unpadded, fixed-width batches.

    Padding is avoided rather than masked. These models use learned absolute
    positions, so a pad on the left would shift every position index, and a pad
    on the right would enter the attention rows of no real token but would still
    occupy query positions whose statistics are meaningless. Grouping by exact
    token length sidesteps both.

    Args:
        sp: SentencePiece processor.
        prompts: Prompt strings.
        batch_size: Maximum prompts per batch.
        device: Device string.

    Returns:
        A list of ``(x, None)`` pairs, materialised (not a generator) so the
        identical batches can be replayed for both checkpoints.
    """
    buckets = defaultdict(list)
    for p in prompts:
        ids = sp.encode(p)
        buckets[len(ids)].append(ids)
    out = []
    for length in sorted(buckets):
        rows = buckets[length]
        for s in range(0, len(rows), batch_size):
            x = torch.tensor(rows[s:s + batch_size], dtype=torch.long,
                             device=device)
            out.append((x, None))
    return out


def load_model(path: str, model_config: str, device: str):
    """Instantiate the architecture and load one checkpoint's weights.

    Args:
        path: Checkpoint file.
        model_config: Architecture YAML; identical for both checkpoints, since
            finetuning changes weights and not shape.
        device: Device string.

    Returns:
        ``(model_in_eval_mode, step)``.
    """
    cfg = TransformerLMConfig.from_yaml(model_config)
    model = TransformerLM(cfg).to(device)
    meta = load_checkpoint(path, model, map_location="cpu", restore_rng=False)
    model.eval()
    return model, meta["step"]


def delta_table(pre: dict, ft: dict, ctx_len: int) -> dict:
    """Per-head and per-layer differences between the two runs.

    Args:
        pre: ``head_statistics`` output for the pretrained checkpoint.
        ft: The same for the finetuned checkpoint.
        ctx_len: Passed to ``classify_head`` for role labelling.

    Returns:
        Per-layer aggregates, the largest individual head movements, and a
        confusion table of head roles before against after.
    """
    keys = ("entropy_normalised", "mean_distance", "prev_token_mass",
            "first_position_mass")
    layers, moves, roles = [], [], Counter()
    for lp, lf in zip(pre["layers"], ft["layers"]):
        agg = {k: {"pretrained": float(np.mean([h[k] for h in lp])),
                   "finetuned": float(np.mean([h[k] for h in lf]))}
               for k in keys}
        for k in keys:
            agg[k]["delta"] = round(agg[k]["finetuned"] - agg[k]["pretrained"], 5)
            agg[k]["pretrained"] = round(agg[k]["pretrained"], 5)
            agg[k]["finetuned"] = round(agg[k]["finetuned"], 5)
        layers.append({"layer": lp[0]["layer"], **agg})
        for hp, hf in zip(lp, lf):
            rp, rf = classify_head(hp, ctx_len), classify_head(hf, ctx_len)
            roles[f"{rp} -> {rf}"] += 1
            moves.append({
                "layer": hp["layer"], "head": hp["head"],
                "role_pretrained": rp, "role_finetuned": rf,
                "d_distance": round(hf["mean_distance"] - hp["mean_distance"], 4),
                "d_entropy_norm": round(hf["entropy_normalised"]
                                        - hp["entropy_normalised"], 4),
                "d_prev_mass": round(hf["prev_token_mass"]
                                     - hp["prev_token_mass"], 4),
            })
    moves.sort(key=lambda m: -abs(m["d_distance"]))
    return {"per_layer": layers, "largest_distance_shifts": moves[:12],
            "role_transitions": dict(roles.most_common()),
            "roles_changed": sum(v for k, v in roles.items()
                                 if k.split(" -> ")[0] != k.split(" -> ")[1]),
            "heads_total": len(moves)}


@torch.no_grad()
def one_prompt_attention(model, sp, prompt: str, device: str):
    """Attention weights for a single prompt, for the heatmaps.

    Args:
        model: A loaded model in eval mode.
        sp: SentencePiece processor.
        prompt: The prompt string to visualise.
        device: Device string.

    Returns:
        ``(pieces, attentions)`` where ``attentions[layer]`` is ``(h, T, T)``.
    """
    ids = sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, attns = model(x, need_weights=True)
    pieces = [sp.id_to_piece(i) for i in ids]
    return pieces, [a[0].float().cpu().numpy() for a in attns]


def plot_comparison(pre_attn, ft_attn, pieces, layer: int, heads: list,
                    out_path: Path, language: str, stamp: str, font_ok: bool):
    """Heatmap grid: pretrained on the top row, finetuned below, same prompt.

    Args:
        pre_attn: ``(h, T, T)`` weights from the pretrained checkpoint.
        ft_attn: The same from the finetuned checkpoint.
        pieces: Token labels.
        layer: Which layer these came from, for the title.
        heads: Head indices to draw.
        out_path: PNG destination.
        language: For the title.
        stamp: Provenance line printed under the figure.
        font_ok: Whether a Devanagari-capable font was found; when False the
            tick labels are suppressed rather than drawn as tofu boxes.
    """
    n = len(heads)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7.6), squeeze=False)
    for col, h in enumerate(heads):
        for row, (attn, tag) in enumerate(((pre_attn, "pretrained"),
                                           (ft_attn, "finetuned"))):
            ax = axes[row][col]
            im = ax.imshow(attn[h], cmap="magma", vmin=0, vmax=1,
                           aspect="auto", interpolation="nearest")
            ax.set_title(f"layer {layer}, head {h} - {tag}", fontsize=9)
            ax.set_xlabel("Key position (attended to)", fontsize=8)
            ax.set_ylabel("Query position (attending from)", fontsize=8)
            if font_ok and len(pieces) <= 48:
                ax.set_xticks(range(len(pieces)))
                ax.set_xticklabels(pieces, rotation=90, fontsize=5)
                ax.set_yticks(range(len(pieces)))
                ax.set_yticklabels(pieces, fontsize=5)
            else:
                ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046,
                         label="Attention probability")
    fig.suptitle(f"{language.title()} - attention on a comparative-reasoning "
                 f"prompt, layer {layer}", fontsize=11)
    fig.text(0.01, 0.005, stamp, fontsize=6, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    """Parse the CLI, run both checkpoints over one prompt set, and compare."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--language", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--n-prompts", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--heads", type=int, nargs="+", default=[0, 1, 4, 7])
    ap.add_argument("--family", default="rel4_pairwise",
                    help="family the visualised prompt is drawn from; the "
                         "default is the deepest chain in the corpus")
    ap.add_argument("--font", default=None)
    ap.add_argument("--outdir", default="report/figures")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import sentencepiece as spm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    cfg = TransformerLMConfig.from_yaml(args.model_config)

    records = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    prompts = [r["prompt"] for r in records[:args.n_prompts]]
    batches = prompt_batches(sp, prompts, args.batch_size, device)
    lengths = [int(x.shape[1]) for x, _ in batches]
    print(f"{len(prompts):,} prompts -> {len(batches)} unpadded batches, "
          f"widths {min(lengths)}-{max(lengths)}")

    pre_model, pre_step = load_model(args.pretrained, args.model_config, device)
    ft_model, ft_step = load_model(args.finetuned, args.model_config, device)
    print(f"pretrained step {pre_step:,}  |  finetuned step {ft_step:,}")

    stats = {}
    for name, model in (("pretrained", pre_model), ("finetuned", ft_model)):
        stats[name] = head_statistics(model, None, cfg.ctx_len,
                                      args.batch_size, None, device,
                                      batches=batches)
        print(f"  {name}: {stats[name]['n_queries']:,} query positions")

    deltas = delta_table(stats["pretrained"], stats["finetuned"], cfg.ctx_len)

    # --- heatmaps: one early layer and one late layer -----------------------
    shown = next((r for r in records if r["family"] == args.family), records[0])
    pieces, pre_a = one_prompt_attention(pre_model, sp, shown["prompt"], device)
    _, ft_a = one_prompt_attention(ft_model, sp, shown["prompt"], device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # find_devanagari_font returns a *family name*, already registered with the
    # font manager -- not a path. Passing it back through FontProperties(fname=)
    # makes matplotlib try to open a file called "Lohit Devanagari". Phase 2's
    # attention_analysis.py assigns it directly; do the same.
    font = find_devanagari_font(args.font)
    if font:
        matplotlib.rcParams["font.family"] = font
    stamp = provenance.stamp_text(provenance.build(
        "src.eval.attention_reasoning",
        inputs={"pretrained": args.pretrained, "finetuned": args.finetuned}))

    tag = args.tag or "ft"
    early, late = 0, cfg.n_layers - 1
    figures = []
    for layer in (early, late):
        p = outdir / f"{args.language}_reasoning_attention_{tag}_layer{layer}.png"
        plot_comparison(pre_a[layer], ft_a[layer], pieces, layer, args.heads,
                        p, args.language, stamp, bool(font))
        figures.append(str(p))
        print(f"  wrote {p}")

    passage = outdir / f"{args.language}_reasoning_attention_{tag}_prompt.txt"
    passage.write_text(
        f"family: {shown['family']}  hops: {shown['meta']['hops']}\n"
        f"{shown['prompt']}{shown['answer_direct']}\n\n"
        f"tokens ({len(pieces)}): " + " ".join(pieces) + "\n",
        encoding="utf-8")

    result = {
        "language": args.language,
        "setup": {"n_prompts": len(prompts), "n_batches": len(batches),
                  "prompt_width_min": min(lengths),
                  "prompt_width_max": max(lengths),
                  "pretrained_step": pre_step, "finetuned_step": ft_step,
                  "early_layer": early, "late_layer": late,
                  "heads_plotted": args.heads,
                  "note": "attention distance scales with sequence length; "
                          "these values are comparable pretrained-vs-finetuned "
                          "only, never against the Phase 2 512-token windows"},
        "pretrained": stats["pretrained"], "finetuned": stats["finetuned"],
        "comparison": deltas,
        "visualised_prompt": {"family": shown["family"],
                              "prompt": shown["prompt"],
                              "answer": shown["answer"],
                              "n_tokens": len(pieces)},
        "figures": figures + [str(passage)],
        "provenance": provenance.build(
            "src.eval.attention_reasoning",
            inputs={"pretrained": args.pretrained, "finetuned": args.finetuned,
                    "test": args.test}),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"\n  {deltas['roles_changed']}/{deltas['heads_total']} heads changed "
          f"role after finetuning")
    for lay in deltas["per_layer"][:: max(len(deltas["per_layer"]) // 4, 1)]:
        d = lay["mean_distance"]
        print(f"    layer {lay['layer']}: distance "
              f"{d['pretrained']:.2f} -> {d['finetuned']:.2f} "
              f"({d['delta']:+.2f})")
    print(f"  -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
