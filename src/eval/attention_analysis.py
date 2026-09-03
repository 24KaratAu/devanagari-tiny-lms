"""Attention analysis: heatmaps, per-head entropy, and mean attention distance.

Brief section 2.5 asks for attention heatmaps from at least one early and one
late layer across several heads on real sentences, plus per-head entropy and
mean attention distance, and then a discussion of which heads look local or
positional and which look content-based. This module produces all of it.

The two summary statistics answer different questions about a head:

  * **Mean attention distance** -- the average number of positions a query looks
    back, ``sum_j p_ij (i - j)`` averaged over query positions. A head that
    always attends to the previous token scores ~1; a head that spreads mass
    over the whole context scores in the hundreds. This is the direct evidence
    for "positional/local" versus "long-range".
  * **Entropy** -- how peaked the distribution is, in nats. Because a query at
    position ``i`` can only attend to ``i + 1`` positions, the maximum possible
    entropy grows with ``i``, so raw entropy conflates head behaviour with
    position. Normalised entropy (``H / log(i + 1)``) is reported alongside and
    is the one to compare across heads: near 0 means the head picks out one
    token, near 1 means it is close to uniform over everything available.

Two further diagnostics separate "local" from "positional" from "content-based",
which raw distance alone cannot do:

  * **Previous-token mass** -- ``p_{i, i-1}`` averaged over queries. A head with
    high previous-token mass is doing something structurally positional rather
    than selecting on content.
  * **First-position mass** -- ``p_{i, 0}``. Decoder-only models routinely
    devote a head to parking probability mass on the first token, an "attention
    sink" that lets a head emit almost no information when it has nothing to
    contribute. A head with high sink mass and high entropy elsewhere is close
    to inactive, and reading it as "long-range" would be a mistake.

Statistics are accumulated over many held-out windows, not over the handful of
sentences drawn for the heatmaps: a single sentence is far too small a sample to
characterise a head, though it is the right size to *look* at.

Devanagari labels. matplotlib's default font has no Devanagari coverage, so
token labels render as empty boxes unless a suitable font is installed. The
module looks for one and, failing that, falls back to labelling axes by position
index and writing the token pieces to a companion text file, rather than
producing a figure full of tofu.

Usage:
    python -m src.eval.attention_analysis \
        --checkpoint <drive>/hindi/checkpoints/checkpoint_best.pt \
        --model-config hindi/configs/model.yaml \
        --tokenizer tokenizers/hindi/hindi_bpe_8k.model \
        --tokens-dir <drive>/hindi/tokens --language hindi \
        --figdir hindi/figures --out hindi/eval/attention_stats.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import sentencepiece as spm

from src.eval import provenance
from src.model.transformer_lm import TransformerLM, TransformerLMConfig
from src.train.checkpoint import load_checkpoint
from src.train.dataset import iter_eval_batches

# Families that actually carry Devanagari glyphs, in preference order.
DEVANAGARI_FONTS = ("Noto Sans Devanagari", "Noto Serif Devanagari",
                    "Nirmala UI", "Lohit Devanagari", "Mangal",
                    "Samyak Devanagari", "FreeSerif")


def find_devanagari_font(explicit: str = None) -> str:
    """Return a font family that can render Devanagari, or None.

    Three passes, because the obvious one is not enough. matplotlib builds its
    font cache once and a font installed later in the same session -- ``apt-get
    install fonts-noto-devanagari`` in an earlier notebook cell -- is invisible
    to a freshly spawned subprocess that reuses the stale cache. That is exactly
    what happened on the first Colab run, which silently fell back to numeric
    axis labels.

      1. an explicitly supplied font file, registered directly;
      2. the already-known families, which covers a normal desktop;
      3. a filesystem scan of the usual font directories, registering any
         matching file with ``addfont`` so it is usable regardless of the cache.

    Args:
        explicit: Optional path to a ``.ttf``/``.otf`` to register and use.

    Returns:
        The family name to set as ``font.family``, or None if the axis labels
        must fall back to position indices.
    """
    if explicit:
        font_manager.fontManager.addfont(explicit)
        return font_manager.FontProperties(fname=explicit).get_name()

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in DEVANAGARI_FONTS:
        if name in installed:
            return name

    # The cache may predate the font's installation; look on disk directly.
    roots = ["/usr/share/fonts", "/usr/local/share/fonts",
             str(Path.home() / ".fonts"), str(Path.home() / ".local/share/fonts")]
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        for f in list(r.rglob("*Devanagari*.ttf")) + list(r.rglob("*Devanagari*.otf")):
            try:
                font_manager.fontManager.addfont(str(f))
                name = font_manager.FontProperties(fname=str(f)).get_name()
                print(f"  registered Devanagari font from disk: {f}")
                return name
            except (OSError, RuntimeError):
                continue
    return None


@torch.no_grad()
def head_statistics(model, bin_path, ctx_len: int, batch_size: int,
                    n_batches: int, device: str) -> dict:
    """Accumulate per-head attention statistics over held-out windows.

    All four statistics are computed on the same forward passes and averaged
    over query positions ``i >= 1``. Position 0 is excluded throughout: it can
    only attend to itself, so its entropy is exactly 0 and its distance exactly
    0 for every head, which would drag every head's average towards the same
    uninformative value.

    Args:
        model: A ``TransformerLM`` in eval mode.
        bin_path: Packed ``.bin`` for the split to analyse.
        ctx_len: Window length; the full ``config.ctx_len`` unless overridden.
        batch_size: Windows per forward pass. Attention weights are
            ``(B, h, T, T)`` per layer and all layers are held at once, so this
            stays small: 8 layers x 4 x 8 x 512 x 512 fp32 is already ~268 MB.
        n_batches: How many batches to accumulate over.
        device: Device string.

    Returns:
        ``{"layers": [[per-head dicts], ...], "n_windows", "n_queries"}``.
    """
    model.eval()
    n_layers, n_heads = model.config.n_layers, model.config.n_heads

    # Running sums, one scalar per (layer, head).
    acc = {k: np.zeros((n_layers, n_heads)) for k in
           ("entropy", "entropy_norm", "distance", "prev_mass", "sink_mass")}
    n_windows = 0

    it = iter_eval_batches(bin_path, ctx_len, batch_size, None, device)
    for bi, (x, _) in enumerate(it):
        if bi >= n_batches:
            break
        _, _, attns = model(x, need_weights=True)
        B, T = x.shape

        # Query positions 1..T-1, and the offsets each one can look back over.
        pos = torch.arange(T, device=device)
        # dist[i, j] = i - j, the number of positions query i looks back to key j.
        dist = (pos.view(-1, 1) - pos.view(1, -1)).float().clamp(min=0)
        max_ent = torch.log(pos.float() + 1).clamp(min=1e-9)      # log(i+1)

        for li, w in enumerate(attns):
            w = w.float()                                     # (B, h, T, T)
            # Masked positions are exactly 0 after softmax; 0*log0 -> 0.
            ent = -(w * torch.log(w.clamp(min=1e-12))).sum(-1)  # (B, h, T)
            d = (w * dist.view(1, 1, T, T)).sum(-1)             # (B, h, T)
            prev = torch.diagonal(w, offset=-1, dim1=-2, dim2=-1)  # (B, h, T-1)
            sink = w[:, :, :, 0]                                # (B, h, T)

            acc["entropy"][li] += ent[:, :, 1:].mean(dim=(0, 2)).cpu().numpy() * B
            acc["entropy_norm"][li] += (
                ent[:, :, 1:] / max_ent[1:].view(1, 1, -1)
            ).mean(dim=(0, 2)).cpu().numpy() * B
            acc["distance"][li] += d[:, :, 1:].mean(dim=(0, 2)).cpu().numpy() * B
            acc["prev_mass"][li] += prev.mean(dim=(0, 2)).cpu().numpy() * B
            acc["sink_mass"][li] += sink[:, :, 1:].mean(dim=(0, 2)).cpu().numpy() * B

        n_windows += B
        del attns

    layers = []
    for li in range(n_layers):
        layers.append([{
            "layer": li, "head": h,
            "entropy_nats": float(acc["entropy"][li, h] / n_windows),
            "entropy_normalised": float(acc["entropy_norm"][li, h] / n_windows),
            "mean_distance": float(acc["distance"][li, h] / n_windows),
            "prev_token_mass": float(acc["prev_mass"][li, h] / n_windows),
            "first_position_mass": float(acc["sink_mass"][li, h] / n_windows),
        } for h in range(n_heads)])

    return {"layers": layers, "n_windows": n_windows,
            "n_queries": n_windows * (ctx_len - 1)}


def classify_head(stat: dict, ctx_len: int) -> str:
    """Label a head from its statistics, for the report's discussion.

    The thresholds are deliberately crude and are stated in the report as such;
    they exist to group 64 heads into something discussable, not to make a
    scientific claim. A head is called:

      * ``sink`` if most of its mass sits on the first position -- it is
        effectively idle and its distance is an artefact of that;
      * ``previous-token`` if it puts a quarter of its mass on ``i - 1``;
      * ``local`` if it looks back only a few positions on average;
      * ``diffuse`` if its normalised entropy is high, meaning it is close to
        uniform and selecting nothing in particular;
      * ``content-based`` otherwise -- concentrated, but not on a fixed offset,
        which is the signature of selection on what the tokens *are*.

    Args:
        stat: One per-head dict from :func:`head_statistics`.
        ctx_len: Context length, used to scale the "local" threshold.

    Returns:
        A one-word label.
    """
    if stat["first_position_mass"] > 0.5:
        return "sink"
    if stat["prev_token_mass"] > 0.25:
        return "previous-token"
    if stat["mean_distance"] < 0.02 * ctx_len:
        return "local"
    if stat["entropy_normalised"] > 0.8:
        return "diffuse"
    return "content-based"


@torch.no_grad()
def sample_sentence(model, bin_path, sp, ctx_len: int, n_tokens: int,
                    offset: int, device: str):
    """Take one real held-out passage and its attention weights.

    Args:
        model: The model.
        bin_path: Packed ``.bin`` to read from.
        sp: SentencePiece processor, for the token labels.
        ctx_len: Model context length.
        n_tokens: Tokens to display; a heatmap beyond ~48 is unreadable.
        offset: Token offset into the split, so the passage is reproducible.
        device: Device string.

    Returns:
        (token_ids, pieces, attentions) where attentions is a list of
        ``(n_heads, n, n)`` arrays, one per layer.
    """
    data = np.memmap(Path(bin_path), dtype=np.uint16, mode="r")
    ids = data[offset:offset + n_tokens].astype(np.int64)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    _, _, attns = model(x, need_weights=True)
    pieces = [sp.id_to_piece(int(i)) for i in ids]
    return ids.tolist(), pieces, [a[0].float().cpu().numpy() for a in attns]


def plot_head_grid(attn, pieces, layer: int, heads: list, out_path: Path,
                   language: str, stamp: str, font_ok: bool):
    """Heatmaps of several heads from one layer on one real passage.

    Args:
        attn: ``(n_heads, n, n)`` attention probabilities for the layer.
        pieces: SentencePiece pieces for the passage, used as tick labels.
        layer: Layer index, for the titles.
        heads: Which heads to draw.
        out_path: Destination PNG.
        language: For the figure title.
        stamp: Provenance line.
        font_ok: Whether Devanagari can be rendered; if not, ticks are indices.
    """
    n = len(pieces)
    labels = [p.replace("▁", " ") for p in pieces] if font_ok \
        else [str(i) for i in range(n)]

    cols = min(len(heads), 4)
    rows = math.ceil(len(heads) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.3 * rows),
                             squeeze=False)

    for ax, h in zip(axes.flat, heads):
        im = ax.imshow(attn[h], cmap="magma", vmin=0.0, aspect="auto",
                       interpolation="nearest")
        ax.set_title(f"Layer {layer}, head {h}", fontsize=10)
        ax.set_xlabel("Key position (attended to)")
        ax.set_ylabel("Query position (attending from)")
        step = max(1, n // 16)
        ax.set_xticks(range(0, n, step))
        ax.set_yticks(range(0, n, step))
        ax.set_xticklabels(labels[::step], rotation=90, fontsize=6)
        ax.set_yticklabels(labels[::step], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.045, label="Attention probability")

    for ax in axes.flat[len(heads):]:
        ax.axis("off")

    note = ("" if font_ok else
            "  (no Devanagari font installed — axes labelled by position; "
            "tokens listed in the companion .txt)")
    fig.suptitle(f"Attention patterns — {language}, layer {layer}, "
                 f"{n}-token held-out passage{note}", fontsize=12)
    fig.tight_layout()
    fig.text(0.5, -0.02, stamp, ha="center", va="top", fontsize=7, color="#6B7280")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_head_summary(stats: dict, out_path: Path, language: str, stamp: str):
    """Per-head distance and entropy across every layer, as two heatmaps.

    This is the figure that makes the 64 heads legible at once: where the dark
    cells sit in the distance panel says whether short-range heads cluster in
    the early layers, which is the standard finding and the one the report's
    discussion turns on.
    """
    layers = stats["layers"]
    n_layers, n_heads = len(layers), len(layers[0])
    dist = np.array([[h["mean_distance"] for h in L] for L in layers])
    ent = np.array([[h["entropy_normalised"] for h in L] for L in layers])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, mat, title, cbar in (
            (axes[0], dist, "Mean attention distance",
             "Positions looked back (tokens)"),
            (axes[1], ent, "Normalised attention entropy",
             "H / log(i+1);  0 = peaked, 1 = uniform")):
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        fig.colorbar(im, ax=ax, label=cbar)

    fig.suptitle(f"Per-head attention statistics — {language} "
                 f"({stats['n_windows']} held-out windows, "
                 f"{stats['n_queries']:,} query positions)", fontsize=12)
    fig.tight_layout()
    fig.text(0.5, -0.03, stamp, ha="center", va="top", fontsize=7, color="#6B7280")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--language", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--display-tokens", type=int, default=40)
    ap.add_argument("--display-offset", type=int, default=100_000)
    ap.add_argument("--heads", default="0,1,2,3")
    ap.add_argument("--font-file", default=None,
                    help="path to a Devanagari .ttf/.otf, if auto-detection fails")
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TransformerLMConfig.from_yaml(args.model_config)
    model = TransformerLM(cfg).to(device)
    meta = load_checkpoint(args.checkpoint, model, map_location="cpu",
                           restore_rng=False)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    font = find_devanagari_font(args.font_file)
    if font:
        matplotlib.rcParams["font.family"] = font
    print(f"{args.language}: checkpoint step {meta['step']:,}, device {device}, "
          f"Devanagari font {font or 'NOT FOUND (labelling by position)'}")

    bin_path = Path(args.tokens_dir) / f"{args.language}_{args.split}.bin"
    prov = provenance.build(
        "src.eval.attention_analysis",
        {"checkpoint": args.checkpoint, "model_config": args.model_config,
         "tokenizer": args.tokenizer, f"{args.split}_bin": bin_path},
        {"torch": torch.__version__, "device": device,
         "devanagari_font": font, "matplotlib": matplotlib.__version__})
    stamp = provenance.stamp_text(prov)

    stats = head_statistics(model, bin_path, cfg.ctx_len, args.batch_size,
                            args.n_batches, device)
    for L in stats["layers"]:
        for h in L:
            h["classification"] = classify_head(h, cfg.ctx_len)

    # Early and late layer, as the brief requires; the middle one is included
    # because that is where the character of the heads visibly changes.
    show_layers = sorted({0, cfg.n_layers // 2, cfg.n_layers - 1})
    heads = [int(h) for h in args.heads.split(",")]

    ids, pieces, attns = sample_sentence(model, bin_path, sp, cfg.ctx_len,
                                         args.display_tokens,
                                         args.display_offset, device)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    figures = []
    for li in show_layers:
        p = figdir / f"attention_layer{li}.png"
        plot_head_grid(attns[li], pieces, li, heads, p, args.language,
                       stamp, font is not None)
        figures.append(str(p))

    summary_png = figdir / "attention_head_summary.png"
    plot_head_summary(stats, summary_png, args.language, stamp)
    figures.append(str(summary_png))

    # The displayed passage, so the heatmaps can be read even without a font.
    passage = figdir / "attention_passage.txt"
    passage.write_text(
        f"language: {args.language}\nsplit: {args.split}\n"
        f"offset: {args.display_offset}\ntokens: {len(ids)}\n\n"
        + "\n".join(f"{i:3d}  {pid:5d}  {p}" for i, (pid, p)
                    in enumerate(zip(ids, pieces)))
        + "\n\ndecoded:\n" + sp.decode([int(i) for i in ids]) + "\n",
        encoding="utf-8")

    counts = {}
    for L in stats["layers"]:
        for h in L:
            counts[h["classification"]] = counts.get(h["classification"], 0) + 1

    results = {"provenance": prov, "language": args.language,
               "checkpoint_step": meta["step"], "split": args.split,
               "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
               "ctx_len": cfg.ctx_len, "display_offset": args.display_offset,
               "figures": figures, "passage_file": str(passage),
               "classification_counts": counts, **stats}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"  {stats['n_windows']} windows, {stats['n_queries']:,} query positions")
    print("  head types: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for li in (0, cfg.n_layers - 1):
        row = stats["layers"][li]
        print(f"  layer {li}: distance " +
              " ".join(f"{h['mean_distance']:6.1f}" for h in row))
    print(f"wrote {out} and {len(figures)} figures to {figdir}/")


if __name__ == "__main__":
    main()
