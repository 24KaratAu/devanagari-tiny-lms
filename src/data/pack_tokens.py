"""
Token Packing Pipeline

Encodes the Phase 1 JSONL splits into flat ``uint16`` token streams that the
Phase 2 trainer memory-maps directly. This is a lossless change of
representation, not a subsample: every document is encoded in file order and
every token is retained.

Why ``uint16``: both Phase 1 vocabularies are exactly 8,000, so a token id fits
in two bytes. Hindi ``train.jsonl`` shrinks 4.55 GiB -> 0.87 GiB and Nepali
6.87 GiB -> 1.06 GiB, which removes SentencePiece encoding and Drive I/O from
the training loop entirely.

Why JSONL and not the ``.txt``: ``create_splits.py`` writes the plain-text
splits as ``text.strip() + "\\n\\n"``, but 81.9% of documents contain internal
blank lines, so a ``\\n\\n`` split cannot distinguish a paragraph break from a
document break (measured 2026-09-02: 1,859,604 blocks vs 848,815 real Hindi
documents, 2.19x). JSONL has one document per line, so ``<eos>`` lands only at
true document boundaries.

Usage:
    python -m src.data.pack_tokens \
        --input-dir /content/drive/MyDrive/24KaratAu/hindi/splits \
        --tokenizer tokenizers/hindi/hindi_bpe_8k.model \
        --output-dir /content/drive/MyDrive/24KaratAu/hindi/tokens \
        --language hindi
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sentencepiece as spm

# uint16 holds ids 0..65535; both Phase 1 vocabularies are 8,000.
DTYPE = np.uint16
MAX_ID = np.iinfo(DTYPE).max

_sp = None  # per-worker SentencePiece processor


def _init_worker(model_path: str):
    """Load one SentencePiece processor per worker process."""
    global _sp
    _sp = spm.SentencePieceProcessor(model_file=model_path)


def _encode_batch(payload):
    """Encode a batch of documents to token ids with a trailing <eos> each.

    Args:
        payload: (texts, eos_id) where texts is a list of document strings.

    Returns:
        (ids, n_docs, n_tokens_no_eos) with ids as a flat list of python ints.
    """
    texts, eos_id = payload
    encoded = _sp.encode(texts)
    ids = []
    for seq in encoded:
        ids.extend(seq)
        ids.append(eos_id)
    return ids, len(encoded), sum(len(s) for s in encoded)


def _iter_batches(path: Path, batch_size: int):
    """Yield (texts, sources, n_bytes, n_chars, n_raw) batches from a JSONL split.

    Reads in binary and decodes per line so a malformed line cannot abort the
    pass. Document order is preserved, which keeps the packed stream
    reproducible from the same input file.

    ``n_bytes`` counts the UTF-8 bytes of the *document text only*. It must not
    include the JSON envelope (``doc_id``, ``source``, ``text_sha256``, ...),
    which is ~45% of a typical line here: bits-per-byte is defined over the text
    the model is scored on, so counting envelope bytes would inflate
    bytes/token and silently understate BPB by roughly 6%. ``n_raw`` keeps the
    on-disk line size separately, for reporting only.
    """
    texts, sources, n_bytes, n_chars, n_raw = [], [], 0, 0, 0
    with open(path, "rb") as fh:
        for raw in fh:
            n_raw += len(raw)
            if not raw.strip():
                continue
            try:
                doc = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            text = (doc.get("text") or "").strip()
            if not text:
                continue
            texts.append(text)
            sources.append(doc.get("source", "unknown"))
            n_chars += len(text)
            n_bytes += len(text.encode("utf-8"))   # text only, no JSON envelope
            if len(texts) >= batch_size:
                yield texts, sources, n_bytes, n_chars, n_raw
                texts, sources, n_bytes, n_chars, n_raw = [], [], 0, 0, 0
    if texts:
        yield texts, sources, n_bytes, n_chars, n_raw


def pack_split(
    input_path: Path,
    output_path: Path,
    model_path: str,
    workers: int,
    batch_size: int = 512,
) -> dict:
    """Encode one JSONL split into a flat uint16 ``.bin`` and return its stats.

    The ``.bin`` is written incrementally, so peak memory stays at roughly one
    in-flight batch per worker regardless of corpus size.

    Args:
        input_path: Source ``{split}.jsonl``.
        output_path: Destination ``{lang}_{split}.bin``.
        model_path: SentencePiece model used to encode.
        workers: Process-pool size.
        batch_size: Documents per encode task.

    Returns:
        Stats dict written alongside the ``.bin`` as a JSON sidecar.
    """
    sp = spm.SentencePieceProcessor(model_file=model_path)
    eos_id, vocab_size = sp.eos_id(), sp.get_piece_size()
    if vocab_size > MAX_ID + 1:
        raise ValueError(f"vocab {vocab_size} does not fit in uint16")

    n_docs = n_tok_raw = n_tok_out = n_bytes = n_chars = n_raw = 0
    per_source = Counter()
    t0 = time.time()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out, Pool(workers, _init_worker, (model_path,)) as pool:
        pending = []
        def flush():
            """Encode queued batches in order and append them to the .bin."""
            nonlocal n_docs, n_tok_raw, n_tok_out
            if not pending:
                return
            # imap (not imap_unordered) so the packed stream keeps file order.
            for ids, nd, nt in pool.imap(_encode_batch, pending):
                arr = np.asarray(ids, dtype=DTYPE)
                out.write(arr.tobytes())
                n_docs += nd
                n_tok_raw += nt
                n_tok_out += arr.size
            pending.clear()

        for texts, sources, nb, nc, nr in _iter_batches(input_path, batch_size):
            n_bytes += nb
            n_chars += nc
            n_raw += nr
            for s in sources:
                per_source[s] += 1
            pending.append((texts, eos_id))
            if len(pending) >= workers * 4:
                flush()
                el = time.time() - t0
                print(f"    {n_docs:>10,} docs  {n_tok_out:>13,} tok  "
                      f"{n_raw / 2**30:5.2f} GiB  {n_raw / 2**20 / el:5.1f} MiB/s",
                      file=sys.stderr, flush=True)
        flush()

    return {
        "language_split": output_path.stem,
        "source_file": str(input_path),
        "n_docs": n_docs,
        "n_tokens": n_tok_out,            # includes one <eos> per document
        "n_tokens_no_eos": n_tok_raw,     # comparable to the Phase 1 figures
        "n_eos": n_docs,
        "n_chars": n_chars,
        "n_bytes_utf8": n_bytes,          # TEXT ONLY -- required for bits-per-byte
        "n_bytes_jsonl": n_raw,           # on-disk line size, reporting only
        "bytes_per_token": n_bytes / n_tok_raw if n_tok_raw else 0.0,
        "chars_per_token": n_chars / n_tok_raw if n_tok_raw else 0.0,
        "docs_per_source": dict(per_source.most_common()),
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "dtype": "uint16",
        "bin_bytes": output_path.stat().st_size,
        "tokenizer": model_path,
        "tokenizer_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "packed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
    }


def verify(bin_path: Path, input_path: Path, model_path: str, n: int = 2000) -> bool:
    """Round-trip check: decode the head of the .bin and match it to the source.

    Catches a wrong tokenizer, a byte-order mistake, or a dtype overflow before
    a multi-hour training run consumes the file.
    """
    sp = spm.SentencePieceProcessor(model_file=model_path)
    ids = np.fromfile(bin_path, dtype=DTYPE, count=n)
    if ids.size == 0:
        print("  FAIL: packed file is empty")
        return False
    if int(ids.max()) >= sp.get_piece_size():
        print(f"  FAIL: token id {ids.max()} outside vocab {sp.get_piece_size()}")
        return False
    # Drop the trailing partial document, then decode up to the last <eos>.
    eos_at = np.flatnonzero(ids == sp.eos_id())
    upto = eos_at[-1] if eos_at.size else ids.size
    decoded = sp.decode([int(i) for i in ids[:upto] if i != sp.eos_id()])

    with open(input_path, "rb") as fh:
        first = json.loads(fh.readline().decode("utf-8"))
    original = (first.get("text") or "").strip()

    # SentencePiece normalises runs of whitespace, so compare on collapsed text.
    norm = lambda t: " ".join(t.split())
    a, b = norm(decoded), norm(original)
    k = min(120, len(a), len(b))
    ok = k > 0 and a[:k] == b[:k]
    print(f"  round-trip {'OK' if ok else 'MISMATCH'} on first {k} chars")
    if not ok:
        print(f"    decoded : {a[:100]!r}")
        print(f"    expected: {b[:100]!r}")
    return bool(ok)


def repair_bytes(input_path: Path, sidecar_path: Path) -> dict:
    """Recompute ``n_bytes_utf8`` in an existing sidecar, without re-tokenizing.

    Sidecars written before the text-bytes fix counted whole JSONL lines, so
    ``bytes_per_token`` included the JSON envelope and BPB would come out ~6%
    too low. This re-reads the source and corrects the byte fields in place;
    the ``.bin`` is untouched because token ids were never affected.

    Args:
        input_path: The ``{split}.jsonl`` the sidecar was built from.
        sidecar_path: The ``{lang}_{split}.json`` to correct.

    Returns:
        The updated stats dict.
    """
    stats = json.loads(sidecar_path.read_text(encoding="utf-8"))
    n_bytes = n_raw = n_chars = 0
    with open(input_path, "rb") as fh:
        for raw in fh:
            n_raw += len(raw)
            if not raw.strip():
                continue
            try:
                text = (json.loads(raw.decode("utf-8")).get("text") or "").strip()
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if text:
                n_bytes += len(text.encode("utf-8"))
                n_chars += len(text)

    before = stats.get("bytes_per_token", 0.0)
    ntok = stats["n_tokens_no_eos"]
    stats["n_bytes_utf8"] = n_bytes
    stats["n_bytes_jsonl"] = n_raw
    stats["n_chars"] = n_chars
    stats["bytes_per_token"] = n_bytes / ntok
    stats["chars_per_token"] = n_chars / ntok
    stats["bytes_repaired_at"] = datetime.now(timezone.utc).isoformat()
    sidecar_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"  {sidecar_path.name}: {before:.2f} -> {stats['bytes_per_token']:.2f} B/tok "
          f"(BPB was understated {100 * (1 - stats['bytes_per_token'] / before):.1f}%)"
          if before else f"  {sidecar_path.name}: {stats['bytes_per_token']:.2f} B/tok")
    return stats


def main():
    """Encode the JSONL splits into flat ``uint16`` token streams.

    Writes one ``.bin`` per split plus a JSON sidecar of packing statistics,
    or, with ``--repair-bytes``, recomputes the byte counts of existing
    sidecars written before the envelope-counting fix.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="directory holding {split}.jsonl")
    ap.add_argument("--tokenizer", required=True, help="Phase 1 SentencePiece .model")
    ap.add_argument("--output-dir", required=True, help="destination for .bin + .json")
    ap.add_argument("--language", required=True, help="hindi | nepali (filename prefix)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--repair-bytes", action="store_true",
                    help="only recompute n_bytes_utf8 in existing sidecars "
                         "(no tokenization, no .bin rewrite)")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)

    if args.repair_bytes:
        print(f"[{args.language}] repairing byte counts (text only, no envelope)")
        for split in args.splits.split(","):
            repair_bytes(in_dir / f"{split}.jsonl",
                         out_dir / f"{args.language}_{split}.json")
        return

    summary = {}
    for split in args.splits.split(","):
        src = in_dir / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = out_dir / f"{args.language}_{split}.bin"
        print(f"\n[{args.language}/{split}] {src} -> {dst}", flush=True)

        stats = pack_split(src, dst, args.tokenizer, args.workers, args.batch_size)
        (out_dir / f"{args.language}_{split}.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        summary[split] = stats

        print(f"  {stats['n_docs']:,} docs / {stats['n_tokens']:,} tokens "
              f"({stats['n_tokens_no_eos']:,} + {stats['n_eos']:,} <eos>) "
              f"-> {stats['bin_bytes'] / 2**30:.2f} GiB in {stats['elapsed_sec']:.0f}s")
        if not verify(dst, src, args.tokenizer):
            raise SystemExit(f"round-trip verification failed for {dst}")

    total = sum(s["n_tokens_no_eos"] for s in summary.values())
    print(f"\n[{args.language}] corpus total (excl. <eos>): {total:,} tokens")
    for split, s in summary.items():
        print(f"  {split:5} {s['n_tokens_no_eos']:>13,} tokens  "
              f"{s['chars_per_token']:.4f} chars/tok  {s['bytes_per_token']:.2f} B/tok")


if __name__ == "__main__":
    main()
