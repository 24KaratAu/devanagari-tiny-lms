"""Export the content-hash registry of a language's manual shards on Drive.

The crawler needs to know what it must not re-collect, but the manual shards are
~780 MB and local disk is nearly full, so we stream them with `rclone cat` and
keep only the hashes (and URLs, for a cheap second dedup key).

Point ``--remote`` at either language's manual directory.

Usage:
    python scripts/export_manual_hashes.py \
        --remote gdrive:24KaratAu/nepali/cleaned/manual \
        --out data/nepali/manifests/existing_manual_hashes.txt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_REMOTE = "gdrive:24KaratAu/hindi/cleaned/manual"


def list_shards(remote: str) -> list[dict]:
    out = subprocess.run(["rclone", "lsjson", remote],
                         capture_output=True, text=True, check=True).stdout
    return sorted((e for e in json.loads(out) if e["Name"].endswith(".jsonl")),
                  key=lambda e: e["Name"])


def stream_shard(remote: str, name: str, expected_size: int, attempts: int = 4):
    """Stream a shard, verifying the byte count against Drive's file size.

    A truncated `rclone cat` looks exactly like a short file to the reader. In
    a dedup registry that is worse than a wrong number: every hash past the
    truncation point goes missing, and the crawler silently re-collects
    documents it already has.
    """
    for attempt in range(attempts):
        proc = subprocess.Popen(["rclone", "cat", f"{remote}/{name}",
                                 "--retries", "5", "--timeout", "60s"],
                                stdout=subprocess.PIPE)
        lines, read_bytes = [], 0
        for raw in proc.stdout:
            read_bytes += len(raw)
            if raw.strip():
                lines.append(raw)
        proc.stdout.close()
        proc.wait()
        if read_bytes == expected_size:
            return lines
        print(f"  {name}: short read {read_bytes:,} != {expected_size:,} "
              f"(attempt {attempt + 1}/{attempts})", file=sys.stderr, flush=True)
    raise RuntimeError(
        f"{name}: could not read all {expected_size:,} bytes after {attempts} "
        f"attempts; refusing to write an incomplete dedup registry")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--remote", default=DEFAULT_REMOTE)
    ap.add_argument("--out", default="data/hindi/manifests/existing_manual_hashes.txt")
    ap.add_argument("--urls-out", default="data/hindi/manifests/existing_manual_urls.txt")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urls_path = Path(args.urls_out)

    shards = list_shards(args.remote)
    print(f"Streaming {len(shards)} manual shards from {args.remote}", file=sys.stderr)

    hashes: set[str] = set()
    urls: set[str] = set()
    docs = 0
    for entry in shards:
        name = entry["Name"]
        for raw in stream_shard(args.remote, name, entry["Size"]):
            try:
                doc = json.loads(raw)
            except Exception:
                continue
            docs += 1
            h = doc.get("text_sha256")
            if h:
                hashes.add(h)
            u = doc.get("url")
            if u:
                urls.add(u)
        print(f"  {name}: {docs:,} docs / {len(hashes):,} hashes",
              file=sys.stderr, flush=True)

    out_path.write_text("\n".join(sorted(hashes)) + "\n", encoding="utf-8")
    urls_path.write_text("\n".join(sorted(urls)) + "\n", encoding="utf-8")
    print(f"Wrote {len(hashes):,} hashes → {out_path}")
    print(f"Wrote {len(urls):,} urls   → {urls_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
