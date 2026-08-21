"""Distil a crawl's bulky provenance into a committable summary.

`done.txt` (12-13 MB) and `rejected.jsonl` (2 MB) are gitignored and archived on
Drive. This reduces them to a few KB of per-host and per-reason outcomes so the
repository still evidences what the crawl did — the figures quoted in report
§3.3 and §3.4.

Usage:
    python scripts/summarise_crawl.py --language nepali
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", default="nepali", choices=("hindi", "nepali"))
    args = ap.parse_args()

    base = Path("data") / args.language / "manifests"
    crawl = base / "manual_web_crawl"

    seeded: set[str] = set()
    seed_file = base / "existing_manual_urls.txt"
    if seed_file.exists():
        seeded = {l.strip() for l in seed_file.read_text().splitlines() if l.strip()}
    done = {l.strip() for l in (crawl / "done.txt").read_text().splitlines() if l.strip()}
    attempted = done - seeded

    reasons: Counter[str] = Counter()
    rejected_by_host: Counter[str] = Counter()
    for line in (crawl / "rejected.jsonl").open():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        reasons[rec["reason"]] += 1
        rejected_by_host[urlparse(rec["url"]).netloc] += 1

    attempted_by_host = Counter(urlparse(u).netloc for u in attempted)
    hosts = {
        h: {"attempted": n,
            "rejected": rejected_by_host.get(h, 0),
            "accepted": n - rejected_by_host.get(h, 0)}
        for h, n in attempted_by_host.most_common()
    }

    state = json.loads((crawl / "state.json").read_text())
    stats_file = base.parent / "sources" / "frontier_stats.json"
    frontier = json.loads(stats_file.read_text()) if stats_file.exists() else {}

    summary = {
        "language": args.language,
        "description": (
            f"{args.language.capitalize()} manual web crawl "
            "(scripts/collect_manual_web_sitemap.py). The full URL frontier, "
            "attempted-URL log, rejection log and dedup registry are archived on "
            "Google Drive under <language>/manifests/; this is the committable "
            "summary. Regenerate with scripts/summarise_crawl.py."),
        "frontier": {
            "urls_discovered": frontier.get("total_urls"),
            "hosts_discovered": len(frontier.get("per_host", {})),
            "discovery_seconds": frontier.get("elapsed_sec"),
            "per_host_discovered": frontier.get("per_host", {}),
        },
        "crawl": {
            "urls_attempted": len(attempted),
            "urls_preseeded_and_skipped": len(seeded),
            "documents_accepted": state.get("accepted_docs"),
            "clean_characters_accepted": state.get("accepted_chars"),
            "shards_written": state.get("next_shard_index"),
            "acceptance_rate": round(
                state.get("accepted_docs", 0) / max(len(attempted), 1), 4),
        },
        "rejections_by_reason": dict(reasons.most_common()),
        "per_host": hosts,
    }
    out = crawl / "crawl_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    print(f"  attempted {len(attempted):,} | accepted "
          f"{state.get('accepted_docs', 0):,} | hosts {len(hosts)}")
    print("  top rejection reasons: "
          + ", ".join(f"{k}={v:,}" for k, v in reasons.most_common(5)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
