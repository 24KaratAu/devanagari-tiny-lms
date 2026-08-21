"""Sitemap-based article-URL discovery.

For each seed domain we read robots.txt, follow every `Sitemap:` directive,
recursively descend sitemap indexes, and emit deduplicated article URLs to a
frontier file. ``SEED_DOMAINS`` below is the Hindi list and is the default;
pass ``--domains`` to target another language (the Nepali run used a separate
list, since most large Nepali outlets publish no sitemap at all).

The Colab pipeline previously guessed sequential article ids
(``news-id-850000`` ...), which 404'd almost everywhere and yielded only ~13 MB
of news text. Sitemaps give us the real, currently-live URL space instead.

Usage:
    python scripts/discover_sitemap_urls.py --out data/hindi/sources/frontier.txt

    python scripts/discover_sitemap_urls.py \
        --domains https://www.makalukhabar.com https://clickmandu.com ... \
        --out data/nepali/sources/frontier.txt
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "hi,en;q=0.8"}

LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.I)
SITEMAP_DIRECTIVE_RE = re.compile(r"(?im)^\s*sitemap:\s*(\S+)")

# Verified by probing robots.txt + sampling articles: every domain here returned
# real Devanagari article bodies through trafilatura.
SEED_DOMAINS = [
    "https://www.amarujala.com",
    "https://www.livehindustan.com",
    "https://www.jagran.com",
    "https://www.bhaskar.com",
    "https://hindi.news18.com",
    "https://www.prabhatkhabar.com",
    "https://www.indiatv.in",
    "https://hindi.oneindia.com",
    "https://www.patrika.com",
    "https://www.punjabkesari.in",
    "https://hindi.webdunia.com",
    "https://hindi.moneycontrol.com",
    "https://www.naidunia.com",
    "https://www.thelallantop.com",
    "https://www.dainiktribuneonline.com",
    "https://hindi.pardaphash.com",
    "https://navbharattimes.indiatimes.com",
    "https://www.abplive.com",
    "https://vikaspedia.in",
]

# Hosts whose sitemaps mix many languages: keep only URLs that look Hindi.
MULTILINGUAL_HOSTS = {"vikaspedia.in", "www.bbc.com", "www.dw.com"}
HINDI_PATH_HINT = re.compile(r"(^|[/.])(hi|hindi)([/.]|$)", re.I)

SKIP_SUFFIXES = (".xml", ".gz", ".jpg", ".jpeg", ".png", ".gif", ".webp",
                 ".pdf", ".css", ".js", ".mp4", ".mp3", ".zip")
# Non-article sections: listing pages, tag/author archives, live blogs, media.
SKIP_PATH_RE = re.compile(
    r"/(tag|tags|topic|topics|author|authors|photos?|photogallery|videos?|"
    r"web-stories|visualstories|shorts|live-tv|livetv|amp)(/|$)", re.I)


class Discoverer:
    def __init__(self, max_sitemaps_per_host: int, timeout: int, verbose: bool):
        self.max_sitemaps_per_host = max_sitemaps_per_host
        self.timeout = timeout
        self.verbose = verbose
        self.session_local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self.session_local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update(HEADERS)
            self.session_local.s = s
        return s

    def get(self, url: str) -> str | None:
        try:
            r = self.session.get(url, timeout=self.timeout)
            if r.status_code != 200:
                return None
            body = r.content
            if body[:2] == b"\x1f\x8b":
                try:
                    body = gzip.decompress(body)
                except OSError:
                    return None
            return body.decode("utf-8", "ignore")
        except Exception:
            return None

    @staticmethod
    def is_article_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        path = parsed.path
        if len(path) < 20:
            return False
        if path.lower().endswith(SKIP_SUFFIXES):
            return False
        if SKIP_PATH_RE.search(path):
            return False
        if parsed.netloc in MULTILINGUAL_HOSTS and not HINDI_PATH_HINT.search(
                parsed.netloc.split(".")[0] + path):
            return False
        return True

    @staticmethod
    def is_sitemap_url(url: str) -> bool:
        low = url.lower()
        return ".xml" in low or low.endswith(".gz") or "sitemap" in low

    def discover_host(self, base: str) -> tuple[str, set[str]]:
        """Return (host, article_urls) after a breadth-first sitemap walk."""
        host = urlparse(base).netloc
        robots = self.get(base.rstrip("/") + "/robots.txt") or ""
        queue = list(dict.fromkeys(SITEMAP_DIRECTIVE_RE.findall(robots)))
        if not queue:
            queue = [base.rstrip("/") + "/sitemap.xml"]

        seen_sitemaps: set[str] = set()
        articles: set[str] = set()
        processed = 0

        while queue and processed < self.max_sitemaps_per_host:
            batch, queue = queue[:16], queue[16:]
            batch = [u for u in batch if u not in seen_sitemaps]
            seen_sitemaps.update(batch)
            if not batch:
                continue
            with ThreadPoolExecutor(max_workers=8) as pool:
                bodies = list(pool.map(self.get, batch))
            processed += len(batch)
            for body in bodies:
                if not body:
                    continue
                for loc in LOC_RE.findall(body):
                    if self.is_sitemap_url(loc):
                        if loc not in seen_sitemaps:
                            queue.append(loc)
                    elif self.is_article_url(loc):
                        articles.add(loc)
            if self.verbose:
                print(f"  [{host}] sitemaps={processed} queued={len(queue)} "
                      f"articles={len(articles):,}", file=sys.stderr, flush=True)
        return host, articles


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/hindi/sources/frontier.txt")
    ap.add_argument("--stats-out", default="data/hindi/sources/frontier_stats.json")
    ap.add_argument("--max-sitemaps-per-host", type=int, default=1200)
    ap.add_argument("--host-workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--domains", nargs="*", default=None,
                    help="Override the built-in seed domain list.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    domains = args.domains or SEED_DOMAINS
    disc = Discoverer(args.max_sitemaps_per_host, args.timeout, not args.quiet)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: keep URLs already in the frontier file.
    all_urls: set[str] = set()
    if out_path.exists():
        all_urls.update(l.strip() for l in out_path.read_text().splitlines() if l.strip())
        print(f"Resuming with {len(all_urls):,} URLs already in {out_path}", file=sys.stderr)

    per_host: dict[str, int] = defaultdict(int)
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.host_workers) as pool:
        futures = {pool.submit(disc.discover_host, d): d for d in domains}
        for fut in as_completed(futures):
            domain = futures[fut]
            try:
                host, urls = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad host must not abort the run
                print(f"✗ {domain}: {exc!r}", file=sys.stderr)
                continue
            new = urls - all_urls
            all_urls |= urls
            per_host[host] = len(urls)
            print(f"✓ {host}: {len(urls):,} article URLs ({len(new):,} new)",
                  file=sys.stderr, flush=True)

    out_path.write_text("\n".join(sorted(all_urls)) + "\n", encoding="utf-8")
    stats = {
        "total_urls": len(all_urls),
        "per_host": dict(sorted(per_host.items(), key=lambda kv: -kv[1])),
        "elapsed_sec": round(time.time() - started, 1),
    }
    Path(args.stats_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
