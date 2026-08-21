"""Probe whether a publisher exposes a crawlable sitemap, and what it yields.

Backs the availability claims in report §3.3 (Hindi) and §3.4 (Nepali): which
outlets publish sitemaps via robots.txt, which 403/404 a direct /sitemap.xml
request, and roughly how much Devanagari article text each yields. Run this
before committing to a crawl — the Nepali result (several of the largest outlets
publish no sitemap at all) determined how that corpus had to be collected.

**This is a shallow availability probe, not a URL census.** It follows at most
four sitemaps and descends at most four children, so `article_urls_found` is a
lower bound and answers only "is there anything crawlable here?". A domain can
read 0 here and still yield six figures under the exhaustive breadth-first walk
in `scripts/discover_sitemap_urls.py` — makalukhabar.com probes at 0 but
discovers 167,268 URLs — because its articles sit below the depth this probe
reaches. Use the discoverer for counts; use this to decide where to point it.

Usage:
    python scripts/probe_sitemap_availability.py --language hindi
    python scripts/probe_sitemap_availability.py --domains https://ekantipur.com ...
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
import trafilatura

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "hi,ne,en;q=0.8"}
LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.I)
SITEMAP_RE = re.compile(r"(?im)^\s*sitemap:\s*(\S+)")

DOMAINS = {
    "hindi": ["https://www.amarujala.com", "https://www.livehindustan.com",
              "https://www.jagran.com", "https://www.bhaskar.com",
              "https://hindi.news18.com", "https://www.prabhatkhabar.com",
              "https://www.indiatv.in", "https://hindi.oneindia.com",
              "https://www.patrika.com", "https://www.punjabkesari.in",
              "https://hindi.webdunia.com", "https://hindi.moneycontrol.com",
              "https://www.naidunia.com", "https://www.thelallantop.com",
              "https://ndtv.in", "https://zeenews.india.com",
              "https://www.aajtak.in", "https://www.abplive.com"],
    "nepali": ["https://ekantipur.com", "https://www.onlinekhabar.com",
               "https://www.setopati.com", "https://www.ratopati.com",
               "https://gorkhapatraonline.com", "https://www.himalkhabar.com",
               "https://nagariknews.nagariknetwork.com",
               "https://www.annapurnapost.com", "https://www.makalukhabar.com",
               "https://www.kathmandupati.com", "https://clickmandu.com",
               "https://www.bizmandu.com", "https://arthasarokar.com",
               "https://deshsanchar.com", "https://www.nepalnews.com",
               "https://www.nepalpatra.com"],
}


def get(url: str, timeout: int = 20) -> tuple[str | None, str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        body = r.content
        if body[:2] == b"\x1f\x8b":
            try:
                body = gzip.decompress(body)
            except OSError:
                return None, "bad gzip"
        return body.decode("utf-8", "ignore"), "ok"
    except Exception as exc:
        return None, type(exc).__name__


def looks_like_article(url: str) -> bool:
    path = urlparse(url).path
    return len(path) > 18 and not path.lower().endswith(
        (".xml", ".gz", ".jpg", ".png", ".pdf"))


def probe(base: str, fetch_articles: int) -> dict:
    out: dict = {"domain": base}
    root = base.rstrip("/")

    robots, robots_status = get(root + "/robots.txt", 15)
    declared = SITEMAP_RE.findall(robots) if robots else []
    out["robots_status"] = robots_status
    out["sitemaps_in_robots"] = len(declared)

    # Does the conventional path work without robots.txt?
    _, direct_status = get(root + "/sitemap.xml", 15)
    out["direct_sitemap_xml"] = direct_status

    sources = declared[:4] or [root + "/sitemap.xml"]
    out["used_fallback"] = not declared

    articles, children = [], []
    for sm in sources:
        txt, _ = get(sm, 25)
        if not txt:
            continue
        for loc in LOC_RE.findall(txt):
            (children if (".xml" in loc or ".gz" in loc) else articles).append(loc)
    for child in children[:4]:
        txt, _ = get(child, 25)
        if txt:
            articles += [u for u in LOC_RE.findall(txt) if looks_like_article(u)]

    articles = [u for u in articles if looks_like_article(u)]
    out["child_sitemaps"] = len(children)
    out["article_urls_found"] = len(articles)
    out["has_usable_sitemap"] = bool(articles)

    if articles and fetch_articles:
        picks = random.Random(2).sample(articles, min(fetch_articles, len(articles)))
        lens, dev = [], []
        with ThreadPoolExecutor(max_workers=5) as pool:
            for body, _ in pool.map(lambda u: get(u, 15), picks):
                if not body:
                    continue
                text = trafilatura.extract(body, include_tables=False) or ""
                if len(text) > 100:
                    lens.append(len(text))
                    dev.append(sum(1 for c in text if "ऀ" <= c <= "ॿ") / len(text))
        out["articles_fetched"] = len(lens)
        out["mean_chars"] = int(sum(lens) / len(lens)) if lens else 0
        out["mean_devanagari_ratio"] = round(sum(dev) / len(dev), 3) if dev else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", choices=sorted(DOMAINS), default="hindi")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--fetch-articles", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    domains = args.domains or DOMAINS[args.language]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe, d, args.fetch_articles): d for d in domains}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"domain": futures[fut], "error": repr(exc)})
            print(".", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    results.sort(key=lambda r: -r.get("article_urls_found", 0))
    print(f"{'domain':<44}{'robots':>8}{'/sitemap.xml':>14}{'urls':>9}{'chars':>8}{'dev':>7}")
    for r in results:
        print(f"{r['domain'][:43]:<44}{r.get('sitemaps_in_robots', 0):>8}"
              f"{r.get('direct_sitemap_xml', '-')[:13]:>14}"
              f"{r.get('article_urls_found', 0):>9,}"
              f"{r.get('mean_chars', 0):>8,}{r.get('mean_devanagari_ratio', 0):>7.2f}")
    usable = sum(1 for r in results if r.get("has_usable_sitemap"))
    print(f"\n{usable}/{len(results)} domains expose a usable sitemap at probe depth.")
    print("Note: this is a shallow probe (<=4 sitemaps, <=4 children). A 0 here means")
    print("'nothing crawlable near the surface', not 'no URLs' — run")
    print("scripts/discover_sitemap_urls.py for exhaustive counts.")
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
