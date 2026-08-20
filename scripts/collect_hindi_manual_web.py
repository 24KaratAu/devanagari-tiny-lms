"""Phase B of the remaining Hindi manual collection: crawl the discovered frontier.

Reads the article-URL frontier produced by ``scripts/discover_hindi_urls.py``,
fetches each page, extracts the article body with trafilatura, applies the
project's Hindi cleaning + quality filters, deduplicates against the manual
shards already on Google Drive, and streams new ``manual_web_shard_XXXX.jsonl``
shards straight to Drive via rclone.

Local disk on this machine has < 4 GB free, so shards are uploaded and deleted
as soon as they are written; peak local usage stays around one shard.

The run is resumable: every attempted URL is appended to ``done.txt`` and the
shard counter / totals live in ``state.json``.

Usage:
    python scripts/collect_hindi_manual_web.py \
        --frontier data/hindi/sources/frontier.txt \
        --target-chars 350000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import io
import os
import random
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.robotparser as robotparser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from urllib.parse import urlparse

import requests
import trafilatura

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.clean.clean_hindi_web import (  # noqa: E402
    clean_web_text, devanagari_ratio, latin_ratio, passes_quality,
)
from src.data.clean.hindi_lang_id import identify_hindi_language  # noqa: E402

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "hi,en;q=0.8"}

DRIVE_MANUAL_DIR = "gdrive:24KaratAu/hindi/cleaned/manual"
DRIVE_RAW_DIR = "gdrive:24KaratAu/hindi/raw/manual_web"

# The raw archive keeps the server's HTML so the cleaning step stays
# reproducible, but a modern Hindi news page is mostly inline JS, CSS and
# base64 images: those gzip to ~56 KB/page against ~12.5 KB for the same page
# with executable and binary assets elided. At ~1 MB/s of upstream that is the
# difference between a 16-minute upload and a two-hour one, and none of the
# removed bytes take part in article extraction — the full document structure,
# text and metadata tags that trafilatura reads are preserved verbatim.
_ASSET_PATTERNS = [
    re.compile(r"<script\b[^>]*>.*?</script>", re.I | re.S),
    re.compile(r"<style\b[^>]*>.*?</style>", re.I | re.S),
    re.compile(r"<svg\b[^>]*>.*?</svg>", re.I | re.S),
    re.compile(r"<noscript\b[^>]*>.*?</noscript>", re.I | re.S),
]
_DATA_URI_RE = re.compile(r'"data:[^"]{200,}"')


def slim_html(html: str) -> str:
    for pat in _ASSET_PATTERNS:
        html = pat.sub("", html)
    return _DATA_URI_RE.sub('"data:elided-by-collector"', html)
SHARD_PREFIX = "manual_web_shard"

_stop = threading.Event()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash64(text_sha256: str) -> int:
    return int(text_sha256[:16], 16)


class HostLimiter:
    """Bounded concurrency + minimum delay per host, so one domain is never hammered."""

    def __init__(self, per_host: int, delay: float):
        self.per_host = per_host
        self.delay = delay
        self._sems: dict[str, threading.Semaphore] = {}
        self._last: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def _sem(self, host: str) -> threading.Semaphore:
        with self._lock:
            if host not in self._sems:
                self._sems[host] = threading.Semaphore(self.per_host)
            return self._sems[host]

    def acquire(self, host: str) -> None:
        self._sem(host).acquire()
        with self._lock:
            wait = self._last[host] + self.delay - time.monotonic()
            if wait > 0:
                self._last[host] = self._last[host] + self.delay
            else:
                self._last[host] = time.monotonic()
        if wait > 0:
            time.sleep(min(wait, 5.0))

    def release(self, host: str) -> None:
        self._sem(host).release()


class RobotsCache:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._parsers: dict[str, robotparser.RobotFileParser | None] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _load(scheme: str, host: str) -> robotparser.RobotFileParser | None:
        """Fetch robots.txt ourselves rather than via RobotFileParser.read().

        read() uses urllib's default User-Agent, which several Hindi news sites
        answer with 403 — and on 403 RobotFileParser silently switches to
        disallow-all, which would reject the entire frontier.
        """
        try:
            resp = requests.get(f"{scheme}://{host}/robots.txt",
                                headers=HEADERS, timeout=15)
        except Exception:
            return None
        if resp.status_code != 200 or not resp.text.strip():
            return None
        rp = robotparser.RobotFileParser()
        try:
            rp.parse(resp.text.splitlines())
        except Exception:
            return None
        return rp

    def allowed(self, url: str) -> bool:
        if not self.enabled:
            return True
        parsed = urlparse(url)
        host = parsed.netloc
        with self._lock:
            if host not in self._parsers:
                self._parsers[host] = self._load(parsed.scheme, host)
            rp = self._parsers[host]
        if rp is None:
            return True
        try:
            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            return True


class Uploader:
    """Uploads finished shards on a background thread.

    rclone moveto of a ~100 MB raw archive takes tens of seconds; doing it
    inline would hold the writer lock and stall every crawl thread.
    """

    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.queue: Queue[tuple[Path, str] | None] = Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, path: Path, remote: str) -> None:
        self.queue.put((path, remote))

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            path, remote = item
            try:
                if self.dry_run:
                    print(f"  [dry-run] kept {path.name} locally", flush=True)
                else:
                    self._upload_with_retry(path, remote)
            finally:
                self.queue.task_done()

    @staticmethod
    def _upload_with_retry(path: Path, remote: str, attempts: int = 4) -> bool:
        """rclone's own --retries does not cover the failure we actually hit.

        When the IPv6 route to googleapis.com drops, rclone aborts while
        *constructing* the remote filesystem, before any transfer retry logic
        applies, so the retry has to live out here. Do NOT "fix" that by adding
        --bind 0.0.0.0: measured on this link, the bind collapsed throughput
        from 1.2 MB/s to 21 KB/s. rclone's own dialer already falls back to
        IPv4 quickly; this loop covers the rest. A shard left on local disk eats
        into the <4 GB of free space, so this must not silently give up.
        """
        last = ""
        for attempt in range(attempts):
            proc = subprocess.run(
                ["rclone", "moveto", str(path), f"{remote}/{path.name}",
                 "--retries", "10", "--low-level-retries", "20",
                 # rclone's 5-minute default idle timeout let a dead socket sit
                 # for a quarter of an hour mid-run; recycle stalls fast instead.
                 "--timeout", "45s", "--contimeout", "20s"],
                capture_output=True, text=True)
            if proc.returncode == 0:
                print(f"  ⬆️  {path.name} → Drive", flush=True)
                return True
            last = proc.stderr.strip()[:300]
            if attempt < attempts - 1:
                time.sleep(10 * (attempt + 1))
        print(f"  ⚠️  upload failed for {path.name} after {attempts} attempts: "
              f"{last}\n     kept at {path}", file=sys.stderr, flush=True)
        return False

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=1800)


class ShardWriter:
    """Buffers accepted docs and flushes full shards to Drive, then deletes locally."""

    def __init__(self, staging: Path, shard_size: int, start_index: int,
                 remote: str, dry_run: bool, uploader: "Uploader",
                 raw_remote: str | None = None):
        self.staging = staging
        self.shard_size = shard_size
        self.index = start_index
        self.remote = remote
        self.raw_remote = raw_remote
        self.uploader = uploader
        self.dry_run = dry_run
        self.buffer: list[dict] = []
        self.lock = threading.Lock()
        self.shards_written = 0
        self._tar: tarfile.TarFile | None = None
        self._tar_path: Path | None = None
        staging.mkdir(parents=True, exist_ok=True)

    def _raw_archive(self) -> tarfile.TarFile:
        """Open (lazily) the .tar.gz that shadows the current JSONL shard.

        The course Q&A asks for raw *and* cleaned data. Appending each response
        body into an open gzip stream keeps memory flat — we never hold 5,000
        HTML documents at once.
        """
        if self._tar is None:
            self._tar_path = self.staging / f"{SHARD_PREFIX}_raw_{self.index:04d}.tar.gz"
            self._tar = tarfile.open(self._tar_path, "w:gz", compresslevel=6)
        return self._tar

    def add(self, doc: dict, raw_html: str | None = None) -> None:
        with self.lock:
            self.buffer.append(doc)
            if raw_html is not None and self.raw_remote:
                payload = raw_html.encode("utf-8", "replace")
                info = tarfile.TarInfo(name=f"{doc['doc_id']}.html")
                info.size = len(payload)
                info.mtime = int(time.time())
                self._raw_archive().addfile(info, io.BytesIO(payload))
            if len(self.buffer) >= self.shard_size:
                self._flush_locked()

    def flush(self) -> None:
        with self.lock:
            if self.buffer:
                self._flush_locked()

    def _flush_locked(self) -> None:
        name = f"{SHARD_PREFIX}_{self.index:04d}.jsonl"
        path = self.staging / name
        with path.open("w", encoding="utf-8") as fh:
            for doc in self.buffer:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        chars = sum(d["clean_text_characters"] for d in self.buffer)
        docs = len(self.buffer)
        self.buffer = []
        print(f"  📦 {name}: {docs:,} docs, {chars:,} chars → upload queue", flush=True)
        self.uploader.submit(path, self.remote)
        if self._tar is not None:
            self._tar.close()
            self._tar = None
            self.uploader.submit(self._tar_path, self.raw_remote)
            self._tar_path = None
        self.index += 1
        self.shards_written += 1


def load_existing_hashes(path: Path) -> set[int]:
    """Load the 64-bit dedup registry built by scripts/export_hindi_manual_hashes.py."""
    hashes: set[int] = set()
    if not path or not path.exists():
        return hashes
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                hashes.add(int(line[:16], 16))
            except ValueError:
                continue
    return hashes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frontier", default="data/hindi/sources/frontier.txt")
    ap.add_argument("--state-dir", default="data/hindi/manifests/manual_web_crawl")
    ap.add_argument("--staging-dir", default=None,
                    help="Where shards are written before upload (default: <state-dir>/staging)")
    ap.add_argument("--hashes", default="data/hindi/manifests/existing_manual_hashes.txt")
    ap.add_argument("--remote", default=DRIVE_MANUAL_DIR)
    ap.add_argument("--raw-remote", default=DRIVE_RAW_DIR)
    ap.add_argument("--no-raw", action="store_true",
                    help="Skip archiving the raw HTML alongside the cleaned shards.")
    ap.add_argument("--target-chars", type=int, default=350_000_000)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--per-host", type=int, default=6)
    ap.add_argument("--host-delay", type=float, default=0.3)
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--retries", type=int, default=1,
                    help="Retries per URL on connection errors/timeouts.")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--accept-uncertain", action="store_true",
                    help="Also keep documents the language ID rates 'uncertain'.")
    ap.add_argument("--start-shard-index", type=int, default=None,
                    help="Override the first shard number (default: from state.json).")
    ap.add_argument("--ignore-robots", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do everything but keep shards on local disk.")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    done_path = state_dir / "done.txt"
    rejects_path = state_dir / "rejected.jsonl"
    staging = Path(args.staging_dir) if args.staging_dir else state_dir / "staging"

    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    shard_index = (args.start_shard_index if args.start_shard_index is not None
                   else state.get("next_shard_index", 0))
    accepted_chars = state.get("accepted_chars", 0)
    accepted_docs = state.get("accepted_docs", 0)

    frontier = [l.strip() for l in Path(args.frontier).read_text().splitlines() if l.strip()]
    done: set[str] = set()
    if done_path.exists():
        done = {l.strip() for l in done_path.read_text().splitlines() if l.strip()}
    todo = [u for u in frontier if u not in done]
    # Interleave hosts so the per-host limiter never becomes the bottleneck.
    random.Random(42).shuffle(todo)

    existing = load_existing_hashes(Path(args.hashes))
    print(f"Frontier: {len(frontier):,} URLs | already attempted: {len(done):,} | "
          f"to crawl: {len(todo):,}")
    print(f"Dedup registry: {len(existing):,} existing manual hashes")
    print(f"Resuming at shard {shard_index:04d}; {accepted_docs:,} docs / "
          f"{accepted_chars:,} chars accepted so far")
    print(f"Target: {args.target_chars:,} new clean chars "
          f"(~{int(args.target_chars / 3.19):,} tokens)")
    if not todo:
        print("Nothing left to crawl.")
        return 0

    uploader = Uploader(args.dry_run)
    writer = ShardWriter(staging, args.shard_size, shard_index, args.remote,
                         args.dry_run, uploader,
                         raw_remote=None if args.no_raw else args.raw_remote)
    limiter = HostLimiter(args.per_host, args.host_delay)
    robots = RobotsCache(not args.ignore_robots)
    seen_hashes = set(existing)
    hash_lock = threading.Lock()
    stats = defaultdict(int)
    stats_lock = threading.Lock()
    totals = {"chars": accepted_chars, "docs": accepted_docs}

    done_fh = done_path.open("a", encoding="utf-8")
    rej_fh = rejects_path.open("a", encoding="utf-8")
    io_lock = threading.Lock()

    thread_local = threading.local()

    def session() -> requests.Session:
        s = getattr(thread_local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update(HEADERS)
            thread_local.s = s
        return s

    def record_reject(url: str, reason: str) -> None:
        with stats_lock:
            stats[reason] += 1
        with io_lock:
            rej_fh.write(json.dumps({"url": url, "reason": reason}) + "\n")

    def handle(url: str) -> None:
        if _stop.is_set():
            return
        host = urlparse(url).netloc
        try:
            if not robots.allowed(url):
                record_reject(url, "robots_disallow")
                return
            resp = None
            last_exc: Exception | None = None
            for attempt in range(args.retries + 1):
                limiter.acquire(host)
                try:
                    try:
                        resp = session().get(url, timeout=args.timeout)
                    except requests.exceptions.ContentDecodingError:
                        # A few Hindi portals advertise gzip/brotli and then send
                        # a body urllib3 cannot inflate. Asking for the raw bytes
                        # recovers roughly one in ten of those pages.
                        resp = session().get(url, timeout=args.timeout,
                                             headers={"Accept-Encoding": "identity"})
                    break
                except (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout) as exc:
                    # At high concurrency several portals drop connections rather
                    # than answering; one backed-off retry recovers most of them.
                    last_exc = exc
                finally:
                    limiter.release(host)
                if attempt < args.retries:
                    time.sleep(1.5 * (attempt + 1))
            if resp is None:
                record_reject(url, f"error_{type(last_exc).__name__}")
                return
            if resp.status_code != 200:
                record_reject(url, f"http_{resp.status_code}")
                return
            body = trafilatura.extract(resp.text, include_tables=False,
                                       include_comments=False)
            if not body:
                record_reject(url, "no_extract")
                return
            text = clean_web_text(body)
            if not passes_quality(text, min_chars=args.min_chars):
                record_reject(url, "quality_filter")
                return
            # Hindi is not the only Devanagari language on the open web: Nepali,
            # Sanskrit and Maithili all pass a script-ratio test. This is the
            # same identifier the Wikipedia pipeline uses, and it is what keeps
            # the Hindi corpus disjoint from the Nepali (Model L) corpus.
            lang = identify_hindi_language(text, raw_text=body)
            if lang["language_status"] == "reject" or (
                    lang["language_status"] == "uncertain" and not args.accept_uncertain):
                record_reject(url, f"langid_{lang['language_status']}_"
                                   f"{lang['language_code']}")
                return
            sha = text_hash(text)
            h64 = hash64(sha)
            with hash_lock:
                if h64 in seen_hashes:
                    record_reject(url, "duplicate")
                    return
                seen_hashes.add(h64)
            writer.add({
                "doc_id": sha[:16],
                "url": url,
                "source": f"Hindi web ({host})",
                "source_type": "news_web",
                "collection_type": "manual",
                "language": "hindi",
                "text": text,
                "clean_text_characters": len(text),
                "devanagari_ratio": lang["devanagari_ratio"],
                "latin_ratio": lang["latin_ratio"],
                "hindi_token_ratio": lang["hindi_token_ratio"],
                "language_confidence": lang["language_confidence"],
                "language_status": lang["language_status"],
                "text_sha256": sha,
            }, raw_html=None if args.no_raw else slim_html(resp.text))
            with stats_lock:
                stats["accepted"] += 1
                totals["chars"] += len(text)
                totals["docs"] += 1
            if totals["chars"] >= args.target_chars:
                _stop.set()
        except Exception as exc:  # noqa: BLE001 - a single bad page must not kill the run
            record_reject(url, f"error_{type(exc).__name__}")
        finally:
            with io_lock:
                done_fh.write(url + "\n")

    def save_state() -> None:
        state_path.write_text(json.dumps({
            "next_shard_index": writer.index,
            "accepted_chars": totals["chars"],
            "accepted_docs": totals["docs"],
            "shards_this_run": writer.shards_written,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, indent=2))

    def on_signal(signum, frame):  # noqa: ARG001
        print("\n⏸  stop requested — finishing in-flight requests…", flush=True)
        _stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    started = time.time()
    processed = 0
    reporter_stop = threading.Event()

    def reporter() -> None:
        while not reporter_stop.wait(30):
            elapsed = max(time.time() - started, 1)
            pct = totals["chars"] / args.target_chars * 100
            print(f"[{elapsed/60:6.1f}m] {processed:,}/{len(todo):,} urls | "
                  f"accepted {stats['accepted']:,} | "
                  f"{totals['chars']:,} chars ({pct:.1f}% of target) | "
                  f"{processed/elapsed:.1f} url/s", flush=True)
            with io_lock:
                done_fh.flush()
                rej_fh.flush()
            save_state()

    threading.Thread(target=reporter, daemon=True).start()

    # Bound the number of in-flight requests with a semaphore rather than
    # submitting in barrier-synchronised batches: a batch only retires when its
    # slowest member hits the 15 s timeout, which was capping throughput at
    # roughly a sixth of what the worker count allows.
    inflight = threading.Semaphore(args.workers * 2)
    counter_lock = threading.Lock()

    def run_one(url: str) -> None:
        nonlocal processed
        try:
            handle(url)
        finally:
            with counter_lock:
                processed += 1
            inflight.release()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for url in todo:
            if _stop.is_set():
                break
            inflight.acquire()
            if _stop.is_set():
                inflight.release()
                break
            pool.submit(run_one, url)

    reporter_stop.set()
    writer.flush()
    print("Waiting for pending uploads…", flush=True)
    uploader.close()
    done_fh.flush(); done_fh.close()
    rej_fh.flush(); rej_fh.close()
    save_state()

    elapsed = time.time() - started
    print("=" * 68)
    print(f"Crawl finished in {elapsed/60:.1f} min")
    print(f"  URLs attempted this run : {processed:,}")
    print(f"  Accepted documents      : {stats['accepted']:,}")
    print(f"  New clean characters    : {totals['chars'] - accepted_chars:,}")
    print(f"  Cumulative clean chars  : {totals['chars']:,} "
          f"(~{int(totals['chars'] / 3.19):,} tokens)")
    print(f"  Shards uploaded         : {writer.shards_written}")
    print("  Rejections by reason:")
    for reason, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        if reason != "accepted":
            print(f"    {reason:24s} {n:,}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
