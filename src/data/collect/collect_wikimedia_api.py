"""
Production Wikimedia API Collector for Nepali Corpus

Features:
  - Fetches native Nepali Wikipedia articles via official Wikimedia REST & Action APIs.
  - Automatically discovers articles using MediaWiki `allpages` or category membership listings.
  - Uses process-safe atomic locking (`ManifestLock`) matching `collect_manual_web.py`.
  - Applies Trafilatura text extraction + `clean_nepali_wikipedia`.
  - Enforces composite Nepali language identification (`identify_nepali_language`).
  - Structured logging for accepts, rejections, and failures.
  - Supports `--target-accepted` limit for safe controlled batch collection.
"""

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura

from src.data.clean.clean_nepali import clean_nepali_wikipedia
from src.data.clean.nepali_lang_id import identify_nepali_language

RAW_BASE_DIR = Path("data/nepali/raw/manual_web")
TEXT_BASE_DIR = Path("data/nepali/processed/manual_web")

MANIFEST_DIR = Path("data/nepali/manifests")
MANIFEST_PATH = MANIFEST_DIR / "manual_web.jsonl"
FAILURES_PATH = MANIFEST_DIR / "manual_web_failures.jsonl"
REJECTED_PATH = MANIFEST_DIR / "manual_web_rejected.jsonl"

LOCK_FILE_PATH = MANIFEST_DIR / "collector.lock"
RESERVATIONS_DIR = MANIFEST_DIR / "reservations"

USER_AGENT = "NepaliCorpusCollector/1.0 (https://github.com/24KaratAu)"
REQUEST_TIMEOUT = 30  # seconds
REQUEST_DELAY = 0.3  # seconds between API calls

# Quality thresholds
MIN_CLEAN_CHARACTERS = 200
MIN_DEVANAGARI_CHARS = 100
MIN_DEVANAGARI_RATIO = 0.35
RESERVATION_STALE_TIMEOUT_SEC = 900  # 15 minutes
DEFAULT_TARGET_ACCEPTED_DOCS = 1000


class ManifestLock:
    """Process-safe file lock using fcntl.flock on collector.lock."""

    def __enter__(self):
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        self.fp = open(LOCK_FILE_PATH, "a+")
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            self.fp.close()
        except OSError:
            pass


def make_doc_id(url: str) -> str:
    """Deterministic 16-hex-char document ID from URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def load_committed_urls() -> set:
    """Load set of URLs already committed in manifest or rejected manifest."""
    committed = set()
    for mpath in [MANIFEST_PATH, REJECTED_PATH]:
        if mpath.exists():
            with open(mpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if "url" in record:
                            committed.add(record["url"])
                    except json.JSONDecodeError:
                        continue
    return committed


def check_and_clean_reservation(doc_id: str) -> bool:
    """Returns True if doc_id is currently reserved by a live process."""
    RESERVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    res_file = RESERVATIONS_DIR / f"{doc_id}.json"
    if not res_file.exists():
        return False

    try:
        data = json.loads(res_file.read_text(encoding="utf-8"))
        res_pid = data.get("pid")
        res_time = data.get("timestamp", 0)

        # Check PID existence
        pid_active = False
        if res_pid:
            try:
                os.kill(res_pid, 0)
                pid_active = True
            except OSError:
                pid_active = False

        is_stale = (time.time() - res_time) > RESERVATION_STALE_TIMEOUT_SEC

        if pid_active and not is_stale:
            return True

        res_file.unlink(missing_ok=True)
        return False
    except Exception:
        res_file.unlink(missing_ok=True)
        return False


def make_reservation(doc_id: str, url: str) -> None:
    """Write process reservation file."""
    RESERVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    res_file = RESERVATIONS_DIR / f"{doc_id}.json"
    res_data = {
        "doc_id": doc_id,
        "url": url,
        "pid": os.getpid(),
        "timestamp": time.time(),
    }
    res_file.write_text(json.dumps(res_data), encoding="utf-8")


def remove_reservation(doc_id: str) -> None:
    """Remove process reservation file."""
    res_file = RESERVATIONS_DIR / f"{doc_id}.json"
    res_file.unlink(missing_ok=True)


def compute_quality_metrics(clean_text: str) -> dict:
    """Calculate quality metrics on clean text."""
    total_chars = len(clean_text)
    devanagari_chars = sum(1 for c in clean_text if "\u0900" <= c <= "\u097F")
    ratio = devanagari_chars / total_chars if total_chars > 0 else 0.0
    return {
        "clean_text_characters": total_chars,
        "devanagari_characters": devanagari_chars,
        "devanagari_ratio": round(ratio, 4),
    }


def fetch_allpages_titles(limit: int, apcontinue: str = None) -> tuple[list[str], str]:
    """Fetch article titles from Action API `allpages` endpoint."""
    api_url = "https://ne.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": "0",
        "aplimit": str(min(limit, 500)),
        "format": "json",
    }
    if apcontinue:
        params["apcontinue"] = apcontinue

    headers = {"User-Agent": USER_AGENT}
    res = requests.get(api_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    if res.status_code != 200:
        return [], None

    data = res.json()
    pages = data.get("query", {}).get("allpages", [])
    titles = [p["title"] for p in pages]

    next_continue = data.get("continue", {}).get("apcontinue")
    return titles, next_continue


def fetch_category_titles(category: str, limit: int = 500) -> list[str]:
    """Fetch article titles from Action API `categorymembers` endpoint."""
    api_url = "https://ne.wikipedia.org/w/api.php"
    if not category.startswith("श्रेणी:") and not category.startswith("Category:"):
        category = f"श्रेणी:{category}"

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": str(min(limit, 500)),
        "cmtype": "page",
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    res = requests.get(api_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    if res.status_code != 200:
        return []

    data = res.json()
    members = data.get("query", {}).get("categorymembers", [])
    return [m["title"] for m in members]


def collect_article(title: str) -> dict:
    """Fetch article REST API payload, extract text, run language ID & quality checks."""
    encoded_title = urllib.parse.quote(title.replace(" ", "_"))
    canonical_url = f"https://ne.wikipedia.org/wiki/{encoded_title}"
    rest_url = f"https://ne.wikipedia.org/api/rest_v1/page/html/{encoded_title}"
    doc_id = make_doc_id(canonical_url)

    headers = {"User-Agent": USER_AGENT}

    try:
        res = requests.get(rest_url, headers=headers, timeout=REQUEST_TIMEOUT)
        if res.status_code != 200:
            return {
                "status": "failure",
                "doc_id": doc_id,
                "url": canonical_url,
                "reason": f"HTTP {res.status_code}",
            }

        html_bytes = res.content
        html_text = res.text
        raw_chars = len(html_text)

        # Text extraction via trafilatura
        extracted_text = trafilatura.extract(html_text) or ""
        clean_text = clean_nepali_wikipedia(extracted_text)

        metrics = compute_quality_metrics(clean_text)

        # Composite language detection
        lang_id_res = identify_nepali_language(clean_text)
        lang_status = lang_id_res.get("language_status", "reject")
        lang_confidence = lang_id_res.get("language_confidence", 0.0)
        lang_reason = lang_id_res.get("rejection_reason") or "low_nepali_token_ratio"
        is_nepali = (lang_status in ("accept", "uncertain"))

        # Quality check
        is_quality_pass = (
            metrics["clean_text_characters"] >= MIN_CLEAN_CHARACTERS
            and metrics["devanagari_characters"] >= MIN_DEVANAGARI_CHARS
            and metrics["devanagari_ratio"] >= MIN_DEVANAGARI_RATIO
        )

        text_sha256 = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        now_iso = datetime.now(timezone.utc).isoformat()

        if not is_quality_pass:
            reason = (
                f"quality_threshold_failed (clean_chars={metrics['clean_text_characters']}, "
                f"devanagari_chars={metrics['devanagari_characters']}, "
                f"devanagari_ratio={metrics['devanagari_ratio']})"
            )
            return {
                "status": "rejected",
                "doc_id": doc_id,
                "url": canonical_url,
                "title": title,
                "reason": reason,
                "metrics": metrics,
                "lang_id": lang_id_res,
                "html_bytes": html_bytes,
                "clean_text": clean_text,
                "text_sha256": text_sha256,
                "raw_chars": raw_chars,
                "timestamp": now_iso,
            }

        if not is_nepali and lang_confidence < 0.6:
            reason = f"language_rejected ({lang_reason})"
            return {
                "status": "rejected",
                "doc_id": doc_id,
                "url": canonical_url,
                "title": title,
                "reason": reason,
                "metrics": metrics,
                "lang_id": lang_id_res,
                "html_bytes": html_bytes,
                "clean_text": clean_text,
                "text_sha256": text_sha256,
                "raw_chars": raw_chars,
                "timestamp": now_iso,
            }

        collection_status = "accept" if is_nepali else "uncertain"

        return {
            "status": "success",
            "doc_id": doc_id,
            "url": canonical_url,
            "title": title,
            "source_type": "wikipedia",
            "source": "Nepali Wikipedia (Wikimedia API)",
            "collection_type": "manual_web_api",
            "collection_date": now_iso,
            "raw_file": f"data/nepali/raw/manual_web/wikipedia/{doc_id}.html",
            "clean_file": f"data/nepali/processed/manual_web/wikipedia/{doc_id}.txt",
            "extracted_characters": raw_chars,
            "clean_text_characters": metrics["clean_text_characters"],
            "text_sha256": text_sha256,
            "license": "CC-BY-SA 4.0 / Public Domain",
            "language": "ne",
            "collection_status": collection_status,
            "devanagari_characters": metrics["devanagari_characters"],
            "devanagari_ratio": metrics["devanagari_ratio"],
            "language_id_confidence": lang_confidence,
            "language_id_reason": lang_reason,
            "html_bytes": html_bytes,
            "clean_text": clean_text,
        }

    except Exception as e:
        return {
            "status": "failure",
            "doc_id": doc_id,
            "url": canonical_url,
            "reason": str(e),
        }


def save_and_commit_article(record: dict) -> None:
    """Save raw and processed files and write atomic manifest entry."""
    doc_id = record["doc_id"]
    status = record["status"]

    if status == "success":
        raw_path = RAW_BASE_DIR / "wikipedia" / f"{doc_id}.html"
        clean_path = TEXT_BASE_DIR / "wikipedia" / f"{doc_id}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.parent.mkdir(parents=True, exist_ok=True)

        raw_path.write_bytes(record["html_bytes"])
        clean_path.write_text(record["clean_text"], encoding="utf-8")

        manifest_entry = {k: v for k, v in record.items() if k not in ("html_bytes", "clean_text", "status")}

        with ManifestLock():
            with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

    elif status == "rejected":
        raw_path = RAW_BASE_DIR / "wikipedia" / f"{doc_id}.html"
        clean_path = TEXT_BASE_DIR / "wikipedia" / f"{doc_id}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.parent.mkdir(parents=True, exist_ok=True)

        raw_path.write_bytes(record["html_bytes"])
        clean_path.write_text(record["clean_text"], encoding="utf-8")

        rej_entry = {
            "doc_id": doc_id,
            "url": record["url"],
            "title": record.get("title", ""),
            "source_type": "wikipedia",
            "reason": record["reason"],
            "clean_text_characters": record["metrics"]["clean_text_characters"],
            "devanagari_characters": record["metrics"]["devanagari_characters"],
            "devanagari_ratio": record["metrics"]["devanagari_ratio"],
            "language_id_confidence": record["lang_id"].get("language_confidence", 0.0),
            "language_id_reason": record["lang_id"].get("rejection_reason", ""),
            "timestamp": record["timestamp"],
        }
        with ManifestLock():
            with open(REJECTED_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rej_entry, ensure_ascii=False) + "\n")

    elif status == "failure":
        fail_entry = {
            "doc_id": doc_id,
            "url": record["url"],
            "reason": record["reason"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with ManifestLock():
            with open(FAILURES_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(fail_entry, ensure_ascii=False) + "\n")


def run_api_collector(category: str = None, max_articles: int = 100, target_accepted: int = 1000) -> None:
    """Main collection loop over Wikimedia API."""
    print("=" * 60)
    print("STARTING WIKIMEDIA API NEPALI COLLECTOR")
    print(f"Target Accepted Limit: {target_accepted}")
    print(f"Max Articles to Process this Run: {max_articles}")
    if category:
        print(f"Category: {category}")
    print("=" * 60)

    # Initial lock to count current accepted documents
    with ManifestLock():
        committed_urls = load_committed_urls()
        current_accepted = 0
        if MANIFEST_PATH.exists():
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                current_accepted = sum(1 for line in f if line.strip())

    print(f"Current accepted documents in manifest: {current_accepted}")
    if current_accepted >= target_accepted:
        print(f"Target accepted limit of {target_accepted} already reached. Stopping.")
        return

    titles = []
    apcontinue = None

    if category:
        titles = fetch_category_titles(category, limit=max_articles)
    else:
        fetched_titles, apcontinue = fetch_allpages_titles(limit=max_articles)
        titles.extend(fetched_titles)

    print(f"Discovered {len(titles)} candidate article titles via MediaWiki API.")

    processed_count = 0
    accepted_count = 0
    rejected_count = 0
    skipped_count = 0

    for title in titles:
        if current_accepted + accepted_count >= target_accepted:
            print(f"\nReached target accepted limit of {target_accepted} documents! Stopping.")
            break

        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
        canonical_url = f"https://ne.wikipedia.org/wiki/{encoded_title}"
        doc_id = make_doc_id(canonical_url)

        # Atomic CHECK & RESERVE
        with ManifestLock():
            if canonical_url in committed_urls:
                skipped_count += 1
                continue
            if check_and_clean_reservation(doc_id):
                skipped_count += 1
                continue
            make_reservation(doc_id, canonical_url)

        try:
            # Collect article via REST API
            record = collect_article(title)
            save_and_commit_article(record)

            status = record["status"]
            if status == "success":
                accepted_count += 1
                clean_len = record["clean_text_characters"]
                lang_status = record["collection_status"]
                print(f"[{current_accepted + accepted_count}/{target_accepted}] ACCEPT ({lang_status}): {title[:30]:30s} ({clean_len:,} chars)")
            elif status == "rejected":
                rejected_count += 1
                reason = record["reason"]
                print(f"[-] REJECT: {title[:30]:30s} -> {reason}")
            else:
                print(f"[!] FAILURE: {title[:30]:30s} -> {record['reason']}")

            # Update committed URLs set locally
            committed_urls.add(canonical_url)
            processed_count += 1
            time.sleep(REQUEST_DELAY)

        finally:
            remove_reservation(doc_id)

    print("\n" + "=" * 60)
    print("WIKIMEDIA API COLLECTION SUMMARY")
    print(f"Processed Titles: {processed_count}")
    print(f"Accepted:         {accepted_count}")
    print(f"Rejected:         {rejected_count}")
    print(f"Skipped:          {skipped_count}")
    print(f"Total Accepted:   {current_accepted + accepted_count}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikimedia API Nepali Collector")
    parser.add_argument("--category", type=str, default=None, help="Wikimedia category to collect from")
    parser.add_argument("--max-articles", type=int, default=50, help="Maximum articles to process in this run")
    parser.add_argument("--target-accepted", type=int, default=DEFAULT_TARGET_ACCEPTED_DOCS, help="Target total accepted count")

    args = parser.parse_args()
    run_api_collector(category=args.category, max_articles=args.max_articles, target_accepted=args.target_accepted)
