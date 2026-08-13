import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.clean.clean_nepali import clean_nepali_wikipedia

SOURCES_DIR = Path("data/nepali/sources")

RAW_BASE_DIR = Path("data/nepali/raw/manual_web")
TEXT_BASE_DIR = Path("data/nepali/processed/manual_web")

MANIFEST_PATH = Path("data/nepali/manifests/manual_web.jsonl")
FAILURES_PATH = Path("data/nepali/manifests/manual_web_failures.jsonl")

USER_AGENT = "24KaratAu-LMA-Corpus/1.0"
REQUEST_TIMEOUT = 30          # seconds
REQUEST_DELAY = 2.0           # seconds between requests
MAX_RETRIES = 3               # retries for transient HTTP errors
RETRY_BACKOFF = 1.0           # exponential backoff factor

SOURCE_METADATA_MAP = {
    "wikipedia_urls.txt": {
        "source_type": "wikipedia",
        "source": "Nepali Wikipedia",
    },
    "news_urls.txt": {
        "source_type": "news",
        "source": "Nepali News",
    },
    "government_urls.txt": {
        "source_type": "government",
        "source": "Nepali Government",
    },
    "education_urls.txt": {
        "source_type": "education",
        "source": "Nepali Education",
    },
    "literature_urls.txt": {
        "source_type": "literature",
        "source": "Nepali Literature",
    },
}


def make_doc_id(url: str) -> str:
    """Deterministic 16-hex-char document ID from URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def load_source_entries() -> list[dict]:
    """Scan data/nepali/sources/*.txt and return list of target URL entries with metadata."""
    entries = []

    if not SOURCES_DIR.exists():
        return entries

    for source_file in sorted(SOURCES_DIR.glob("*.txt")):
        filename = source_file.name
        meta = SOURCE_METADATA_MAP.get(
            filename,
            {
                "source_type": filename.replace("_urls.txt", "").replace(".txt", ""),
                "source": f"Nepali {filename.replace('_urls.txt', '').capitalize()}",
            },
        )

        for line in source_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            entries.append(
                {
                    "url": line,
                    "source_file": filename,
                    "source_type": meta["source_type"],
                    "source": meta["source"],
                }
            )

    return entries


def load_collected_urls() -> set[str]:
    """Load the set of URLs already present in the manifest."""
    collected = set()

    if not MANIFEST_PATH.exists():
        return collected

    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            collected.add(record["url"])
        except (json.JSONDecodeError, KeyError):
            continue

    return collected


def build_session() -> requests.Session:
    """Build a requests.Session with retries and User-Agent."""
    session = requests.Session()

    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({"User-Agent": USER_AGENT})

    return session


def collect_page(session: requests.Session, url: str) -> tuple[requests.Response, str, str]:
    """Fetch a single page and extract text with Trafilatura."""
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    html = response.text

    text = trafilatura.extract(
        html,
        include_links=False,
        include_images=False,
        include_tables=True,
    )

    if not text:
        raise RuntimeError(f"Trafilatura could not extract text from {url}")

    return response, html, text


def save_manifest_entry(record: dict):
    """Append one JSON record to the manifest."""
    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_failure_entry(url: str, error: Exception):
    """Append one failure record to the failures log."""
    entry = {
        "url": url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    with FAILURES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    entries = load_source_entries()
    already_collected = load_collected_urls()

    print(f"Total URLs found in source lists: {len(entries)}")
    print(f"Already collected (in manifest):   {len(already_collected)}")
    print()

    session = build_session()

    skipped = 0
    succeeded = 0
    failed = 0
    total_raw_chars = 0
    total_clean_chars = 0

    for index, entry in enumerate(entries, start=1):
        url = entry["url"]
        source_type = entry["source_type"]
        source_name = entry["source"]
        doc_id = make_doc_id(url)

        print(f"[{index}/{len(entries)}] [{source_type}] {url}")
        print(f"  doc_id: {doc_id}")

        if url in already_collected:
            print(f"  SKIP already collected: {url}")
            skipped += 1
            print()
            continue

        raw_dir = RAW_BASE_DIR / source_type
        text_dir = TEXT_BASE_DIR / source_type

        raw_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)

        try:
            response, html, raw_text = collect_page(session, url)

            clean_text = clean_nepali_wikipedia(raw_text)

            if not clean_text.strip():
                raise RuntimeError("Cleaned text is empty after cleaning")

            raw_path = raw_dir / f"{doc_id}.html"
            raw_path.write_text(html, encoding="utf-8")

            text_path = text_dir / f"{doc_id}.txt"
            text_path.write_text(clean_text, encoding="utf-8")

            record = {
                "doc_id": doc_id,
                "url": url,
                "source": source_name,
                "source_type": source_type,
                "collection_type": "manual_web",
                "collection_date": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status_code,
                "raw_html_path": str(raw_path),
                "text_path": str(text_path),
                "raw_text_characters": len(raw_text),
                "clean_text_characters": len(clean_text),
                "extraction_success": True,
                "collection_status": "success",
            }

            save_manifest_entry(record)

            succeeded += 1
            total_raw_chars += len(raw_text)
            total_clean_chars += len(clean_text)

            print(
                f"  OK  raw={len(raw_text):,}  clean={len(clean_text):,} chars"
            )

        except Exception as exc:
            failed += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            save_failure_entry(url, exc)

        print()

        if index < len(entries):
            time.sleep(REQUEST_DELAY)

    print("=" * 50)
    print("Nepali Manual Web Collection Summary")
    print("=" * 50)
    print(f"Total URLs in source lists: {len(entries)}")
    print(f"Already collected (skip):   {skipped}")
    print(f"Successfully collected:     {succeeded}")
    print(f"Failed:                     {failed}")
    print(f"Raw characters collected:   {total_raw_chars:,}")
    print(f"Clean characters collected: {total_clean_chars:,}")
    print("=" * 50)
    print(f"Manifest:     {MANIFEST_PATH}")
    print(f"Failures log: {FAILURES_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()