"""
Production-Safe Nepali Manual Web Collector

Features:
  - Process-safe atomic locking (CHECK -> RESERVE -> COLLECT -> COMMIT).
  - Stale reservation auto-recovery (prevents crashed processes from permanently blocking URLs).
  - Composite Nepali language identification (distinguishes Nepali from Hindi/Sanskrit/English).
  - Structured rejection logging with full metadata.
  - Preserves uncertain/borderline cases with collection_status='uncertain'.
  - Full backward compatibility with existing 17-key manifest format + language audit fields.
"""

import fcntl
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.clean.clean_nepali import clean_nepali_wikipedia
from src.data.clean.nepali_lang_id import identify_nepali_language

SOURCES_DIR = Path("data/nepali/sources")

RAW_BASE_DIR = Path("data/nepali/raw/manual_web")
TEXT_BASE_DIR = Path("data/nepali/processed/manual_web")

MANIFEST_DIR = Path("data/nepali/manifests")
MANIFEST_PATH = MANIFEST_DIR / "manual_web.jsonl"
FAILURES_PATH = MANIFEST_DIR / "manual_web_failures.jsonl"
REJECTED_PATH = MANIFEST_DIR / "manual_web_rejected.jsonl"

LOCK_FILE_PATH = MANIFEST_DIR / "collector.lock"
RESERVATIONS_DIR = MANIFEST_DIR / "reservations"

USER_AGENT = "24KaratAu-LMA-Corpus/1.0"
REQUEST_TIMEOUT = 30  # seconds
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3  # retries for transient HTTP errors
RETRY_BACKOFF = 1.0  # exponential backoff factor

# Quality thresholds
MIN_CLEAN_CHARACTERS = 200
MIN_DEVANAGARI_CHARS = 100
MIN_DEVANAGARI_RATIO = 0.35
RESERVATION_STALE_TIMEOUT_SEC = 900  # 15 minutes

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


def is_pid_alive(pid: int) -> bool:
    """Check whether a process ID is currently running on Linux."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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


def load_committed_urls() -> set[str]:
    """Load the set of committed URLs from the manifest."""
    committed = set()
    if not MANIFEST_PATH.exists():
        return committed

    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            committed.add(record["url"])
        except (json.JSONDecodeError, KeyError):
            continue

    return committed


def check_and_clean_reservation(doc_id: str) -> tuple[bool, str]:
    """Check if a URL doc_id is active-reserved or stale.

    Returns (is_active_reserved, status_message).
    """
    RESERVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    res_file = RESERVATIONS_DIR / f"{doc_id}.json"
    if not res_file.exists():
        return False, "no_reservation"

    try:
        data = json.loads(res_file.read_text(encoding="utf-8"))
        pid = data.get("pid", 0)
        reserved_at = datetime.fromisoformat(data.get("reserved_at"))
        elapsed = (datetime.now(timezone.utc) - reserved_at).total_seconds()

        # Stale if older than timeout or process dead
        if elapsed > RESERVATION_STALE_TIMEOUT_SEC or not is_pid_alive(pid):
            res_file.unlink(missing_ok=True)
            return False, f"stale_reservation_recovered (pid={pid}, elapsed={elapsed:.0f}s)"
        else:
            return True, f"active_reserved (pid={pid}, elapsed={elapsed:.0f}s)"

    except Exception:
        res_file.unlink(missing_ok=True)
        return False, "corrupted_reservation_cleared"


def reserve_url(url: str, doc_id: str):
    """Reserve a URL with current PID and timestamp."""
    RESERVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    res_file = RESERVATIONS_DIR / f"{doc_id}.json"
    res_data = {
        "url": url,
        "doc_id": doc_id,
        "pid": os.getpid(),
        "reserved_at": datetime.now(timezone.utc).isoformat(),
    }
    res_file.write_text(json.dumps(res_data), encoding="utf-8")


def clear_reservation(doc_id: str):
    """Clear reservation file for a doc_id."""
    res_file = RESERVATIONS_DIR / f"{doc_id}.json"
    res_file.unlink(missing_ok=True)


def compute_quality_metrics(text: str) -> tuple[dict, str | None]:
    """Compute text quality signals and evaluate minimum quality threshold."""
    total = len(text)
    if total < MIN_CLEAN_CHARACTERS:
        return {}, f"clean_text_too_short ({total} chars < {MIN_CLEAN_CHARACTERS})"

    dev_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    lat_chars = sum(1 for c in text if "a" <= c.lower() <= "z")
    dig_chars = sum(1 for c in text if c.isdigit() or "\u0966" <= c <= "\u096F")

    dev_ratio = round(dev_chars / total, 4)
    lat_ratio = round(lat_chars / total, 4)
    dig_ratio = round(dig_chars / total, 4)

    if dev_chars < MIN_DEVANAGARI_CHARS or dev_ratio < MIN_DEVANAGARI_RATIO:
        return {}, f"insufficient_devanagari (dev_chars={dev_chars}, dev_ratio={dev_ratio:.2f})"

    sha = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    metrics = {
        "text_sha256": sha,
        "devanagari_ratio": dev_ratio,
        "latin_ratio": lat_ratio,
        "digit_ratio": dig_ratio,
    }

    return metrics, None


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
    """Append one JSON record atomically under file lock."""
    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_failure_entry(url: str, error: Exception):
    """Append one failure record to the failures log atomically."""
    entry = {
        "url": url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    with FAILURES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_rejection_entry(
    url: str,
    doc_id: str,
    source_type: str,
    rejection_stage: str,
    rejection_reason: str,
    lang_res: dict = None,
    clean_chars: int = 0,
):
    """Append detailed rejection record to rejected log."""
    entry = {
        "url": url,
        "doc_id": doc_id,
        "source_type": source_type,
        "rejection_stage": rejection_stage,
        "rejection_reason": rejection_reason,
        "language_code": lang_res.get("language_code") if lang_res else "unknown",
        "language_confidence": lang_res.get("language_confidence") if lang_res else 0.0,
        "devanagari_ratio": lang_res.get("devanagari_ratio", 0.0) if lang_res else 0.0,
        "latin_ratio": lang_res.get("latin_ratio", 0.0) if lang_res else 0.0,
        "digit_ratio": lang_res.get("digit_ratio", 0.0) if lang_res else 0.0,
        "clean_text_characters": clean_chars,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with REJECTED_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def process_single_url(entry: dict, session: requests.Session) -> str:
    """Process a single URL using atomic CHECK -> RESERVE -> COLLECT -> COMMIT workflow.

    Returns status string: 'skipped', 'reserved', 'succeeded', 'rejected', 'failed'.
    """
    url = entry["url"]
    source_type = entry["source_type"]
    source_name = entry["source"]
    doc_id = make_doc_id(url)

    # STEP 1: CHECK & RESERVE under ManifestLock
    with ManifestLock():
        committed_urls = load_committed_urls()
        if url in committed_urls:
            return "skipped"

        is_reserved, res_msg = check_and_clean_reservation(doc_id)
        if is_reserved:
            return "reserved"

        if "stale" in res_msg:
            print(f"  [RECOVER] {res_msg} for {url}")

        reserve_url(url, doc_id)

    # STEP 2: COLLECT (HTTP fetch & text extraction outside lock)
    raw_dir = RAW_BASE_DIR / source_type
    text_dir = TEXT_BASE_DIR / source_type

    raw_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    try:
        response, html, raw_text = collect_page(session, url)
        clean_text = clean_nepali_wikipedia(raw_text)

        # STEP 3: Language Identification & Quality Checks
        lang_res = identify_nepali_language(clean_text)
        q_metrics, rejection_reason = compute_quality_metrics(clean_text)

        rejection_stage = None
        if rejection_reason:
            rejection_stage = "quality_threshold"
        elif lang_res["language_status"] == "reject":
            rejection_stage = "language_identification"
            rejection_reason = lang_res["rejection_reason"]

        # STEP 4: ATOMIC COMMIT under ManifestLock
        with ManifestLock():
            # Re-check committed in case another process committed during fetch
            if url in load_committed_urls():
                clear_reservation(doc_id)
                return "skipped"

            if rejection_reason:
                save_rejection_entry(
                    url=url,
                    doc_id=doc_id,
                    source_type=source_type,
                    rejection_stage=rejection_stage,
                    rejection_reason=rejection_reason,
                    lang_res=lang_res,
                    clean_chars=len(clean_text),
                )
                clear_reservation(doc_id)
                return "rejected"

            # Write raw and clean files
            raw_path = raw_dir / f"{doc_id}.html"
            raw_path.write_text(html, encoding="utf-8")

            text_path = text_dir / f"{doc_id}.txt"
            text_path.write_text(clean_text, encoding="utf-8")

            status_val = "success" if lang_res["language_status"] == "accept" else "uncertain"

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
                "text_sha256": q_metrics["text_sha256"],
                "devanagari_ratio": q_metrics["devanagari_ratio"],
                "latin_ratio": q_metrics["latin_ratio"],
                "digit_ratio": q_metrics["digit_ratio"],
                "language_code": lang_res["language_code"],
                "language_confidence": lang_res["language_confidence"],
                "language_method": lang_res["language_method"],
                "language_status": lang_res["language_status"],
                "extraction_success": True,
                "collection_status": status_val,
            }

            save_manifest_entry(record)
            clear_reservation(doc_id)
            return "succeeded"

    except Exception as exc:
        with ManifestLock():
            save_failure_entry(url, exc)
            clear_reservation(doc_id)
        return "failed"


def main():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    entries = load_source_entries()
    with ManifestLock():
        already_collected = load_committed_urls()

    print(f"Total URLs found in source lists: {len(entries)}")
    print(f"Already committed (in manifest):  {len(already_collected)}")
    print()

    session = build_session()

    skipped = 0
    reserved = 0
    succeeded = 0
    rejected = 0
    failed = 0

    for index, entry in enumerate(entries, start=1):
        status = process_single_url(entry, session)
        if status == "skipped":
            skipped += 1
        elif status == "reserved":
            reserved += 1
        elif status == "succeeded":
            succeeded += 1
        elif status == "rejected":
            rejected += 1
        elif status == "failed":
            failed += 1

        if index < len(entries):
            time.sleep(REQUEST_DELAY)

    print("=" * 50)
    print("Nepali Manual Web Collection Summary")
    print("=" * 50)
    print(f"Total URLs in source lists: {len(entries)}")
    print(f"Already committed (skip):   {skipped}")
    print(f"Active reserved (skip):     {reserved}")
    print(f"Successfully collected:     {succeeded}")
    print(f"Rejected (quality/lang):    {rejected}")
    print(f"Failed:                     {failed}")
    print("=" * 50)


if __name__ == "__main__":
    main()