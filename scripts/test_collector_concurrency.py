"""
Test Script for Collector Concurrency Safety

Simulates multiple worker processes attempting to collect overlapping target URL sets concurrently.
Verifies:
  - Process file locking and atomic reservation prevent duplicate collection.
  - Each target URL produces at most ONE successful manifest entry.
  - Zero duplicate doc_ids and zero duplicate URLs in manifest.
"""

import json
import multiprocessing
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collect.collect_manual_web import (
    ManifestLock,
    make_doc_id,
)

# Fresh test URLs to isolate concurrency test from existing manifest contents
TEST_PREFIX = f"https://ne.wikipedia.org/wiki/test_conc_{uuid.uuid4().hex[:6]}"

URL1 = f"{TEST_PREFIX}_1"
URL2 = f"{TEST_PREFIX}_2"
URL3 = f"{TEST_PREFIX}_3"
URL4 = f"{TEST_PREFIX}_4"
URL5 = f"{TEST_PREFIX}_5"
URL6 = f"{TEST_PREFIX}_6"

WORKER_URL_MAP = {
    "worker_A": [URL1, URL2, URL3, URL4],
    "worker_B": [URL2, URL3, URL4, URL5],
    "worker_C": [URL1, URL3, URL5, URL6],
}


def dummy_fetch_and_process(entry):
    """Mock worker execution of process_single_url without real network traffic."""
    url = entry["url"]
    doc_id = make_doc_id(url)
    source_type = entry["source_type"]

    # Acquire lock for CHECK & RESERVE
    with ManifestLock():
        from src.data.collect.collect_manual_web import (
            check_and_clean_reservation,
            load_committed_urls,
            reserve_url,
        )

        if url in load_committed_urls():
            return "skipped"

        is_reserved, _ = check_and_clean_reservation(doc_id)
        if is_reserved:
            return "reserved"

        reserve_url(url, doc_id)

    # Simulate network fetch delay outside lock
    time.sleep(0.05)

    # Acquire lock for ATOMIC COMMIT
    with ManifestLock():
        from src.data.collect.collect_manual_web import (
            clear_reservation,
            save_manifest_entry,
        )

        if url in load_committed_urls():
            clear_reservation(doc_id)
            return "skipped"

        rec = {
            "doc_id": doc_id,
            "url": url,
            "source": "Nepali Wikipedia",
            "source_type": source_type,
            "collection_type": "manual_web",
            "collection_date": "2026-08-14T00:00:00Z",
            "http_status": 200,
            "raw_html_path": f"data/nepali/raw/manual_web/wikipedia/{doc_id}.html",
            "text_path": f"data/nepali/processed/manual_web/wikipedia/{doc_id}.txt",
            "raw_text_characters": 5000,
            "clean_text_characters": 4500,
            "text_sha256": "dummy_sha256_concurrency_test",
            "devanagari_ratio": 0.82,
            "latin_ratio": 0.05,
            "digit_ratio": 0.03,
            "language_code": "ne",
            "language_confidence": 0.95,
            "language_method": "composite_lexical_v1",
            "language_status": "accept",
            "extraction_success": True,
            "collection_status": "success",
        }
        save_manifest_entry(rec)
        clear_reservation(doc_id)
        return "succeeded"


def worker_task(worker_name, urls, return_dict):
    results = []
    for url in urls:
        entry = {"url": url, "source_type": "wikipedia", "source": "Nepali Wikipedia"}
        status = dummy_fetch_and_process(entry)
        results.append((url, status))
    return_dict[worker_name] = results


def run_concurrency_test():
    print("=" * 65)
    print("STARTING COLLECTOR PROCESS CONCURRENCY TEST")
    print("=" * 65)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []
    for w_name, urls in WORKER_URL_MAP.items():
        p = multiprocessing.Process(target=worker_task, args=(w_name, urls, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\nWorker Execution Results:")
    for w_name, res in return_dict.items():
        print(f"  {w_name}:")
        for u, st in res:
            print(f"    {u[-20:]}: {st}")

    # Inspect manifest for duplicates
    manifest_path = Path("data/nepali/manifests/manual_web.jsonl")
    records = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Focus on test URLs
    test_urls = set(u for urls in WORKER_URL_MAP.values() for u in urls)
    test_records = [r for r in records if r["url"] in test_urls]

    test_rec_urls = [r["url"] for r in test_records]
    test_rec_ids = [r["doc_id"] for r in test_records]

    dup_urls = len(test_rec_urls) - len(set(test_rec_urls))
    dup_ids = len(test_rec_ids) - len(set(test_rec_ids))

    print("\nConcurrency Test Verification:")
    print(f"  Total unique test URLs targeted:   {len(test_urls)}")
    print(f"  Total test records in manifest:    {len(test_records)}")
    print(f"  Duplicate test URLs in manifest:  {dup_urls}")
    print(f"  Duplicate test doc_ids in manifest:{dup_ids}")

    if dup_urls == 0 and dup_ids == 0 and len(test_records) == len(test_urls):
        print("\nCONCURRENCY TEST PASSED: Exactly 1 record per unique URL, 0 duplicates! ✅")
    else:
        print("\nFAILED: Duplicate records or missing records detected!")
        sys.exit(1)


if __name__ == "__main__":
    run_concurrency_test()
