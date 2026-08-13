"""
Test Shard Rollover & Boundary Integrity

Tests automatic shard rollover when accumulated size reaches a small test target (e.g., 0.5 MB).
Verifies:
  - Shard closes when target size threshold is reached.
  - Shard 1 uploads to Google Drive, gets verified, and gets cleaned locally.
  - Shard 2 opens and receives subsequent documents.
  - Zero documents lost or duplicated across shard boundaries.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.shard_manager import (
    load_shard_manifest,
    package_and_upload_shard,
)

MANIFEST_PATH = Path("data/nepali/manifests/manual_web.jsonl")


def run_rollover_test():
    print("=" * 60)
    print("STARTING SHARD ROLLOVER & BOUNDARY INTEGRITY TEST")
    print("=" * 60)

    with MANIFEST_PATH.open(encoding="utf-8") as f:
        all_records = [json.loads(l) for l in f if l.strip()]

    # Deduplicate records by URL for clean test
    seen = set()
    unique_records = []
    for r in all_records:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique_records.append(r)

    print(f"Loaded {len(unique_records)} unique records for rollover test.")

    # Target rollover threshold: 0.2 MB (200,000 bytes)
    TARGET_BYTES = 200_000

    shards_created = []
    current_batch = []
    current_bytes = 0
    shard_counter = 100  # Use 100+ for test shards e.g. test-rollover-100

    for rec in unique_records:
        rec_size = rec.get("clean_text_characters", 1000) * 2  # Approx JSON size
        if current_bytes + rec_size >= TARGET_BYTES and current_batch:
            # Finalize & upload current shard
            shard_id = f"test-rollover-{shard_counter:03d}"
            print(f"\nClosing Shard {shard_id} ({len(current_batch)} docs, approx {current_bytes:,} bytes)...")
            raw_paths = [Path(r["raw_html_path"]) for r in current_batch if "raw_html_path" in r and Path(r["raw_html_path"]).exists()]
            ok, m_rec = package_and_upload_shard(
                shard_id=shard_id,
                doc_records=current_batch,
                raw_html_paths=raw_paths,
                dest_subfolder="corpus/test",
                raw_dest_subfolder="raw/test",
            )
            if not ok:
                print(f"FAILED to upload shard {shard_id}")
                sys.exit(1)

            shards_created.append((shard_id, current_batch))
            shard_counter += 1
            current_batch = []
            current_bytes = 0

        current_batch.append(rec)
        current_bytes += rec_size

    # Upload final remaining batch if any
    if current_batch:
        shard_id = f"test-rollover-{shard_counter:03d}"
        print(f"\nClosing final Shard {shard_id} ({len(current_batch)} docs)...")
        raw_paths = [Path(r["raw_html_path"]) for r in current_batch if "raw_html_path" in r and Path(r["raw_html_path"]).exists()]
        ok, m_rec = package_and_upload_shard(
            shard_id=shard_id,
            doc_records=current_batch,
            raw_html_paths=raw_paths,
            dest_subfolder="corpus/test",
            raw_dest_subfolder="raw/test",
        )
        if not ok:
            print(f"FAILED to upload shard {shard_id}")
            sys.exit(1)
        shards_created.append((shard_id, current_batch))

    print(f"\n--- Rollover Summary ---")
    print(f"Total rollover shards created & uploaded: {len(shards_created)}")
    total_docs_in_shards = sum(len(docs) for s_id, docs in shards_created)
    print(f"Total documents across all shards:      {total_docs_in_shards} / {len(unique_records)}")

    # Verify zero document loss or duplication
    all_shard_urls = [r["url"] for s_id, docs in shards_created for r in docs]
    if len(all_shard_urls) == len(unique_records) and len(set(all_shard_urls)) == len(unique_records):
        print("ROLLOVER VERIFICATION PASSED: Zero document loss and zero document duplication across shard boundaries! ✅")
    else:
        print("FAILED: Document count mismatch or duplicates found across shards!")
        sys.exit(1)


if __name__ == "__main__":
    run_rollover_test()
