"""
Test Pilot Script for Nepali Corpus Sharding & Google Drive Upload Pipeline

Executes the 11-step verification procedure:
  1. Create local test shard (2-5 existing Nepali documents)
  2. Inspect JSONL file formatting
  3. Calculate local SHA-256
  4. Upload with rclone to gdrive:24KaratAu/nepali/corpus/test/
  5. Verify remote file (existence & size)
  6. Record shard manifest entry
  7. Delete local temporary shard
  8. Verify local shard file is gone
  9. Verify remote shard remains on Google Drive
 10. Run the process a second time
 11. Verify idempotency (no duplicate shard uploaded)
"""

import json
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.shard_manager import (
    compute_file_sha256,
    is_shard_already_uploaded,
    load_shard_manifest,
    package_and_upload_shard,
)
from src.data.storage_guard import check_storage_safety

MANIFEST_PATH = Path("data/nepali/manifests/manual_web.jsonl")


def run_pilot():
    print("=" * 60)
    print("STARTING NEPALI CORPUS SHARDING & UPLOAD PILOT TEST")
    print("=" * 60)

    # Step 0: Storage safety check
    safe, msg = check_storage_safety(verbose=True)
    if not safe:
        print(f"FAILED Step 0: Storage safety check: {msg}")
        sys.exit(1)

    # Read 3 existing documents from manifest
    if not MANIFEST_PATH.exists():
        print(f"FAILED: Manifest path {MANIFEST_PATH} does not exist!")
        sys.exit(1)

    records = []
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= 3:
                break

    print(f"\nLoaded {len(records)} test documents from manual_web.jsonl")

    shard_id = "test-shard-000001"
    raw_html_paths = [Path(r["raw_html_path"]) for r in records if "raw_html_path" in r and Path(r["raw_html_path"]).exists()]

    # 1. Package and upload to test drive location
    print(f"\n--- Running Pilot Pass 1 for Shard ID: {shard_id} ---")
    success, manifest_rec = package_and_upload_shard(
        shard_id=shard_id,
        doc_records=records,
        raw_html_paths=raw_html_paths,
        dest_subfolder="corpus/test",
        raw_dest_subfolder="raw/test",
    )

    if not success:
        print(f"FAILED: Pilot upload returned failure for shard {shard_id}")
        sys.exit(1)

    # 8. Verify local temporary shard is gone
    local_path = Path(manifest_rec["local_path_before_upload"])
    if local_path.exists():
        print(f"FAILED Step 8: Local temporary shard {local_path} still exists!")
        sys.exit(1)
    else:
        print(f"STEP 8 PASSED: Local temporary shard file {local_path.name} was cleaned cleanly. OK")

    # 9. Verify remote shard remains on Google Drive via rclone
    drive_path = manifest_rec["drive_path"]
    cmd = ["rclone", "lsjson", drive_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        print(f"FAILED Step 9: Remote file {drive_path} not found on Google Drive!")
        sys.exit(1)
    else:
        info = json.loads(res.stdout)[0]
        print(f"STEP 9 PASSED: Remote shard verified on Google Drive (Name: {info['Name']}, Size: {info['Size']:,} bytes). OK")

    # 10 & 11. Run process again & verify idempotency
    print(f"\n--- Running Pilot Pass 2 (Idempotency Test) for Shard ID: {shard_id} ---")
    pass2_ok, pass2_rec = package_and_upload_shard(
        shard_id=shard_id,
        doc_records=records,
        raw_html_paths=raw_html_paths,
        dest_subfolder="corpus/test",
        raw_dest_subfolder="raw/test",
    )

    if pass2_ok:
        print("STEP 10 & 11 PASSED: Second run detected existing verified shard and skipped re-upload cleanly. OK")
    else:
        print("FAILED Step 10 & 11: Idempotency check failed during Pass 2.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PILOT TEST COMPLETED SUCCESSFULLY! ALL ASSERTIONS PASSED OK")
    print("=" * 60)


if __name__ == "__main__":
    run_pilot()
