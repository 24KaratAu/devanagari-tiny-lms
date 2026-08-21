"""
Test Storage Guard & Upload Failure Recovery Script

Tests:
  1. Storage guard enforces configured limit (LOCAL_TEMP_MAX_GB).
  2. Storage guard rejects execution when limit is exceeded or disk free space is low.
  3. Upload failure recovery: When an upload fails (e.g. invalid target remote path),
     the local temporary shard MUST NOT be deleted, and upload_status="failed" is logged in the manifest.
  4. Cleanup: Local temporary test files are safely removed after the test completes.
"""

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.shard_manager import package_and_upload_shard
from src.data.storage_guard import check_storage_safety


def run_failure_test():
    print("=" * 60)
    print("STARTING UPLOAD FAILURE RECOVERY & STORAGE GUARD TEST")
    print("=" * 60)

    # 1. Test storage guard with artificially low limit (0.000001 GB)
    print("\n--- Testing Storage Guard Limit Enforcement ---")
    os.environ["LOCAL_TEMP_MAX_GB"] = "0.000001"
    safe, msg = check_storage_safety(verbose=False)
    if not safe and "exceeded" in msg:
        print("STEP 1 PASSED: Storage guard correctly rejected operation when max limit exceeded! OK")
    else:
        print(f"FAILED Step 1: Storage guard did not reject exceeded limit: {msg}")
        sys.exit(1)

    # Reset environment variable
    os.environ["LOCAL_TEMP_MAX_GB"] = "5.0"

    # 2. Test upload failure recovery using an invalid rclone remote name
    print("\n--- Testing Upload Failure Recovery & Data Preservation ---")
    shard_id = f"test-fail-{uuid.uuid4().hex[:6]}"
    dummy_records = [
        {
            "doc_id": "faildoc001",
            "text": "नेपाली परीक्षण पाठ सामग्री...",
            "url": "https://ne.wikipedia.org/wiki/नेपाल",
            "source": "Nepali Wikipedia",
            "source_type": "wikipedia",
            "collection_type": "manual_web",
            "clean_text_characters": 30,
            "text_sha256": "1234567890abcdef",
        }
    ]

    # Force failure by setting RCLONE_REMOTE to non-existent remote
    os.environ["RCLONE_REMOTE"] = "invalid_remote_xyz_999"

    success, manifest_rec = package_and_upload_shard(
        shard_id=shard_id,
        doc_records=dummy_records,
        dest_subfolder="corpus/test",
    )

    # Reset RCLONE_REMOTE
    os.environ["RCLONE_REMOTE"] = "gdrive"

    local_shard_path = Path(manifest_rec["local_path_before_upload"])

    if not success and manifest_rec["upload_status"] == "failed":
        print("STEP 2 PASSED: Upload failure detected and recorded as 'failed' in manifest! OK")
    else:
        print(f"FAILED Step 2: Upload failure was not handled properly.")
        sys.exit(1)

    if local_shard_path.exists():
        print(f"STEP 3 PASSED: Local temporary shard file {local_shard_path.name} PRESERVED on disk after upload failure! OK")
        # Clean test shard file manually
        local_shard_path.unlink()
    else:
        print("FAILED Step 3: Local temporary shard file was deleted despite upload failure!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STORAGE GUARD & UPLOAD FAILURE RECOVERY TEST PASSED OK")
    print("=" * 60)


if __name__ == "__main__":
    run_failure_test()
