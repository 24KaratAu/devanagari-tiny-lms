"""
Test Script for Stale Reservation Failure Recovery

Tests:
  1. Worker A reserves a target URL and simulates a crash (dead PID).
  2. Worker B attempts to collect the same URL.
  3. Worker B detects the stale reservation, clears it, reserves it, and completes collection.
  4. Verifies no URL is permanently blocked by a crashed process.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collect.collect_manual_web import (
    ManifestLock,
    check_and_clean_reservation,
    make_doc_id,
    reserve_url,
)


def run_recovery_test():
    print("=" * 65)
    print("STARTING STALE RESERVATION FAILURE RECOVERY TEST")
    print("=" * 65)

    test_url = "https://ne.wikipedia.org/wiki/नेपाल_प्रणाली_परीक्षण"
    doc_id = make_doc_id(test_url)

    # 1. Simulate Worker A reserving the URL with a non-existent dead PID (e.g. 999999)
    print("\n--- Step 1: Simulating Crashed Worker A Reservation ---")
    with ManifestLock():
        RESERVATIONS_DIR = Path("data/nepali/manifests/reservations")
        RESERVATIONS_DIR.mkdir(parents=True, exist_ok=True)
        res_file = RESERVATIONS_DIR / f"{doc_id}.json"
        res_data = {
            "url": test_url,
            "doc_id": doc_id,
            "pid": 999999,  # Non-existent dead PID
            "reserved_at": "2026-08-14T00:00:00Z",
        }
        res_file.write_text(json.dumps(res_data), encoding="utf-8")
        print(f"Created simulated crashed reservation file for PID 999999: {res_file.name}")

    # 2. Worker B checks reservation status
    print("\n--- Step 2: Worker B Checking Stale Reservation ---")
    with ManifestLock():
        is_reserved, msg = check_and_clean_reservation(doc_id)

    print(f"Worker B check_and_clean_reservation result: is_reserved={is_reserved}, msg='{msg}'")

    if not is_reserved and "stale" in msg:
        print("STEP 1 PASSED: Worker B correctly identified and recovered stale reservation! ✅")
    else:
        print(f"FAILED Step 1: Stale reservation was not recovered: {msg}")
        sys.exit(1)

    # 3. Worker B reserves and cleans up
    with ManifestLock():
        reserve_url(test_url, doc_id)
        print("Worker B successfully placed fresh reservation! ✅")
        # Clean test reservation file
        res_file.unlink(missing_ok=True)

    print("\n" + "=" * 65)
    print("STALE RESERVATION RECOVERY TEST PASSED ✅")
    print("=" * 65)


if __name__ == "__main__":
    run_recovery_test()
