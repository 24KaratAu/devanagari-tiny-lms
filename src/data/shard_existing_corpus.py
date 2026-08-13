"""
Shard Existing Corpus Utility

Packages the 75 existing manual web documents into a canonical corpus shard,
archives their raw HTML, uploads to Google Drive via rclone, verifies remote files,
and records the shard entry in data/nepali/manifests/manual_shards.jsonl.
"""

import json
import sys
from pathlib import Path

from src.data.shard_manager import package_and_upload_shard

MANIFEST_PATH = Path("data/nepali/manifests/manual_web.jsonl")


def shard_existing_corpus(shard_id: str = "shard-000000"):
    if not MANIFEST_PATH.exists():
        print(f"Manifest path {MANIFEST_PATH} does not exist!")
        sys.exit(1)

    records = []
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} existing manual-web records from {MANIFEST_PATH}")

    raw_paths = []
    for r in records:
        if "raw_html_path" in r:
            p = Path(r["raw_html_path"])
            if p.exists():
                raw_paths.append(p)

    print(f"Found {len(raw_paths)} associated raw HTML files.")

    success, manifest_rec = package_and_upload_shard(
        shard_id=shard_id,
        doc_records=records,
        raw_html_paths=raw_paths,
        dest_subfolder="corpus/manual",
        raw_dest_subfolder="raw/manual_web",
    )

    if success:
        print("\nExisting corpus successfully sharded, uploaded to Google Drive, and verified! ✅")
        print(f"Shard ID:           {manifest_rec['shard_id']}")
        print(f"Drive Path:         {manifest_rec['drive_path']}")
        print(f"Raw Archive Path:   {manifest_rec.get('raw_archive_drive_path')}")
        print(f"Document Count:     {manifest_rec['document_count']}")
        print(f"Clean Characters:   {manifest_rec['clean_characters']:,}")
        print(f"Shard Size:         {manifest_rec['size_bytes']:,} bytes")
        print(f"SHA-256 Checksum:   {manifest_rec['sha256']}")
        print(f"Upload Duration:    {manifest_rec.get('upload_duration_seconds')}s")
    else:
        print(f"\nFAILED to shard and upload existing corpus: {manifest_rec.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    shard_existing_corpus()
