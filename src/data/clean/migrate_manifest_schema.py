"""
Manifest Schema Migration & Normalization Script

Normalizes records in data/nepali/manifests/manual_web.jsonl to ensure
every record adheres to the standard schema:
  - doc_id
  - url
  - source
  - source_type
  - collection_type
  - collection_date
  - http_status
  - raw_html_path
  - text_path
  - raw_text_characters
  - clean_text_characters
  - extraction_success
  - collection_status

Idempotent: running multiple times produces identical schema without mutating data.
"""

import json
from pathlib import Path

import trafilatura

MANIFEST_PATH = Path("data/nepali/manifests/manual_web.jsonl")

STANDARD_KEYS = [
    "doc_id",
    "url",
    "source",
    "source_type",
    "collection_type",
    "collection_date",
    "http_status",
    "raw_html_path",
    "text_path",
    "raw_text_characters",
    "clean_text_characters",
    "extraction_success",
    "collection_status",
]


def normalize_record(record: dict) -> tuple[dict, bool]:
    """Normalize a single manifest record to the standard schema.

    Returns (normalized_record, modified_flag).
    """
    modified = False

    # 1. Determine clean_text_characters
    clean_chars = record.get("clean_text_characters")
    if clean_chars is None:
        clean_chars = record.get("text_characters")
        if clean_chars is None:
            text_p = Path(record.get("text_path", ""))
            if text_p.exists():
                clean_chars = len(text_p.read_text(encoding="utf-8"))
            else:
                clean_chars = 0
        modified = True
    clean_chars = int(clean_chars)

    # 2. Determine raw_text_characters
    raw_chars = record.get("raw_text_characters")
    if raw_chars is None:
        raw_p = Path(record.get("raw_html_path", ""))
        if raw_p.exists():
            html = raw_p.read_text(encoding="utf-8")
            raw_text = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                include_tables=True,
            )
            raw_chars = len(raw_text) if raw_text else clean_chars
        else:
            raw_chars = clean_chars
        modified = True
    raw_chars = int(raw_chars)

    # 3. Determine extraction_success
    ext_success = record.get("extraction_success")
    if ext_success is None:
        ext_success = clean_chars > 0
        modified = True
    else:
        ext_success = bool(ext_success)

    # 4. Determine collection_status
    status = record.get("collection_status")
    if status is None:
        status = (
            "success"
            if record.get("http_status") == 200 and clean_chars > 0
            else "failed"
        )
        modified = True
    else:
        status = str(status)

    # Assemble in exact STANDARD_KEYS order
    new_record = {
        "doc_id": record.get("doc_id"),
        "url": record.get("url"),
        "source": record.get("source"),
        "source_type": record.get("source_type"),
        "collection_type": record.get("collection_type"),
        "collection_date": record.get("collection_date"),
        "http_status": record.get("http_status"),
        "raw_html_path": record.get("raw_html_path"),
        "text_path": record.get("text_path"),
        "raw_text_characters": raw_chars,
        "clean_text_characters": clean_chars,
        "extraction_success": ext_success,
        "collection_status": status,
    }

    if list(record.keys()) != STANDARD_KEYS or record != new_record:
        modified = True

    return new_record, modified


def migrate_manifest(manifest_path: Path = MANIFEST_PATH):
    if not manifest_path.exists():
        print(f"Manifest file not found: {manifest_path}")
        return

    raw_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    records_before = [json.loads(line) for line in raw_lines if line.strip()]

    print(f"Manifest records before migration: {len(records_before)}")

    normalized_records = []
    modified_count = 0

    for rec in records_before:
        norm_rec, was_mod = normalize_record(rec)
        normalized_records.append(norm_rec)
        if was_mod:
            modified_count += 1

    # Write normalized JSONL
    temp_path = manifest_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for r in normalized_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    temp_path.replace(manifest_path)

    # Re-read and validate
    records_after = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    urls = [r["url"] for r in records_after]
    doc_ids = [r["doc_id"] for r in records_after]

    dup_urls = len(urls) - len(set(urls))
    dup_ids = len(doc_ids) - len(set(doc_ids))
    non_standard = [
        i for i, r in enumerate(records_after) if list(r.keys()) != STANDARD_KEYS
    ]

    print(f"Manifest records after migration:  {len(records_after)}")
    print(f"Records normalized/updated:       {modified_count}")
    print(f"Duplicate URLs:                    {dup_urls}")
    print(f"Duplicate doc_ids:                 {dup_ids}")
    print(f"Non-standard schema records:       {len(non_standard)}")

    if (
        len(records_before) == len(records_after)
        and dup_urls == 0
        and dup_ids == 0
        and len(non_standard) == 0
    ):
        print("Migration SUCCESSFUL and validated.")
    else:
        print("WARNING: Migration validation found potential issues!")


if __name__ == "__main__":
    migrate_manifest()
