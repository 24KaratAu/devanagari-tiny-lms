"""
Manifest Schema Migration & Normalization Script

Normalizes records in data/nepali/manifests/manual_web.jsonl to ensure
every record adheres to the standard schema and removes duplicate records:
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
  - text_sha256
  - devanagari_ratio
  - latin_ratio
  - digit_ratio
  - extraction_success
  - collection_status

Idempotent: running multiple times produces identical schema without mutating data.
"""

import hashlib
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
    "text_sha256",
    "devanagari_ratio",
    "latin_ratio",
    "digit_ratio",
    "extraction_success",
    "collection_status",
]


def compute_text_quality_metrics(text: str) -> dict:
    total = len(text)
    if total == 0:
        return {
            "text_sha256": hashlib.sha256(b"").hexdigest(),
            "devanagari_ratio": 0.0,
            "latin_ratio": 0.0,
            "digit_ratio": 0.0,
        }

    dev_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    lat_chars = sum(1 for c in text if "a" <= c.lower() <= "z")
    dig_chars = sum(1 for c in text if c.isdigit() or "\u0966" <= c <= "\u096F")
    sha = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    return {
        "text_sha256": sha,
        "devanagari_ratio": round(dev_chars / total, 4),
        "latin_ratio": round(lat_chars / total, 4),
        "digit_ratio": round(dig_chars / total, 4),
    }


def normalize_record(record: dict) -> tuple[dict, bool]:
    """Normalize a single manifest record to the standard schema."""
    modified = False

    text_path = Path(record.get("text_path", ""))
    text_content = ""
    if text_path.exists():
        text_content = text_path.read_text(encoding="utf-8")

    clean_chars = record.get("clean_text_characters")
    if clean_chars is None:
        clean_chars = record.get("text_characters")
        if clean_chars is None:
            clean_chars = len(text_content)
        modified = True
    clean_chars = int(clean_chars)

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

    sha = record.get("text_sha256")
    dev_ratio = record.get("devanagari_ratio")
    lat_ratio = record.get("latin_ratio")
    dig_ratio = record.get("digit_ratio")

    if sha is None or dev_ratio is None or lat_ratio is None or dig_ratio is None:
        q_metrics = compute_text_quality_metrics(text_content)
        sha = q_metrics["text_sha256"]
        dev_ratio = q_metrics["devanagari_ratio"]
        lat_ratio = q_metrics["latin_ratio"]
        dig_ratio = q_metrics["digit_ratio"]
        modified = True

    ext_success = record.get("extraction_success")
    if ext_success is None:
        ext_success = clean_chars > 0
        modified = True
    else:
        ext_success = bool(ext_success)

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
        "text_sha256": sha,
        "devanagari_ratio": float(dev_ratio),
        "latin_ratio": float(lat_ratio),
        "digit_ratio": float(dig_ratio),
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

    seen_urls = set()
    normalized_records = []
    modified_count = 0

    for rec in records_before:
        if rec.get("url") in seen_urls:
            modified_count += 1
            continue
        seen_urls.add(rec.get("url"))

        norm_rec, was_mod = normalize_record(rec)
        normalized_records.append(norm_rec)
        if was_mod:
            modified_count += 1

    temp_path = manifest_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for r in normalized_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    temp_path.replace(manifest_path)

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
        dup_urls == 0
        and dup_ids == 0
        and len(non_standard) == 0
    ):
        print("Migration SUCCESSFUL and validated.")
    else:
        print("WARNING: Migration validation found potential issues!")


if __name__ == "__main__":
    migrate_manifest()
