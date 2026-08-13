"""
Corpus Shard Manager & Google Drive Upload Engine

Handles batching documents into large JSONL shards, archiving raw HTML,
uploading via rclone to Google Drive, verifying remote integrity,
updating shard manifests, and safely cleaning local temporary files.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from src.data.storage_guard import check_storage_safety

RCLONE_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")
DRIVE_ROOT = os.environ.get("DRIVE_ROOT", "24KaratAu/nepali")

SHARD_MANIFEST_PATH = Path("data/nepali/manifests/manual_shards.jsonl")
LOCAL_TMP_DIR = Path("data/nepali/tmp")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_drive_path(subfolder: str) -> str:
    """Construct rclone destination path string."""
    return f"{RCLONE_REMOTE}:{DRIVE_ROOT}/{subfolder}"


def load_shard_manifest() -> list[dict]:
    """Load existing shard manifest records."""
    if not SHARD_MANIFEST_PATH.exists():
        return []
    records = []
    for line in SHARD_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def record_shard_manifest(record: dict):
    """Append a record to the shard manifest."""
    SHARD_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SHARD_MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_shard_already_uploaded(shard_id: str) -> bool:
    """Check if a shard ID is already recorded as verified in the manifest."""
    for r in load_shard_manifest():
        if r.get("shard_id") == shard_id and r.get("verification_status") == "verified":
            return True
    return False


def upload_with_rclone(local_path: Path, drive_dest_dir: str) -> tuple[bool, str]:
    """Upload a local file to Google Drive using rclone copy with retry and backoff."""
    cmd = [
        "rclone",
        "copy",
        str(local_path),
        drive_dest_dir,
        "--retries",
        "5",
        "--low-level-retries",
        "10",
        "--tpslimit",
        "2",
    ]
    time.sleep(1.0)  # Rate limit safety delay
    try:
        start_t = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_t
        return True, f"Upload succeeded in {duration:.2f}s"
    except subprocess.CalledProcessError as err:
        return False, f"rclone error (code {err.returncode}): {err.stderr}"


def verify_remote_file(local_path: Path, drive_file_dest: str) -> tuple[bool, str]:
    """Verify that the file exists on remote Google Drive and matches local size."""
    time.sleep(1.0)  # Rate limit safety delay
    remote_parts = drive_file_dest.split(":", 1)
    if len(remote_parts) != 2:
        return False, f"Invalid remote path format: {drive_file_dest}"

    remote_name = remote_parts[0]
    remote_path = remote_parts[1]

    cmd = ["rclone", "lsjson", f"{remote_name}:{remote_path}"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        items = json.loads(res.stdout)
        if not items:
            return False, "Remote file not found in lsjson output"

        remote_size = items[0].get("Size", -1)
        local_size = local_path.stat().st_size

        if remote_size == local_size:
            return True, f"Verified remote match (size: {remote_size:,} bytes)"
        else:
            return False, f"Size mismatch: local={local_size:,} vs remote={remote_size:,}"

    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        return False, f"Verification command failed: {e}"


def package_and_upload_shard(
    shard_id: str,
    doc_records: list[dict],
    raw_html_paths: list[Path] = None,
    dest_subfolder: str = "corpus/manual",
    raw_dest_subfolder: str = "raw/manual_web",
) -> tuple[bool, dict]:
    """Package document records into a JSONL shard (and raw HTML archive), upload to Google Drive, verify, and clean local temp files."""
    # 1. Storage safety check
    safe, msg = check_storage_safety(verbose=False)
    if not safe:
        raise RuntimeError(f"Storage safety check failed: {msg}")

    # 2. Check if already uploaded
    if is_shard_already_uploaded(shard_id):
        print(f"Shard {shard_id} is already uploaded and verified. Skipping.")
        for r in load_shard_manifest():
            if r.get("shard_id") == shard_id:
                return True, r

    LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

    shard_filename = f"{shard_id}.jsonl"
    local_shard_path = LOCAL_TMP_DIR / shard_filename

    # Deduplicate records by URL/doc_id before creating shard
    seen_urls = set()
    unique_doc_records = []
    for rec in doc_records:
        u = rec.get("url")
        if u not in seen_urls:
            seen_urls.add(u)
            unique_doc_records.append(rec)

    # 3. Create local corpus shard JSONL
    print(f"Creating local shard: {local_shard_path} ({len(unique_doc_records)} unique docs)")
    total_clean_chars = 0
    with local_shard_path.open("w", encoding="utf-8") as f:
        for rec in unique_doc_records:
            clean_rec = {
                "doc_id": rec["doc_id"],
                "text": rec.get("text") or rec.get("clean_text") or Path(rec["text_path"]).read_text(encoding="utf-8"),
                "language": "nepali",
                "collection_type": rec.get("collection_type", "manual_web"),
                "source": rec.get("source", "Nepali Web"),
                "source_type": rec.get("source_type", "web"),
                "url": rec.get("url", ""),
                "collection_date": rec.get("collection_date", ""),
                "clean_text_characters": rec.get("clean_text_characters", len(rec.get("text", ""))),
                "text_sha256": rec.get("text_sha256", ""),
            }
            total_clean_chars += clean_rec["clean_text_characters"]
            f.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")

    local_size_bytes = local_shard_path.stat().st_size
    local_sha256 = compute_file_sha256(local_shard_path)

    # 4. Create raw HTML archive if raw_html_paths provided
    local_raw_archive_path = None
    raw_drive_path = ""
    if raw_html_paths:
        raw_archive_filename = f"raw-{shard_id}.tar.gz"
        local_raw_archive_path = LOCAL_TMP_DIR / raw_archive_filename
        print(f"Creating raw HTML archive: {local_raw_archive_path}")
        with tarfile.open(local_raw_archive_path, "w:gz") as tar:
            for hp in raw_html_paths:
                if hp.exists():
                    tar.add(hp, arcname=hp.name)

        raw_drive_dir = get_drive_path(raw_dest_subfolder)
        print(f"Uploading raw archive to {raw_drive_dir}...")
        raw_up_ok, raw_up_msg = upload_with_rclone(local_raw_archive_path, raw_drive_dir)
        if not raw_up_ok:
            print(f"WARNING: Raw HTML archive upload failed: {raw_up_msg}")

        raw_drive_path = f"{raw_drive_dir}/{raw_archive_filename}"

    # 5. Upload corpus shard JSONL
    drive_dest_dir = get_drive_path(dest_subfolder)
    drive_shard_file_path = f"{drive_dest_dir}/{shard_filename}"

    print(f"Uploading corpus shard to {drive_dest_dir}...")
    start_time = time.time()
    up_ok, up_msg = upload_with_rclone(local_shard_path, drive_dest_dir)
    upload_duration = time.time() - start_time

    if not up_ok:
        print(f"UPLOAD FAILED for {shard_id}: {up_msg}")
        manifest_record = {
            "shard_id": shard_id,
            "collection_type": "manual_web",
            "language": "nepali",
            "local_path_before_upload": str(local_shard_path),
            "drive_path": drive_shard_file_path,
            "size_bytes": local_size_bytes,
            "document_count": len(unique_doc_records),
            "clean_characters": total_clean_chars,
            "sha256": local_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "uploaded_at": None,
            "upload_status": "failed",
            "verification_status": "unverified",
            "error": up_msg,
        }
        record_shard_manifest(manifest_record)
        return False, manifest_record

    # 6. Verify Remote Integrity
    print(f"Verifying remote file at {drive_shard_file_path}...")
    ver_ok, ver_msg = verify_remote_file(local_shard_path, drive_shard_file_path)
    print(f"Verification result: {ver_msg}")

    if not ver_ok:
        print(f"VERIFICATION FAILED for {shard_id}")
        manifest_record = {
            "shard_id": shard_id,
            "collection_type": "manual_web",
            "language": "nepali",
            "local_path_before_upload": str(local_shard_path),
            "drive_path": drive_shard_file_path,
            "size_bytes": local_size_bytes,
            "document_count": len(unique_doc_records),
            "clean_characters": total_clean_chars,
            "sha256": local_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "upload_status": "success",
            "verification_status": "failed",
            "error": ver_msg,
        }
        record_shard_manifest(manifest_record)
        return False, manifest_record

    # 7. Record Success in Manifest
    manifest_record = {
        "shard_id": shard_id,
        "collection_type": "manual_web",
        "language": "nepali",
        "local_path_before_upload": str(local_shard_path),
        "drive_path": drive_shard_file_path,
        "raw_archive_drive_path": raw_drive_path,
        "size_bytes": local_size_bytes,
        "document_count": len(unique_doc_records),
        "clean_characters": total_clean_chars,
        "sha256": local_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "upload_duration_seconds": round(upload_duration, 2),
        "upload_status": "success",
        "verification_status": "verified",
    }
    record_shard_manifest(manifest_record)

    # 8. Clean Local Temporary Shards ONLY AFTER Verification
    print(f"Cleaning local temp files for verified shard {shard_id}...")
    if local_shard_path.exists():
        local_shard_path.unlink()
    if local_raw_archive_path and local_raw_archive_path.exists():
        local_raw_archive_path.unlink()

    print(f"Shard {shard_id} successfully uploaded, verified, and cleaned locally. ✅")
    return True, manifest_record
