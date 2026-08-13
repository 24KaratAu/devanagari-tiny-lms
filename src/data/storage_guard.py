"""
Local Storage Safety & Storage Report Utility

Monitors local disk usage and staging directories to enforce
storage safety limits before running corpus collection or sharding.
"""

import os
import shutil
from pathlib import Path

LOCAL_TEMP_ROOT = Path("data/nepali")
RCLONE_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")
DRIVE_ROOT = os.environ.get("DRIVE_ROOT", "24KaratAu/nepali")


def get_dir_size_bytes(path: Path) -> int:
    """Calculate total size of all files in a directory recursively."""
    if not path.exists():
        return 0
    total = 0
    for p in path.glob("**/*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def check_storage_safety(verbose: bool = True) -> tuple[bool, str]:
    """Check whether local temporary corpus usage is within configured limits and disk space is safe."""
    max_gb = float(os.environ.get("LOCAL_TEMP_MAX_GB", 5.0))
    temp_bytes = get_dir_size_bytes(LOCAL_TEMP_ROOT)
    temp_gb = temp_bytes / (1024 ** 3)

    disk_usage = shutil.disk_usage(LOCAL_TEMP_ROOT if LOCAL_TEMP_ROOT.exists() else ".")
    free_gb = disk_usage.free / (1024 ** 3)

    is_temp_safe = temp_gb <= max_gb
    is_disk_safe = free_gb >= 1.0  # Require at least 1 GB free disk space
    status_str = "SAFE" if (is_temp_safe and is_disk_safe) else "LIMIT EXCEEDED / DISK LOW"

    if verbose:
        print("=" * 50)
        print("LOCAL TEMP STORAGE REPORT")
        print("=" * 50)
        print(f"Temp root:           {LOCAL_TEMP_ROOT.resolve()}")
        print(f"Current usage:       {temp_gb:.3f} GB ({temp_bytes:,} bytes)")
        print(f"Configured max:      {max_gb:.1f} GB")
        print(f"Free local disk:     {free_gb:.2f} GB")
        print(f"Status:              {status_str}")
        print("=" * 50)

    if not is_temp_safe:
        return False, f"Configured storage limit exceeded ({temp_gb:.3f} GB > {max_gb:.6f} GB)"
    if not is_disk_safe:
        return False, f"Free disk space critically low ({free_gb:.2f} GB available)"

    return True, "Storage status safe"


if __name__ == "__main__":
    safe, msg = check_storage_safety(verbose=True)
    if not safe:
        print(f"ERROR: {msg}")
