"""Provenance stamping shared by every module that writes an evaluation artefact.

A number in a report is a claim, and a claim needs a chain back to its inputs.
Each evaluation module writes a JSON result and, in the plotting case, a PNG;
without a record of *what produced them* the reader has to take the author's
word for which checkpoint, which split, and which revision of the code they came
from. This module builds that record once so every artefact carries the same one.

What is recorded:

  * the exact command line, so the run can be repeated verbatim;
  * the git commit, branch, and whether the tree was dirty at the time -- a dirty
    tree means the commit alone does not identify the code, and saying so is more
    useful than a commit hash that quietly lies;
  * a SHA-256 for each named input, so a log or checkpoint re-pulled from Drive
    can be checked byte-for-byte against the one actually used;
  * the Python and library versions, since metric definitions can move between
    releases.

Large inputs. Checkpoints run to hundreds of megabytes and live on a Drive mount
where reading one twice is slow. Files above ``max_digest_bytes`` are recorded by
size and modification time instead of by digest, and are marked ``digested:
false`` so the weaker guarantee is visible rather than assumed.
"""

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# 256 MiB: comfortably covers logs, sidecars, and generation dumps, while the
# ~300 MB checkpoints fall back to size + mtime.
MAX_DIGEST_BYTES = 256 * 1024 * 1024


def _git(*args) -> str:
    """Run a git command, returning None outside a repo or if git is missing."""
    try:
        r = subprocess.run(("git",) + args, capture_output=True, text=True,
                           timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def file_record(path, max_digest_bytes: int = MAX_DIGEST_BYTES) -> dict:
    """Describe one input file: size, and a digest when it is small enough.

    Args:
        path: File to describe.
        max_digest_bytes: Files larger than this are not hashed.

    Returns:
        A dict with ``path``, ``bytes``, ``digested``, and either ``sha256`` or
        ``mtime``. A missing file is recorded as ``{"path", "missing": True}``
        rather than raising -- the artefact should still be written, with the
        gap visible in it.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "missing": True}

    size = p.stat().st_size
    rec = {"path": str(path), "bytes": size}
    if size <= max_digest_bytes:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        rec["sha256"] = h.hexdigest()
        rec["digested"] = True
    else:
        rec["mtime"] = datetime.fromtimestamp(
            p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        rec["digested"] = False
    return rec


def build(module: str, inputs: dict, extra: dict = None) -> dict:
    """Assemble the provenance block embedded in an evaluation artefact.

    Args:
        module: Dotted name of the module being run, e.g. ``src.eval.intrinsic``.
        inputs: ``{label: path}`` for every file the run read.
        extra: Optional additional fields (library versions, device, seed).

    Returns:
        The provenance dict.
    """
    dirty = _git("status", "--porcelain")
    block = {
        "module": module,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": f"python -m {module} " + " ".join(sys.argv[1:]),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty),
        "python": sys.version.split()[0],
        "inputs": {k: file_record(v) for k, v in inputs.items()},
    }
    if extra:
        block.update(extra)
    return block


def stamp_text(block: dict) -> str:
    """Condense a provenance block to one line, for printing onto a figure.

    A figure that travels into a report loses its context, so the commit and the
    input digests are drawn into the image itself.
    """
    commit = (block.get("git_commit") or "no-git")[:8]
    dirty = "+dirty" if block.get("git_dirty") else ""
    parts = []
    for label, rec in block["inputs"].items():
        if rec.get("missing"):
            parts.append(f"{label}:missing")
        elif rec.get("digested"):
            parts.append(f"{label}:{rec['sha256'][:8]}")
        else:
            parts.append(f"{label}:{rec['bytes'] / 1e6:.0f}MB")
    return (f"{block['module']} @ {commit}{dirty} | " + " ".join(parts)
            + f" | {block['generated_at']}")
