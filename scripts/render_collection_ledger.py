"""Render report/collection_ledger.md from report/collection_ledger.json.

The ledger is the definitive record of what was collected, by which script or
notebook, at which commit, and into which shards. Keeping the prose generated
from the JSON means the machine-readable and human-readable versions cannot
drift apart — edit the JSON, then re-run this.

Usage:
    python scripts/render_collection_ledger.py
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path("report/collection_ledger.json")
DST = Path("report/collection_ledger.md")


def fmt(v):
    return f"{v:,}" if isinstance(v, int) else str(v)


def truncate(text: str, n: int) -> str:
    return text if len(text) < n else text[: n - 3] + "…"


def render(d: dict) -> str:
    out = [
        "# Data Collection Ledger",
        "",
        "**Generated from `report/collection_ledger.json` by "
        "`scripts/render_collection_ledger.py`** — edit the JSON, not this file.",
        "",
        d["note"],
        "",
        "`where` is **colab** for a notebook cell run on a Google Colab runtime writing",
        "to the mounted Drive, and **local** for a script run on the author's machine",
        "streaming to Drive over `rclone`.",
        "",
        "---",
        "",
        "## At a glance",
        "",
        "| # | Date (IST) | Where | Language | Ran | Shards produced |",
        "| ---: | :--- | :--- | :--- | :--- | :--- |",
    ]
    for s in d["stages"]:
        out.append(
            f"| {s['stage']} | {s['date']} | {s['where']} | {s['language']} | "
            f"`{truncate(s.get('ran', '—'), 62)}` | {truncate(s.get('produced', '—'), 58)} |"
        )
    out += ["", "---", "", "## Stage detail", ""]

    for s in d["stages"]:
        out.append(
            f"### Stage {s['stage']} — {s['language'].capitalize()}, "
            f"{s['date']} ({s['where']})"
        )
        out.append("")
        if s.get("commits"):
            out.append(f"* **Commits**: {', '.join('`' + c + '`' for c in s['commits'])}")
        out.append(f"* **Ran**: `{s.get('ran', '—')}`")
        if s.get("source"):
            out.append(f"* **Source**: {s['source']}")
        out.append(f"* **Produced**: `{s.get('produced', '—')}`")
        if s.get("collection_type"):
            out.append(f"* **Counts toward**: **{s['collection_type']}**")
        bits = []
        if s.get("shards"):
            bits.append(f"{s['shards']} shards")
        if s.get("raw_archives"):
            bits.append(f"{s['raw_archives']} raw archives")
        if bits:
            out.append(f"* **Volume**: {', '.join(bits)}")

        m = s.get("measured")
        if m:
            out += ["", "| Measurement | Value |", "| :--- | ---: |"]
            for k, v in m.items():
                if k in ("note", "basis"):
                    continue
                if isinstance(v, dict):
                    v = ", ".join(f"{a} {fmt(b)}" for a, b in v.items())
                out.append(f"| {k.replace('_', ' ')} | {fmt(v)} |")
            if m.get("basis"):
                out += ["", f"> Basis: {m['basis']}"]
            if m.get("note"):
                out += ["", f"> **{m['note']}**"]
        if s.get("note"):
            out += ["", f"> {s['note']}"]
        out += ["", "---", ""]
    return "\n".join(out)


def main() -> int:
    DST.write_text(render(json.loads(SRC.read_text())), encoding="utf-8")
    print(f"Wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
