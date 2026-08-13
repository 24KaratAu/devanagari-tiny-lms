"""
Audit Existing 113-Document Corpus for Nepali Language Identification

Runs the new composite Nepali language identifier over the 113 unique baseline documents.
READ-ONLY: Does not delete, modify, or rewrite production data or Drive shards.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clean.nepali_lang_id import identify_nepali_language

MANIFEST_PATH = Path("data/nepali/manifests/manual_web.jsonl")


def audit_corpus_language():
    if not MANIFEST_PATH.exists():
        print(f"Manifest {MANIFEST_PATH} not found!")
        sys.exit(1)

    records = []
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Deduplicate by URL to evaluate the 113 unique baseline documents
    seen = set()
    unique_records = []
    for r in records:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique_records.append(r)

    print("=" * 65)
    print("NEPALI LANGUAGE AUDIT — 113 BASELINE UNIQUE DOCUMENTS")
    print("=" * 65)

    accepted_docs = []
    uncertain_docs = []
    rejected_docs = []
    by_source = {}

    for r in unique_records:
        st = r.get("source_type", "unknown")
        by_source.setdefault(st, {"accept": 0, "uncertain": 0, "reject": 0, "total": 0})
        by_source[st]["total"] += 1

        text_path = Path(r["text_path"])
        text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""

        lang_res = identify_nepali_language(text)
        r["lang_audit"] = lang_res

        status = lang_res["language_status"]
        by_source[st][status] += 1

        if status == "accept":
            accepted_docs.append(r)
        elif status == "uncertain":
            uncertain_docs.append(r)
        else:
            rejected_docs.append(r)

    print(f"Total Unique Baseline Documents Inspected: {len(unique_records)}")
    print(f"Clearly Nepali (accept):                    {len(accepted_docs)} ({len(accepted_docs)/len(unique_records):.1%})")
    print(f"Uncertain / Borderline:                    {len(uncertain_docs)} ({len(uncertain_docs)/len(unique_records):.1%})")
    print(f"Likely Non-Nepali / Rejected:               {len(rejected_docs)} ({len(rejected_docs)/len(unique_records):.1%})")
    print(f"Would Change Status from Previous Baseline: {len(uncertain_docs) + len(rejected_docs)}")

    print("\n" + "-" * 65)
    print("BREAKDOWN BY SOURCE CATEGORY:")
    print("-" * 65)
    hdr_st = "Source Type"
    hdr_tot = "Total"
    hdr_acc = "Accept"
    hdr_unc = "Uncertain"
    hdr_rej = "Reject"
    print(f"  {hdr_st:15s} | {hdr_tot:5s} | {hdr_acc:6s} | {hdr_unc:9s} | {hdr_rej:6s}")
    print("  " + "-" * 55)
    for st, counts in sorted(by_source.items()):
        print(f"  {st:15s} | {counts['total']:5d} | {counts['accept']:6d} | {counts['uncertain']:9d} | {counts['reject']:6d}")

    if rejected_docs or uncertain_docs:
        print("\n" + "-" * 65)
        print("DETAILS OF UNCERTAIN / REJECTED DOCUMENTS:")
        print("-" * 65)
        for r in (rejected_docs + uncertain_docs)[:5]:
            res = r["lang_audit"]
            print(f"  URL:     {r['url']}")
            print(f"  Status:  {res['language_status']} (lang={res['language_code']}, conf={res['language_confidence']:.2f})")
            print(f"  Ratios:  dev={res['devanagari_ratio']:.1%}, ne_tok={res['nepali_token_ratio']:.1%}, hi_tok={res['hindi_token_ratio']:.1%}")
            if res.get("rejection_reason"):
                print(f"  Reason:  {res['rejection_reason']}")
            print()

    print("=" * 65)
    print("AUDIT COMPLETE — NO PRODUCTION DATA WAS MODIFIED ✅")
    print("=" * 65)


if __name__ == "__main__":
    audit_corpus_language()
