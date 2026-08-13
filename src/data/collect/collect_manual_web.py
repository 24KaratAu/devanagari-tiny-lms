import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura

# from clean_nepali import clean_nepali_wikipedia
from src.data.clean.clean_nepali import clean_nepali_wikipedia


URLS_FILE = Path("data/nepali/sources/wikipedia_urls.txt")

RAW_DIR = Path("data/nepali/raw/manual_web/wikipedia")
TEXT_DIR = Path("data/nepali/processed/manual_web/wikipedia")
MANIFEST_PATH = Path("data/nepali/manifests/manual_web.jsonl")

USER_AGENT = "24KaratAu-LMA-Corpus/1.0"

REQUEST_DELAY = 2.0


def make_doc_id(url):
    return hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]


def load_urls():
    urls = []

    for line in URLS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        urls.append(line)

    return urls


def already_collected(doc_id):
    return (
        (RAW_DIR / f"{doc_id}.html").exists()
        and (TEXT_DIR / f"{doc_id}.txt").exists()
    )


def collect_page(session, url):
    response = session.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    html = response.text

    text = trafilatura.extract(
        html,
        include_links=False,
        include_images=False,
        include_tables=True,
    )

    if not text:
        raise RuntimeError(
            f"Could not extract text from {url}"
        )

    return response, html, text


def save_manifest(record):
    with MANIFEST_PATH.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def main():
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    urls = load_urls()

    print(f"Found {len(urls)} URLs.")

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
        }
    )

    for index, url in enumerate(urls, start=1):

        doc_id = make_doc_id(url)

        print()
        print(
            f"[{index}/{len(urls)}] {url}"
        )

        if already_collected(doc_id):
            print("Already collected. Skipping.")
            continue

        try:
            response, html, raw_text = collect_page(
                session,
                url,
            )

            clean_text = clean_nepali_wikipedia(
                raw_text
            )

            if not clean_text.strip():
                raise RuntimeError(
                    "Cleaned text is empty"
                )

            raw_path = (
                RAW_DIR / f"{doc_id}.html"
            )

            text_path = (
                TEXT_DIR / f"{doc_id}.txt"
            )

            raw_path.write_text(
                html,
                encoding="utf-8",
            )

            text_path.write_text(
                clean_text,
                encoding="utf-8",
            )

            record = {
                "doc_id": doc_id,
                "url": url,
                "source": "Nepali Wikipedia",
                "source_type": "wikipedia",
                "collection_type": "manual_web",
                "collection_date": datetime.now(
                    timezone.utc
                ).isoformat(),
                "http_status": response.status_code,
                "raw_html_path": str(raw_path),
                "text_path": str(text_path),
                "raw_text_characters": len(
                    raw_text
                ),
                "clean_text_characters": len(
                    clean_text
                ),
            }

            save_manifest(record)

            print(
                "Collected successfully."
            )
            print(
                "Raw characters:",
                len(raw_text),
            )
            print(
                "Clean characters:",
                len(clean_text),
            )

        except Exception as exc:
            print(
                f"FAILED: {exc}"
            )

        if index < len(urls):
            time.sleep(REQUEST_DELAY)


if __name__ == "__main__":
    main()