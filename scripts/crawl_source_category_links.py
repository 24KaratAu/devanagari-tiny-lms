"""
Category Link Crawling Script to Expand Source URL Lists for Batch 1 (~1,000 documents)

Fetches article permalinks from Nepali Wikipedia category pages across:
  - Wikipedia (geography, history, science, fauna, flora, biographies, culture)
  - News (newspapers, journalism, media, broadcasting)
  - Government (ministries, law, constitution, courts, commissions)
  - Education (universities, colleges, schools, education boards)
  - Literature (writers, poets, books, awards, genres)

Appends clean article URLs to data/nepali/sources/*.txt.
"""

import re
import urllib.parse
from pathlib import Path
import requests

SOURCES_DIR = Path("data/nepali/sources")
USER_AGENT = "24KaratAu-LMA-Corpus/1.0"

CATEGORY_MAP = {
    "wikipedia_urls.txt": [
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_जिल्लाहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_नदीहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालको_इतिहास",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_हिमालहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_तालहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपाली_व्यक्तिहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_राष्ट्रिय_निकुञ्जहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_नगरपालिकाहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_गाउँपालिकाहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_चाडपर्वहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालको_भूगोल",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालमा_जलविद्युत्",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_मन्दिरहरू",
    ],
    "news_urls.txt": [
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_समाचारपत्रहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपाली_पत्रकारहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालमा_पत्रकारिता",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_रेडियो_स्टेसनहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_टेलिभिजन_च्यानलहरू",
    ],
    "government_urls.txt": [
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_प्रधानमन्त्रीहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_मन्त्रालयहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालको_कानुन",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_संवैधानिक_निकायहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_अदालतहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_राजदूतहरू",
    ],
    "education_urls.txt": [
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_विश्वविद्यालयहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालका_क्याम्पसहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपालमा_शिक्षा",
        "https://ne.wikipedia.org/wiki/श्रेणी:त्रिभुवन_विश्वविद्यालयका_क्याम्पसहरू",
    ],
    "literature_urls.txt": [
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपाली_साहित्यकारहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपाली_कविहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपाली_उपन्यासकारहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपाली_पुस्तकहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:मदन_पुरस्कार_विजेताहरू",
        "https://ne.wikipedia.org/wiki/श्रेणी:नेपाली_साहित्य",
    ],
}


def crawl_category_links():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    print("Crawling Wikipedia category index pages for high-quality article permalinks...")

    for file_name, cat_urls in CATEGORY_MAP.items():
        file_path = SOURCES_DIR / file_name
        existing_lines = []
        if file_path.exists():
            existing_lines = [l.strip() for l in file_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        collected_urls = set(l for l in existing_lines if not l.startswith("#"))
        new_count = 0

        for cat_url in cat_urls:
            try:
                res = session.get(cat_url, timeout=20)
                if res.status_code != 200:
                    continue

                # Find all /wiki/ article links in HTML
                matches = re.findall(r'href="(/wiki/[^":#?]+)"', res.text)
                for m in matches:
                    decoded = urllib.parse.unquote(m)
                    # Ignore special namespaces like श्रेणी:, ढाँचा:, सहायता:, मिडियाWiki:
                    if any(prefix in decoded for prefix in ["श्रेणी:", "ढाँचा:", "सहायता:", "मिडियाWiki:", "विशेष:", "वार्ता:", "प्रयोगपृष्ठ:"]):
                        continue
                    
                    full_url = f"https://ne.wikipedia.org{decoded}"
                    if full_url not in collected_urls:
                        collected_urls.add(full_url)
                        new_count += 1

            except Exception as e:
                print(f"  Warning: failed to crawl {cat_url}: {e}")

        # Save expanded list
        header_comment = existing_lines[0] if existing_lines and existing_lines[0].startswith("#") else f"# Nepali {file_name.split('_')[0].capitalize()} URLs"
        clean_urls = sorted(list(collected_urls))
        content = header_comment + "\n" + "\n".join(clean_urls) + "\n"
        file_path.write_text(content, encoding="utf-8")
        print(f"  {file_name:25s}: {len(clean_urls)} total URLs (added {new_count} new links).")

    print("\nSource list expansion complete!")


if __name__ == "__main__":
    crawl_category_links()
