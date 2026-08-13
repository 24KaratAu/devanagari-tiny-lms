# this file is there to collect the first 1000 docs so that we know the hugging face pipeline is working before downloading the whole dataset

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "ai4bharat/sangraha"
DATA_DIR = "verified/nep"

OUTPUT_PATH = Path(
    "data/nepali/raw/public/sangraha_verified/"
    "sangraha_verified_nepali_pilot.jsonl"
)

NUM_DOCUMENTS = 1000


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        DATASET_NAME,
        data_dir=DATA_DIR,
        streaming=True,
    )

    stream = dataset["train"]

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, row in enumerate(
            tqdm(stream, total=NUM_DOCUMENTS, desc="Collecting Nepali")
        ):
            record = {
                "doc_id": row["doc_id"],
                "text": row["text"],
                "type": row["type"],
                "source_dataset": DATASET_NAME,
                "source_subset": DATA_DIR,
                "collection_type": "public_external",
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            if i + 1 >= NUM_DOCUMENTS:
                break

    print(f"\nSaved {NUM_DOCUMENTS} documents to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()