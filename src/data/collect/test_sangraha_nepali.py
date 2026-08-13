from datasets import load_dataset

DATASET = "ai4bharat/sangraha"
DATA_DIR = "verified/nep"


def main():
    print("Loading Nepali Sangraha stream...")

    ds = load_dataset(
        DATASET,
        data_dir=DATA_DIR,
        streaming=True,
    )

    print("\nDataset:")
    print(ds)

    if hasattr(ds, "keys"):
        print("\nSplits:")
        print(list(ds.keys()))

        split_name = next(iter(ds.keys()))
        data = ds[split_name]
    else:
        split_name = None
        data = ds

    print("\nInspecting first 3 documents...\n")

    for i, row in enumerate(data):
        print("=" * 80)
        print(f"DOCUMENT {i + 1}")
        print("Fields:", row.keys())

        for key, value in row.items():
            print(f"\n--- {key} ---")

            if isinstance(value, str):
                print(value[:2000])
            else:
                print(value)

        if i >= 2:
            break


if __name__ == "__main__":
    main()