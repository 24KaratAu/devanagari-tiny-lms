import re


def remove_wikipedia_tables(text):
    lines = text.splitlines()

    cleaned = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Markdown table rows
        if stripped.startswith("|"):
            in_table = True
            continue

        # Leave table mode once normal prose starts
        if in_table and not stripped:
            in_table = False
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def remove_citation_markers(text):
    # Wikipedia citations such as [1], [25], [100]
    return re.sub(r"\[\d+\]", "", text)


def normalize_whitespace(text):
    # Remove trailing/leading whitespace on each line
    lines = [line.strip() for line in text.splitlines()]

    # Remove repeated blank lines
    result = []
    previous_blank = False

    for line in lines:
        if not line:
            if previous_blank:
                continue
            previous_blank = True
            result.append("")
        else:
            previous_blank = False
            result.append(line)

    return "\n".join(result).strip()


def clean_nepali_wikipedia(text):
    text = remove_wikipedia_tables(text)
    text = remove_citation_markers(text)
    text = normalize_whitespace(text)

    return text


if __name__ == "__main__":
    from pathlib import Path

    input_path = Path(
        "data/nepali/processed/manual_web/wikipedia/"
        "0d9f295172a4d8ff.txt"
    )

    output_path = Path(
        "data/nepali/processed/manual_web/wikipedia/"
        "0d9f295172a4d8ff.clean.txt"
    )

    text = input_path.read_text(encoding="utf-8")

    cleaned = clean_nepali_wikipedia(text)

    output_path.write_text(cleaned, encoding="utf-8")

    print("Original characters:", len(text))
    print("Cleaned characters:", len(cleaned))
    print("Saved:", output_path)