"""
Custom BPE Tokenizer Training

Trains a SentencePiece BPE tokenizer from scratch on the training split.
Separate tokenizer per language (no sharing between Model H and Model L).

Requirements:
    pip install sentencepiece

Usage:
    python -m src.tokenizer.train_bpe \
        --train-file data/nepali/splits/train.txt \
        --model-prefix nepali/tokenizer/nepali_bpe_8k \
        --vocab-size 8000 \
        --language nepali
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def train_tokenizer(
    train_file: str,
    model_prefix: str,
    vocab_size: int = 8000,
    language: str = "nepali",
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
    num_threads: int = 4,
    max_sentence_length: int = 16384,
    input_sentence_size: int = 5_000_000,
    shuffle_input: bool = True,
):
    """Train a SentencePiece BPE tokenizer from scratch.

    Args:
        train_file: Path to training text file (one document per paragraph)
        model_prefix: Output path prefix for .model and .vocab files
        vocab_size: Target vocabulary size
        language: Language name for logging
        character_coverage: Character coverage for BPE (0.9995 for Devanagari)
        model_type: "bpe" or "unigram"
        num_threads: CPU threads for training
        max_sentence_length: Max input sentence length in bytes
        input_sentence_size: Max sentences to sample for training
        shuffle_input: Whether to shuffle input sentences
    """
    import sentencepiece as spm

    # Ensure output directory exists
    model_dir = Path(model_prefix).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    file_size_mb = train_path.stat().st_size / (1024 * 1024)
    print(f"Training {model_type.upper()} tokenizer for {language.upper()}")
    print(f"  Training file:      {train_file} ({file_size_mb:.1f} MB)")
    print(f"  Vocab size:         {vocab_size:,}")
    print(f"  Character coverage: {character_coverage}")
    print(f"  Model type:         {model_type}")
    print(f"  Model prefix:       {model_prefix}")
    print("=" * 60)

    # Special tokens
    # pad=0, unk=1, bos=2, eos=3
    spm.SentencePieceTrainer.train(
        input=str(train_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        max_sentence_length=max_sentence_length,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        # Devanagari-specific: treat the Devanagari virama (्) and other
        # combining marks as part of the base character
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_digits=True,
        byte_fallback=True,
    )

    print(f"\nOK  Tokenizer trained successfully!")
    print(f"  Model file: {model_prefix}.model")
    print(f"  Vocab file: {model_prefix}.vocab")

    # Verify the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    actual_vocab = sp.get_piece_size()
    print(f"\nVerification:")
    print(f"  Actual vocab size: {actual_vocab:,}")
    print(f"  Special tokens: pad={sp.pad_id()}, unk={sp.unk_id()}, "
          f"bos={sp.bos_id()}, eos={sp.eos_id()}")

    # Test encoding
    if language == "nepali":
        test_texts = [
            "नेपाल एक दक्षिण एशियाली राष्ट्र हो।",
            "काठमाडौं नेपालको राजधानी हो।",
        ]
    else:
        test_texts = [
            "भारत एक दक्षिण एशियाई देश है।",
            "नई दिल्ली भारत की राजधानी है।",
        ]

    print(f"\nSample encodings:")
    total_tokens = 0
    total_chars = 0
    for text in test_texts:
        tokens = sp.encode(text, out_type=str)
        token_ids = sp.encode(text, out_type=int)
        print(f"  Text: {text}")
        print(f"  Tokens ({len(tokens)}): {tokens}")
        print(f"  IDs: {token_ids}")
        total_tokens += len(tokens)
        total_chars += len(text)
        print()

    fertility = total_tokens / max(1, len(" ".join(test_texts).split()))
    print(f"  Fertility (tokens/word): ~{fertility:.2f}")

    # Save training config
    config = {
        "language": language,
        "model_type": model_type,
        "vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab,
        "character_coverage": character_coverage,
        "train_file": str(train_file),
        "train_file_size_bytes": train_path.stat().st_size,
        "model_file": f"{model_prefix}.model",
        "vocab_file": f"{model_prefix}.vocab",
        "special_tokens": {
            "pad": {"id": sp.pad_id(), "piece": "<pad>"},
            "unk": {"id": sp.unk_id(), "piece": "<unk>"},
            "bos": {"id": sp.bos_id(), "piece": "<bos>"},
            "eos": {"id": sp.eos_id(), "piece": "<eos>"},
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    config_path = Path(f"{model_prefix}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  Config saved: {config_path}")
    return config


def tokenize_corpus(
    model_path: str,
    input_file: str,
    output_file: str,
):
    """Tokenize an entire corpus file and count tokens.

    Args:
        model_path: Path to .model file
        input_file: Input text file
        output_file: Output file with token IDs (one doc per line)
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    total_tokens = 0
    total_docs = 0
    total_chars = 0

    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Tokenizing {input_file} with {model_path}...")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        # Read paragraph-by-paragraph (documents separated by blank lines)
        current_doc = []
        for line in fin:
            if line.strip():
                current_doc.append(line.strip())
            else:
                if current_doc:
                    text = " ".join(current_doc)
                    token_ids = sp.encode(text, out_type=int)
                    fout.write(json.dumps(token_ids) + "\n")
                    total_tokens += len(token_ids)
                    total_chars += len(text)
                    total_docs += 1
                    current_doc = []

                    if total_docs % 50000 == 0:
                        print(f"  Processed {total_docs:,} docs | {total_tokens:,} tokens")

        # Handle last document
        if current_doc:
            text = " ".join(current_doc)
            token_ids = sp.encode(text, out_type=int)
            fout.write(json.dumps(token_ids) + "\n")
            total_tokens += len(token_ids)
            total_chars += len(text)
            total_docs += 1

    print(f"\nTokenization complete:")
    print(f"  Documents: {total_docs:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Chars/token: {total_chars / max(1, total_tokens):.2f}")
    print(f"  Output: {output_file}")

    return {
        "documents": total_docs,
        "total_tokens": total_tokens,
        "total_characters": total_chars,
        "chars_per_token": round(total_chars / max(1, total_tokens), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer or tokenize corpus")
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train tokenizer")
    train_parser.add_argument("--train-file", required=True, help="Training text file")
    train_parser.add_argument("--model-prefix", required=True, help="Output model prefix")
    train_parser.add_argument("--vocab-size", type=int, default=8000)
    train_parser.add_argument("--language", default="nepali")
    train_parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    train_parser.add_argument("--character-coverage", type=float, default=0.9995)

    # Tokenize subcommand
    tok_parser = subparsers.add_parser("tokenize", help="Tokenize a corpus file")
    tok_parser.add_argument("--model-path", required=True, help="Path to .model file")
    tok_parser.add_argument("--input-file", required=True)
    tok_parser.add_argument("--output-file", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train_tokenizer(
            train_file=args.train_file,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            language=args.language,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
        )
    elif args.command == "tokenize":
        tokenize_corpus(
            model_path=args.model_path,
            input_file=args.input_file,
            output_file=args.output_file,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
