"""Prepare reproducible corpus files from recorded Gutenberg sources."""

import hashlib
import json
from pathlib import Path

from causal_transformer_lab.corpus import (
    combine_parts,
    extract_gutenberg_body,
    normalize_text,
    split_work,
    validate_manifest,
    verify_sha256,
)


def prepare_corpus(project_root: Path) -> None:
    """Prepare corpus files from sources under the supplied project root."""

    # Point to sources.json file
    manifest_path = project_root / "data" / "sources.json"

    # Read the .json file as a UTF-8 string and store in manifest_text
    manifest_text = manifest_path.read_text(encoding="utf-8")

    # Convert manifest_text to Python objects
    manifest = json.loads(manifest_text)

    # Validate manifest
    validate_manifest(manifest)

    # Process the works in their recorded order, verify each raw file’s hash,
    # extract, normalize, and split each work
    works = sorted(manifest["works"], key=lambda work: work["order"])

    train_parts = []
    validation_parts = []
    test_parts = []
    work_metadata = []

    for work in works:
        raw_path = project_root / "data" / "raw" / work["raw_filename"]
        raw_bytes = raw_path.read_bytes()

        verify_sha256(raw_bytes, work["source_sha256"])

        raw_text = raw_bytes.decode("utf-8-sig")
        body_text = extract_gutenberg_body(raw_text)
        normalized_text = normalize_text(body_text)
        train_text, validation_text, test_text = split_work(normalized_text)

        # Store work metadata
        work_record = {
            "order": work["order"],
            "key": work["key"],
            "source_sha256": work["source_sha256"],
            "normalized_character_count": len(normalized_text),
            "train_end": len(train_text),
            "validation_end": len(train_text) + len(validation_text),
            "train_character_count": len(train_text),
            "validation_character_count": len(validation_text),
            "test_character_count": len(test_text),
        }

        # Append the work record to the metadata list
        work_metadata.append(work_record)

        # Append the splits for each work
        train_parts.append(train_text)
        validation_parts.append(validation_text)
        test_parts.append(test_text)

    # Create the full train, validation, and test sets
    train_corpus = combine_parts(train_parts)
    validation_corpus = combine_parts(validation_parts)
    test_corpus = combine_parts(test_parts)

    # Create the directory to store our processed sets
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create a dictionary to connect a split's name to its text
    corpora = {
        "train": train_corpus,
        "validation": validation_corpus,
        "test": test_corpus,
    }

    # Create a dictionary to store output metadata
    output_metadata = {}

    # Iterate through the corpora dictionary, convert text to bytes,
    # and write to .txt files
    for split_name, corpus_text in corpora.items():
        corpus_bytes = corpus_text.encode("utf-8")
        output_path = processed_dir / f"{split_name}.txt"
        output_path.write_bytes(corpus_bytes)

        # Create a sorted list of all unique characters used in this corpus
        vocabulary = sorted(set(corpus_text))  # set() removes duplicates

        # Store metadata
        output_metadata[split_name] = {
            "filename": output_path.name,
            "character_count": len(corpus_text),
            "byte_count": len(corpus_bytes),
            "sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "vocabulary_size": len(vocabulary),
            "vocabulary": vocabulary,
        }

    # Create metadata dictionary
    metadata = {
        "schema_version": 1,
        "sources_manifest": "data/sources.json",
        "source_encoding": "utf-8-sig",
        "output_encoding": "utf-8",
        "normalization": {
            "line_endings": "LF",
            "unicode_form": "NFC",
            "strip_boundary_whitespace": True,
        },
        "split": {
            "unit": "characters",
            "per_work": True,
            "train_end": "N * 8 // 10",
            "validation_end": "N * 9 // 10",
        },
        "combination": {
            "separator": "\n\n",
            "appended_suffix": "\n",
        },
        "works": work_metadata,
        "outputs": output_metadata,
    }

    # Convert metadata dictionary into JSON text
    metadata_text = json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    # Write metadata file
    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_bytes((metadata_text + "\n").encode("utf-8"))

    print(f"Prepared {len(work_metadata)} works in {processed_dir}.")


def main() -> None:
    """Run corpus preparation from the repository root."""
    project_root = Path(__file__).resolve().parents[1]
    prepare_corpus(project_root)


if __name__ == "__main__":
    main()
