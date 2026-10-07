import hashlib
import json

import pytest

from scripts.prepare_corpus import prepare_corpus


# Helper to create a synthetic Gutenberg source
def write_sample_source(project_root, *, order, key, body):
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_text = (
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\r\n"
        f"{body}\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\r\n"
    )
    raw_bytes = raw_text.encode("utf-8-sig")
    filename = f"{key}.txt"
    (raw_dir / filename).write_bytes(raw_bytes)

    return {
        "order": order,
        "key": key,
        "raw_filename": filename,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def test_prepare_corpus_writes_expected_outputs(tmp_path):
    # Create first sample book
    first = write_sample_source(tmp_path, order=1, key="first", body="abcdefghij")

    # Create a second sample book
    second = write_sample_source(tmp_path, order=2, key="second", body="ABCDEFGHIJ")

    # Create manifest using sample books
    manifest = {
        "schema_version": 1,
        "works": [second, first],
    }
    # Write manifest to temp location
    manifest_path = tmp_path / "data" / "sources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Run the prepare script
    prepare_corpus(tmp_path)

    # Script should return the following train/val/test split
    processed_dir = tmp_path / "data" / "processed"
    expected_outputs = {
        "train": "abcdefgh\n\nABCDEFGH\n",
        "validation": "i\n\nI\n",
        "test": "j\n\nJ\n",
    }

    # Construct the path to the JSON file created by prepare_corpus
    metadata_path = processed_dir / "metadata.json"

    # Read the file as text, then convert JSON into Python dictionaries and lists
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    for split_name, expected_text in expected_outputs.items():
        # Check the generated file against our explicitly expected text
        output_path = processed_dir / f"{split_name}.txt"
        assert output_path.read_bytes() == expected_text.encode("utf-8")

        # Retrieve the metadata dictionary for this particular split
        record = metadata["outputs"][split_name]

        # Calculate expected bytes and vocabulary from our expected text
        expected_bytes = expected_text.encode("utf-8")
        expected_vocabulary = sorted(set(expected_text))

        # Check the metadata
        assert record["filename"] == f"{split_name}.txt"
        assert record["character_count"] == len(expected_text)
        assert record["byte_count"] == len(expected_bytes)
        assert record["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
        assert record["vocabulary"] == expected_vocabulary
        assert record["vocabulary_size"] == len(expected_vocabulary)

    # Collect work keys in their recorded metadata order
    work_keys = [work["key"] for work in metadata["works"]]

    # The input manifest listed second first, but metadata should follow order
    assert work_keys == ["first", "second"]

    # Each synthetic book contains exactly ten normalized characters
    for work in metadata["works"]:
        assert work["normalized_character_count"] == 10

        # Training stops at index 8; validation stops at index 9
        assert work["train_end"] == 8
        assert work["validation_end"] == 9

        # Each book contributes 8 training, 1 validation, and 1 test character
        assert work["train_character_count"] == 8
        assert work["validation_character_count"] == 1
        assert work["test_character_count"] == 1

        # Save the exact bytes of all four generated files.
    filenames = ("train.txt", "validation.txt", "test.txt", "metadata.json")
    original_bytes = {}

    for filename in filenames:
        original_bytes[filename] = (processed_dir / filename).read_bytes()

    # Prepare the same sources again.
    prepare_corpus(tmp_path)

    # Check that every file is byte-for-byte identical.
    for filename in filenames:
        assert (processed_dir / filename).read_bytes() == original_bytes[filename]


def test_prepare_corpus_rejects_hash_mismatch(tmp_path):
    # Create first sample book
    first = write_sample_source(tmp_path, order=1, key="first", body="abcdefghij")

    # Create manifest using sample book
    manifest = {
        "schema_version": 1,
        "works": [first],
    }

    # Write the manifest containing the original source hash
    manifest_path = tmp_path / "data" / "sources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Replace the source contents after its original hash was recorded
    raw_path = tmp_path / "data" / "raw" / first["raw_filename"]
    raw_path.write_bytes(b"changed source contents")

    # Preparation must reject the changed bytes
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_corpus(tmp_path)
