import pytest

from causal_transformer_lab.corpus import (
    combine_parts,
    extract_gutenberg_body,
    normalize_text,
    split_work,
    validate_manifest,
    verify_sha256,
)


def test_extract_gutenberg_body_removes_markers_and_surrounding_text():
    raw_text = (
        "Project Gutenberg header\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "\n"
        "First paragraph.\n"
        "\n"
        "Second paragraph.\n"
        "\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "Project Gutenberg footer\n"
    )

    result = extract_gutenberg_body(raw_text)

    assert result == "First paragraph.\n\nSecond paragraph."


def test_extract_gutenberg_body_rejects_missing_start_marker():
    raw_text = (
        "Project Gutenberg header\n"
        "Book contents\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
    )

    with pytest.raises(ValueError, match="start marker not found"):
        extract_gutenberg_body(raw_text)


def test_extract_gutenberg_body_rejects_missing_end_marker():
    raw_text = (
        "Project Gutenberg header\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "\n"
        "First paragraph.\n"
        "\n"
        "Second paragraph.\n"
        "\n"
    )

    with pytest.raises(ValueError, match="end marker not found"):
        extract_gutenberg_body(raw_text)


def test_extract_gutenberg_body_rejects_reversed_markers():
    raw_text = (
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "Book contents\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
    )

    with pytest.raises(ValueError, match="end marker must follow start marker"):
        extract_gutenberg_body(raw_text)


def test_extract_gutenberg_body_rejects_duplicate_start_markers():
    raw_text = (
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "First section\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "Second section\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
    )

    with pytest.raises(ValueError, match="multiple start markers"):
        extract_gutenberg_body(raw_text)


def test_extract_gutenberg_body_rejects_duplicate_end_markers():
    raw_text = (
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "Body\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "Footer Text\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
    )

    with pytest.raises(ValueError, match="multiple end markers"):
        extract_gutenberg_body(raw_text)


def test_normalize_text_standardizes_line_endings():
    text = "First paragraph\r\n\r\nSecond paragraph\rThird line"

    assert normalize_text(text) == "First paragraph\n\nSecond paragraph\nThird line"


def test_normalize_text_trims_boundary_whitespace():
    text = " \t\nFirst paragraph\n\nSecond paragraph\n\t "

    assert normalize_text(text) == "First paragraph\n\nSecond paragraph"


def test_normalize_text_normalizes_unicode_to_nfc():
    text = "Cafe\u0301"

    assert normalize_text(text) == "Caf\u00e9"


def test_normalize_text_preserves_internal_text():
    text = "“Holmes,”\t said  Watson.\n\n“Why?”"

    assert normalize_text(text) == text


def test_verify_sha256_rejects_mismatched_hash():
    raw_bytes = b"abc"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(raw_bytes, "0" * 64)


def test_verify_sha256_accepts_matching_hash():
    expected_sha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    assert verify_sha256(b"abc", expected_sha256) is None


def test_split_work_uses_eighty_ten_ten():
    text = "abcdefghij"

    train_text, validation_text, test_text = split_work(text)

    assert train_text == "abcdefgh"
    assert validation_text == "i"
    assert test_text == "j"


def test_split_work_rounds_boundaries_down():
    text = "abcdefghijk"

    train_text, validation_text, test_text = split_work(text)

    assert train_text == "abcdefgh"
    assert validation_text == "i"
    assert test_text == "jk"


def test_combine_parts_preserves_order_and_adds_separators():
    parts = ["First work.", "Second work.", "Third work."]

    assert combine_parts(parts) == "First work.\n\nSecond work.\n\nThird work.\n"


def test_validate_manifest_rejects_unsupported_schema():
    manifest = {
        "schema_version": 2,
        "works": [
            {
                "order": 1,
                "key": "sample",
                "raw_filename": "sample.txt",
                "source_sha256": "0" * 64,
            }
        ],
    }

    with pytest.raises(ValueError, match="Unsupported source manifest schema version"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_empty_works():
    manifest = {
        "schema_version": 1,
        "works": [],
    }

    with pytest.raises(ValueError, match="works must be a non-empty list"):
        validate_manifest(manifest)


def test_validate_manifest_accepts_valid_manifest():
    manifest = {
        "schema_version": 1,
        "works": [
            {
                "order": 1,
                "key": "sample",
                "raw_filename": "sample.txt",
                "source_sha256": "0" * 64,
            }
        ],
    }

    assert validate_manifest(manifest) is None


# Tells pytest to run the test four times, assigning a different field name each time
@pytest.mark.parametrize(
    "missing_field",
    ["order", "key", "raw_filename", "source_sha256"],
)
def test_validate_manifest_rejects_missing_work_fields(missing_field):
    manifest = {
        "schema_version": 1,
        "works": [
            {
                "order": 1,
                "key": "sample",
                "raw_filename": "sample.txt",
                "source_sha256": "0" * 64,
            }
        ],
    }

    # Delete the selected field
    del manifest["works"][0][missing_field]

    with pytest.raises(ValueError, match="missing required field"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_non_dictionary_work():
    manifest = {
        "schema_version": 1,
        "works": [None],
    }

    with pytest.raises(ValueError, match="work must be a dictionary"):
        validate_manifest(manifest)


@pytest.mark.parametrize("invalid_order", ["1", 1.5, 0, -1, None, True])
def test_validate_manifest_rejects_invalid_order(invalid_order):
    manifest = {
        "schema_version": 1,
        "works": [
            {
                "order": invalid_order,
                "key": "sample",
                "raw_filename": "sample.txt",
                "source_sha256": "0" * 64,
            }
        ],
    }

    with pytest.raises(ValueError, match="order must be a positive integer"):
        validate_manifest(manifest)
