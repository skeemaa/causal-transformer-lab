"""Utilities for reproducible corpus preparation before tokenization.

Provides Project Gutenberg body extraction, text normalization, raw-byte
SHA-256 verification, contiguous 80/10/10 splits, and ordered text assembly.
"""

import hashlib
import unicodedata

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def extract_gutenberg_body(raw_text: str) -> str:
    """Extract book text, rejecting missing, duplicate, or reversed markers."""

    start_index = None
    end_index = None

    # List of strings from raw_text split based on line-breaks
    lines = raw_text.splitlines()

    # Loop through list and store index of start marker
    for index, line in enumerate(lines):
        if line.startswith(START_MARKER):
            # Reject multiple start markers
            if start_index is not None:
                raise ValueError("Project Gutenberg multiple start markers found.")

            start_index = index

    # Reject inputs that are missing start marker
    if start_index is None:
        raise ValueError("Project Gutenberg start marker not found.")

    # Loop through list and store index of end marker
    for index, line in enumerate(lines):
        if line.startswith(END_MARKER):
            # Reject multiple end markers
            if end_index is not None:
                raise ValueError("Project Gutenberg multiple end markers found.")

            end_index = index

    # Reject inputs that are missing end marker
    if end_index is None:
        raise ValueError("Project Gutenberg end marker not found.")

    # Reject inputs that have swapped start and end markers
    if end_index <= start_index:
        raise ValueError("Project Gutenberg end marker must follow start marker.")

    # List of lines in between the start marker and end marker
    body_lines = lines[start_index + 1 : end_index]

    # Reassemble the lines into one string
    body_text = "\n".join(body_lines)

    # Remove leading and trailing whitespace, preserving internal whitespace
    return body_text.strip()


def normalize_text(text: str) -> str:
    """Normalize line endings and Unicode NFC, then trim boundary whitespace."""

    # Replace both \r\n and \r with \n
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Apply Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    return text.strip()


def verify_sha256(raw_bytes: bytes, expected_sha256: str) -> None:
    """Raise ValueError if raw bytes do not match the expected SHA-256."""

    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    if actual_sha256 != expected_sha256:
        raise ValueError("SHA-256 mismatch.")


def split_work(text: str) -> tuple[str, str, str]:
    """Split one work into contiguous training, validation, and test portions."""

    # Round the 80% and 90% boundaries down to whole-character indices.
    train_end = len(text) * 8 // 10
    validation_end = len(text) * 9 // 10

    return (
        text[:train_end],
        text[train_end:validation_end],
        text[validation_end:],
    )


def combine_parts(parts: list[str]) -> str:
    """Combine text portions in supplied order with paragraph separators."""

    combined_works = "\n\n".join(parts)

    return combined_works + "\n"


def validate_manifest(manifest: dict) -> None:
    """Validate the source manifest's supported schema and work records."""

    # Get schema version and reject if not 1
    schema_version = manifest.get("schema_version")

    if schema_version != 1:
        raise ValueError("Unsupported source manifest schema version.")

    # Get works and reject if the value is not a list, an empty list
    works = manifest.get("works")

    if not isinstance(works, list) or not works:
        raise ValueError("Source manifest works must be a non-empty list.")

    # Ensure each work has the required fields
    required_fields = ("order", "key", "raw_filename", "source_sha256")

    for work in works:
        # Ensure each work is a dictionary
        if not isinstance(work, dict):
            raise ValueError("Source manifest work must be a dictionary.")

        # Ensure each work has all the required fields
        for field in required_fields:
            if field not in work:
                raise ValueError(
                    f"Source manifest work is missing required field: {field}."
                )

        # Ensure each works order is a positive integer
        order = work["order"]
        if type(order) is not int or order <= 0:
            raise ValueError("Source manifest work order must be a positive integer")
