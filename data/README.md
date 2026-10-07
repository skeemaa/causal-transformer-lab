# Corpus Preparation

This directory contains the source manifest and prepared character-level
corpus for causal-transformer-lab. Run all commands below from the repository
root with the project environment activated.

## Sources and Provenance

The corpus uses these works by Arthur Conan Doyle, in this fixed order:

| Order | Work | Gutenberg ebook | Recorded update date |
|---|---|---|---|
| 1 | The Adventures of Sherlock Holmes | 1661 | 2023-10-10 |
| 2 | A Study in Scarlet | 244 | 2026-09-06 |
| 3 | The Hound of the Baskervilles | 2852 | 2025-08-04 |

[sources.json](sources.json) records the retrieval date of 2026-09-25,
landing pages, download URLs, filenames, exact source SHA-256 hashes,
and recorded public-domain status in the USA. It also records the
[Project Gutenberg license page](https://www.gutenberg.org/policy/license).

Raw downloads belong in `data/raw/`, which Git ignores. Prepared files
under `data/processed/` are intended to be committed.

## Preparation Rules

For each work, preparation:

1. Verifies the SHA-256 of the original bytes before decoding.
2. Decodes using `utf-8-sig`, accepting an optional UTF-8 byte-order mark.
3. Extracts lines strictly between the Gutenberg start and end markers.
   Missing, duplicate, or reversed markers are rejected.
4. Standardizes line endings to LF, applies Unicode NFC normalization,
   and strips boundary whitespace.

Case, punctuation, accents, and internal whitespace are preserved.
Titles, contents pages, and other material inside the markers remain.

Each normalized work is split independently and contiguously.
For `N` characters:

- Training: `text[:N * 8 // 10]`
- Validation: `text[N * 8 // 10:N * 9 // 10]`
- Test: `text[N * 9 // 10:]`

Integer boundaries round down, so proportions are approximately 80/10/10.
Every normalized character belongs to exactly one portion.

Matching portions are combined in recorded source order, with two LF
characters between works and one final LF. Portions are not trimmed again
after splitting. With three works, assembly adds five characters per output.

## Generated Files

Outputs use UTF-8 without a byte-order mark and LF line endings.

| File | Characters | Bytes | Unique characters |
|---|---:|---:|---:|
| train.txt | 924,603 | 945,536 | 93 |
| validation.txt | 115,580 | 117,903 | 78 |
| test.txt | 115,582 | 117,468 | 77 |

`processed/metadata.json` records preparation settings, per-work counts
and split boundaries, and each output's filename, counts, SHA-256,
and sorted character vocabulary. It contains no runtime timestamp,
allowing identical inputs to produce identical output bytes.

Validation guides development and checkpoint selection. Test evaluation
is reserved until model choices are settled. These splits assess held-out
portions of the same collection, rather than generalization to new authors.

## Reproducing the Corpus

Place the exact recorded source files in `data/raw/`, using the filenames
in `sources.json`, then run:

```bash
python scripts/prepare_corpus.py
