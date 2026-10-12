"""Tests for deterministic character tokenization."""

import pytest

from causal_transformer_lab.tokenizer import CharacterTokenizer


def test_vocabulary_is_sorted_and_unique():
    tokenizer = CharacterTokenizer("banana")

    assert tokenizer.vocabulary == ("a", "b", "n")


def test_encode_preserves_character_order_and_repetitions():
    tokenizer = CharacterTokenizer("banana")

    assert tokenizer.encode("banana") == [1, 0, 2, 0, 2, 0]


def test_decode_reconstructs_text():
    tokenizer = CharacterTokenizer("banana")

    assert tokenizer.decode([1, 0, 2, 0, 2, 0]) == "banana"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "banana",
        "Hello, world!\n",
        "  repeated spaces  ",
        "café — tea",
    ],
)
def test_encode_decode_round_trip(text):
    tokenizer = CharacterTokenizer("banana\nHello, world!  repeated spaces  café — tea")

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_rejects_unknown_character():
    tokenizer = CharacterTokenizer("banana")

    with pytest.raises(ValueError, match="Unknown character"):
        tokenizer.encode("ban!")


@pytest.mark.parametrize("invalid_id", [-1, 3, 1.5, "1", None, True])
def test_decode_rejects_invalid_id(invalid_id):
    tokenizer = CharacterTokenizer("banana")

    with pytest.raises(ValueError, match="Invalid token ID"):
        tokenizer.decode([invalid_id])


def test_mapping_is_independent_of_training_character_order():
    first = CharacterTokenizer("banana")
    second = CharacterTokenizer("nnbaa")

    assert first.vocabulary == second.vocabulary
    assert first.encode("nab") == [2, 0, 1]
    assert second.encode("nab") == [2, 0, 1]
