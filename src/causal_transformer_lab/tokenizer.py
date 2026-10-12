"""Tokenizer and utilities.

Provides character level tokenization via encoder and decoder.
"""


class CharacterTokenizer:
    """Map characters to deterministic token IDs."""

    def __init__(self, training_text: str) -> None:
        """Build a sorted vocabulary from training text."""
        self.vocabulary = tuple(sorted(set(training_text)))

        self._character_to_id = {
            character: token_id
            # For each (token_id, character) pair, create an entry with
            # the character as the key and the ID as the value
            for token_id, character in enumerate(self.vocabulary)
        }

    def encode(self, text: str) -> list[int]:
        """Encode text, rejecting characters outside the vocabulary."""
        try:
            return [self._character_to_id[character] for character in text]
        except KeyError as error:
            character = error.args[0]
            raise ValueError(f"Unknown character: {character!r}") from error

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into text."""
        characters = []

        # Check each id for validity
        for token_id in ids:
            if type(token_id) is not int or not (0 <= token_id < len(self.vocabulary)):
                raise ValueError(f"Invalid token ID: {token_id!r}")

            characters.append(self.vocabulary[token_id])

        # Combine characters into one string
        return "".join(characters)
