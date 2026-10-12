# causal-transformer-lab

A tested, modular decoder-only transformer implemented with PyTorch tensor operations to explore attention, training, and autoregressive generation from first principles.

## Documentation

- [Project design](docs/design.md) — Approved architecture, implementation boundaries, testing strategy, and reproducibility requirements

- [Corpus preparation](data/README.md) - Corpus preparation workflow, sources, and rules

## Character tokenizer

`CharacterTokenizer` builds a sorted, unique character vocabulary from training
text. Vocabulary positions define zero-based token IDs. Use the same tokenizer
for training, validation, and test text.

```python
from causal_transformer_lab.tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer("banana")

print(tokenizer.vocabulary)                    # ("a", "b", "n")
print(tokenizer.encode("banana"))              # [1, 0, 2, 0, 2, 0]
print(tokenizer.decode([1, 0, 2, 0, 2, 0]))    # banana
```

Encoding and decoding preserve character order, repetitions, whitespace, and
Unicode characters without additional normalization. Encoding an unknown
character raises `ValueError`. Decoding requires integer IDs within the
vocabulary range; negative IDs, booleans, and other invalid values raise
`ValueError`.

For the prepared corpus, construct the tokenizer from
`data/processed/train.txt` only. The current training vocabulary contains
93 characters and covers every character in the validation and test splits.