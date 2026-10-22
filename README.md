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

print(tokenizer.vocabulary)  # ("a", "b", "n")
print(tokenizer.encode("banana"))  # [1, 0, 2, 0, 2, 0]
print(tokenizer.decode([1, 0, 2, 0, 2, 0]))  # banana
```

Encoding and decoding preserve character order, repetitions, whitespace, and
Unicode characters without additional normalization. Encoding an unknown
character raises `ValueError`. Decoding requires integer IDs within the
vocabulary range; negative IDs, booleans, and other invalid values raise
`ValueError`.

For the prepared corpus, construct the tokenizer from
`data/processed/train.txt` only. The current training vocabulary contains
93 characters and covers every character in the validation and test splits.

## Training windows and batches

Each window uses `T + 1` source tokens to construct `T` input IDs and
`T` next-token targets. Targets are shifted one position ahead of inputs.
For example, `"abcd"` produces inputs `"abc"` and targets `"bcd"`.

```python
import torch

from causal_transformer_lab.batching import get_window, sample_batch
from causal_transformer_lab.tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer("abcdef")
tokens = torch.tensor(tokenizer.encode("abcdef"), dtype=torch.long)

inputs, targets = get_window(tokens, start=0, context_length=3)
print(tokenizer.decode(inputs.tolist()))  # abc
print(tokenizer.decode(targets.tolist()))  # bcd

generator = torch.Generator().manual_seed(42)
inputs, targets = sample_batch(
    tokens,
    batch_size=2,
    context_length=3,
    generator=generator,
)
print(inputs.shape)  # torch.Size([2, 3])
print(targets.shape)  # torch.Size([2, 3])
```

Pass a one-dimensional CPU tensor of token IDs with dtype `torch.long`.
Batch inputs and targets have shape `(B, T)`. Sampling is uniform over valid
start positions, with replacement; windows may overlap or repeat.

Seed a generator once and reuse it across calls for a reproducible sequence
of batches. The training loop is responsible for moving batches to the
selected device.

Sample each corpus split separately. Windows may cross separators between
works within a combined split, but never cross between splits.

Both functions require a positive context length and enough source tokens
for a complete window. `get_window` also requires a valid, nonnegative
start; `sample_batch` requires a positive batch size. Invalid requests
raise `ValueError`.
